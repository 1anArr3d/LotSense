"""
data/collector.py — Facebook Marketplace listing fetcher (Playwright).

Launches a headless Chrome browser with your Facebook session cookies, navigates
to the Marketplace search page, and captures GraphQL responses
(CometMarketplaceSearchContentPaginationQuery) as they stream in. Scrolling
triggers FB's infinite-scroll pagination without fighting bot detection.

Raises FacebookAuthError when the page redirects to login.

Price note: listing_price.amount is a dollar string ("5900.00") → price_cents.
Mileage note: pre-parsed by FB as "78K miles" in custom_sub_titles_with_rendering_flags.
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Iterator
from urllib.parse import urlencode

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv()

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/147.0.0.0 Safari/537.36"
)

_QUERY_NAME = "CometMarketplaceSearchContentPaginationQuery"


class FacebookAuthError(Exception):
    """Session cookies are expired or invalid. Re-copy from browser DevTools."""


@dataclass
class RawListing:
    fb_listing_id: str
    title: str
    price_cents: int | None      # round(float(listing_price.amount) * 100)
    city: str | None
    state: str | None
    photo_url: str | None        # CDN URL (expires)
    creation_time: int           # unix epoch from FB
    mileage_raw: str | None      # e.g. "78K miles" — normalised in parser.py
    is_sold: bool
    seller_name: str | None
    raw: dict = field(repr=False)


class FacebookMarketplaceCollector:

    def __init__(self) -> None:
        self._cookies = _load_cookies()

    def _playwright_cookies(self) -> list[dict]:
        return [
            {"name": name, "value": value, "domain": ".facebook.com", "path": "/"}
            for name, value in self._cookies.items()
        ]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ndjson(text: str) -> list[dict]:
        chunks = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # FB prefixes responses with "for (;;);" as a JSON-hijacking guard
            if line.startswith("for (;;);"):
                line = line[len("for (;;);"):]
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return chunks

    @staticmethod
    def _extract_edges(chunks: list[dict]) -> list[dict]:
        """Return MarketplaceFeedListingStoryObject nodes from parsed NDJSON chunks."""
        nodes = []
        for chunk in chunks:
            feed_units = (
                (chunk.get("data") or {})
                .get("marketplace_search") or {}
            ).get("feed_units", {})
            for edge in feed_units.get("edges", []):
                node = edge.get("node", {})
                if node.get("__typename") == "MarketplaceFeedListingStoryObject":
                    nodes.append(node)
        return nodes

    @staticmethod
    def _node_to_listing(node: dict) -> RawListing:
        listing = node.get("listing") or {}
        loc = (listing.get("location") or {}).get("reverse_geocode") or {}

        price_raw = (listing.get("listing_price") or {}).get("amount")
        try:
            price_cents = round(float(price_raw) * 100) if price_raw is not None else None
        except (ValueError, TypeError):
            price_cents = None

        subtitles = listing.get("custom_sub_titles_with_rendering_flags") or []
        mileage_raw = subtitles[0].get("subtitle") if subtitles else None

        primary_photo = listing.get("primary_listing_photo") or {}
        photo_url = (
            (primary_photo.get("image") or {}).get("uri")
            or (primary_photo.get("photo") or {}).get("image", {}).get("uri")
        )

        seller = listing.get("marketplace_listing_seller") or {}
        seller_name = seller.get("name") or seller.get("display_name")

        return RawListing(
            fb_listing_id=listing.get("id", ""),
            title=listing.get("marketplace_listing_title", ""),
            price_cents=price_cents,
            city=loc.get("city"),
            state=loc.get("state"),
            photo_url=photo_url,
            creation_time=listing.get("creation_time", 0),
            mileage_raw=mileage_raw,
            is_sold=listing.get("is_sold", False),
            seller_name=seller_name,
            raw=node,
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _is_junk(listing: "RawListing", query_terms: set[str] | None = None) -> bool:
        """
        Drop listings that are clearly not whole-car sales:
          - no price or mileage
          - odometer > 300,000 (999K miles parting-out trick)
          - asking price < $2,500 (bait listings)
          - title doesn't contain all query words (wrong make/model recommendations)
        """
        from data.parser import parse_mileage
        if listing.price_cents is None:
            return True
        miles = parse_mileage(listing.mileage_raw)
        if miles is None:
            return True
        if miles > 300_000:
            return True
        if listing.price_cents < 250_000:
            return True
        if query_terms:
            title_lower = listing.title.lower()
            if not all(term in title_lower for term in query_terms):
                return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        max_pages: int = 10,
        query: str = "Vehicles",
        debug: bool = False,
        min_price: int | None = None,
        max_price: int | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        locations: list[str] | None = None,
    ) -> Iterator[RawListing]:
        """
        Yield RawListing objects from a Marketplace search.

        locations: list of FB Marketplace city slugs, e.g. ["sanantonio", "austin",
                   "houston"]. Each is fetched separately and deduplicated by listing ID.
                   Omit to use your account's default location.
        """
        query_terms = {w.lower() for w in query.split() if len(w) > 1}
        search_locations = locations or [None]

        params = {"query": query, "vehicleType": "car_truck"}
        if min_price is not None:
            params["minPrice"] = str(min_price)
        if max_price is not None:
            params["maxPrice"] = str(max_price)
        if min_year is not None:
            params["minYear"] = str(min_year)
        if max_year is not None:
            params["maxYear"] = str(max_year)
        qs = urlencode(params)

        all_listings: list[RawListing] = []
        for location in search_locations:
            if location:
                url = f"https://www.facebook.com/marketplace/{location}/search?{qs}"
            else:
                url = f"https://www.facebook.com/marketplace/search?{qs}"

            if debug:
                print(f"\n  [location: {location or 'default'}]")

            all_listings.extend(self._collect_one(url, max_pages, query_terms, debug))

        # Final dedup — greenlet switches inside on_response can cause duplicates
        # within a session; cross-location overlap is also caught here.
        seen: set[str] = set()
        for listing in all_listings:
            if listing.fb_listing_id not in seen:
                seen.add(listing.fb_listing_id)
                yield listing

    def _collect_one(
        self,
        url: str,
        max_pages: int,
        query_terms: set[str],
        debug: bool,
    ) -> list[RawListing]:
        listings: list[RawListing] = []
        page_count = 0

        def on_response(response) -> None:
            nonlocal page_count
            if "/api/graphql/" not in response.url:
                return
            if _QUERY_NAME not in (response.request.post_data or ""):
                return
            try:
                text = response.text()
            except Exception:
                return
            chunks = FacebookMarketplaceCollector._parse_ndjson(text)
            nodes = FacebookMarketplaceCollector._extract_edges(chunks)
            if not nodes:
                return
            page_count += 1
            if debug:
                junk = sum(
                    1 for n in nodes
                    if FacebookMarketplaceCollector._is_junk(
                        FacebookMarketplaceCollector._node_to_listing(n), query_terms,
                    )
                )
                print(f"  [page {page_count}] raw_nodes={len(nodes)}  junk={junk}")
            for node in nodes:
                listing = FacebookMarketplaceCollector._node_to_listing(node)
                if not FacebookMarketplaceCollector._is_junk(listing, query_terms):
                    listings.append(listing)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=_USER_AGENT,
                viewport={"width": 1280, "height": 900},
            )
            context.add_cookies(self._playwright_cookies())

            pw_page = context.new_page()
            pw_page.on("response", on_response)
            pw_page.goto(url, wait_until="domcontentloaded", timeout=60000)

            if "/login" in pw_page.url:
                browser.close()
                raise FacebookAuthError(
                    "Session cookies are expired.\n"
                    "  1. Log in to facebook.com in your browser\n"
                    "  2. DevTools → Application → Cookies → facebook.com\n"
                    "  3. Copy datr, sb, c_user, fr, xs into .env"
                )

            pw_page.wait_for_timeout(4000)

            for _ in range(max_pages - 1):
                if page_count >= max_pages:
                    break
                pw_page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                pw_page.wait_for_timeout(int(random.uniform(3000, 5000)))

            browser.close()

        return listings


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_cookies() -> dict:
    required = ["FB_DATR", "FB_SB", "FB_C_USER", "FB_FR", "FB_XS"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required env vars: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your Facebook session cookies."
        )
    return {
        "datr": os.environ["FB_DATR"],
        "sb": os.environ["FB_SB"],
        "c_user": os.environ["FB_C_USER"],
        "fr": os.environ["FB_FR"],
        "xs": os.environ["FB_XS"],
        "ps_l": "1",
        "ps_n": "1",
        "locale": "en_US",
    }
