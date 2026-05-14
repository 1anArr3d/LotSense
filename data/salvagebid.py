"""
data/salvagebid.py — SalvageBid auction listing fetcher.

Uses curl_cffi to handle Cloudflare without session cookies.
Targets Texas-only salvage/auction inventory (~5 600 active lots).

Field mapping:
    id                → lot_number
    year/make/model   → year/make/model
    vin               → vin
    odometer_value    → mileage  (Actual status only by default)
    retail_value      → retail_estimate (dollars → cents)
    current_bid_value → hammer_price if already_sold else 0 (dollars → cents)
    damage            → damage_description
    location_city/state → city/state
"""

import time
import random
from typing import Iterator

from curl_cffi import requests as cffi_requests

from data.parser import AuctionListing

_SEARCH_URL = "https://www.salvagebid.com/rest-api/v1.0/lots/search"
_SEARCH_PATH = "/search/auction-only/location-state-texas"
_PAGE_SIZE = 100

_HEADERS = {
    "accept": "application/json",
    "referer": "https://www.salvagebid.com/search/auction-only/location-state-texas",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/147.0.0.0 Safari/537.36"
    ),
}


class SalvageBidCollector:

    def __init__(self):
        self._session = cffi_requests.Session(impersonate="chrome120")

    def search(
        self,
        max_pages: int | None = None,
        actual_odometer_only: bool = True,
        debug: bool = False,
    ) -> Iterator[AuctionListing]:
        """
        Yield AuctionListing objects for all Texas auction lots on SalvageBid.

        Set actual_odometer_only=False to include lots with non-verified
        odometer readings (TMU, inoperable dash, exempt).
        """
        seen: set[str] = set()
        page = 1

        while True:
            resp = self._session.get(
                _SEARCH_URL,
                params={
                    "search_path": _SEARCH_PATH,
                    "page": page,
                    "per_page": _PAGE_SIZE,
                    "_locale": "en",
                },
                headers=_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            pagination = data.get("pagination", {})
            lots = data.get("lots") or []
            last_page = pagination.get("last_page", page)

            if debug:
                total = pagination.get("total", "?")
                print(f"  [page {page}/{last_page}] {len(lots)} lots  ({total} total)")

            if not lots:
                break

            for lot in lots:
                listing = _lot_to_listing(lot, actual_odometer_only)
                if listing and listing.lot_number not in seen:
                    seen.add(listing.lot_number)
                    yield listing

            if max_pages and page >= max_pages:
                break
            if page >= last_page:
                break

            page += 1
            time.sleep(random.uniform(0.5, 1.0))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _lot_to_listing(lot: dict, actual_only: bool) -> AuctionListing | None:
    odo_status = lot.get("odometer_status") or ""
    if actual_only and odo_status != "Actual":
        return None

    mileage = lot.get("odometer_value")
    if not mileage or int(mileage) <= 0:
        return None

    make = (lot.get("make") or "").title()
    model = (lot.get("model") or "").title()
    year = lot.get("year")
    if not all([make, model, year]):
        return None

    vin = lot.get("vin") or None
    retail_raw = lot.get("retail_value") or lot.get("acv")
    hammer_raw = lot.get("current_bid_value") if lot.get("already_sold") else None

    return AuctionListing(
        source="salvagebid",
        lot_number=str(lot.get("id", "")),
        make=make,
        model=model,
        year=int(year),
        mileage=int(mileage),
        hammer_price=round(hammer_raw * 100) if hammer_raw else 0,
        retail_estimate=round(retail_raw * 100) if retail_raw else None,
        condition_score=None,
        damage_description=lot.get("damage") or lot.get("primary_damage"),
        city=(lot.get("location_city") or "").title() or None,
        state=lot.get("location_state"),
        vin=vin,
    )
