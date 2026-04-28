"""
data/copart.py — Copart auction listing fetcher.

Uses plain httpx — no browser or session cookies required.
Copart's search API is publicly accessible with standard headers.

Field mapping from search results JSON:
    lotNumberStr → lot_number
    mkn          → make
    lm           → model
    lcy          → year
    orr          → mileage (float odometer reading)
    la           → retail_estimate (ACV in dollars)
    hb           → hammer_price (highest bid — final price for sold lots)
    crg          → condition_score (AutoGrade 0–5)
    dd           → damage_description (primary)
    locState     → state
    yn           → city (yard name, e.g. "TX - SAN ANTONIO")
    tims         → lead thumbnail URL
"""

import time
import random
from typing import Iterator

import httpx

from data.parser import AuctionListing

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/147.0.0.0 Safari/537.36"
)

_SEARCH_URL = "https://www.copart.com/public/lots/search-results"
_PAGE_SIZE = 100

_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "x-requested-with": "XMLHttpRequest",
    "origin": "https://www.copart.com",
    "referer": "https://www.copart.com/vehicleSearchCriteria",
    "user-agent": _USER_AGENT,
}


class CopartCollector:

    def search(
        self,
        make: str | None = None,
        model: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        min_mileage: int | None = None,
        max_mileage: int | None = None,
        graded_only: bool = False,
        max_pages: int = 10,
        debug: bool = False,
    ) -> Iterator[AuctionListing]:
        """
        Yield AuctionListing objects from Copart's search API.

        Fetches salvage-title lots only (TITLEGROUP_S). Pass make + model to
        narrow results; omit for a broad sweep of all vehicles.
        Set graded_only=True to skip lots without an AutoGrade score.
        """
        filters: dict = {"TITL": ["title_group_code:TITLEGROUP_S"]}

        if make:
            filters["MAKE"] = [f'lot_make_desc:"{make.upper()}"']
        if model:
            filters["MODL"] = [f'lot_model_desc:"{model.upper()}"']

        if min_year is not None or max_year is not None:
            lo = min_year or 1900
            hi = max_year or 2100
            filters["YEAR"] = [f"lot_year:[{lo} TO {hi}]"]

        if min_mileage is not None or max_mileage is not None:
            lo = min_mileage or 0
            hi = max_mileage or 9_999_999
            filters["ODM"] = [f"odometer_reading_received:[{lo} TO {hi}]"]

        if graded_only:
            filters["CRG"] = ["crg:[0.1 TO 5]"]

        seen: set[str] = set()

        with httpx.Client(headers=_HEADERS, timeout=30) as client:
            for page_num in range(max_pages):
                payload = _build_payload(["*"], filters, page=page_num)
                resp = client.post(_SEARCH_URL, json=payload)
                resp.raise_for_status()

                data = resp.json()
                if data.get("returnCode") != 1:
                    break

                results = data["data"]["results"]
                content = results.get("content") or []
                total = results.get("totalElements", 0)

                if debug:
                    fetched = page_num * _PAGE_SIZE + len(content)
                    print(f"  [page {page_num + 1}] {len(content)} lots  ({fetched}/{total} total)")

                if not content:
                    break

                for lot in content:
                    listing = _lot_to_listing(lot)
                    if listing and listing.lot_number not in seen:
                        seen.add(listing.lot_number)
                        yield listing

                if (page_num + 1) * _PAGE_SIZE >= total:
                    break

                time.sleep(random.uniform(1.0, 2.5))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_payload(query: list[str], filters: dict, page: int) -> dict:
    return {
        "query": query,
        "filter": filters,
        "sort": ["auction_date_utc desc"],
        "page": page,
        "size": _PAGE_SIZE,
        "start": page * _PAGE_SIZE,
        "watchListOnly": False,
        "freeFormSearch": False,
        "hideImages": False,
        "defaultSort": False,
        "specificRowProvided": False,
        "displayName": "",
        "searchName": "",
        "backUrl": "",
        "includeTagByField": {},
        "rawParams": {},
    }


def _lot_to_listing(lot: dict) -> AuctionListing | None:
    mileage_raw = lot.get("orr")
    if not mileage_raw:
        return None
    mileage = int(mileage_raw)
    if mileage <= 0:
        return None

    make = (lot.get("mkn") or "").title()
    model = (lot.get("lm") or "").title()
    year = lot.get("lcy")
    if not all([make, model, year]):
        return None

    return AuctionListing(
        source="copart",
        lot_number=str(lot.get("lotNumberStr", "")),
        make=make,
        model=model,
        year=int(year),
        mileage=mileage,
        hammer_price=round((lot.get("hb") or 0) * 100),
        retail_estimate=round(lot["la"] * 100) if lot.get("la") else None,
        condition_score=lot.get("crg"),
        damage_description=lot.get("dd"),
        city=(lot.get("yn") or "").split(" - ")[-1].title() or None,
        state=lot.get("locState"),
    )
