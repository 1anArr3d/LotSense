"""
data/parser.py — Normalize RawListing into structured features for the estimator.

Title format from FB: "2016 Toyota Camry · LE Sedan 4D"
  → year=2016, make="Toyota", model="Camry" (trim dropped)
"""

import re
from dataclasses import dataclass

from data.collector import RawListing

# Common trim/package tokens — stop collecting model words when we hit one
_TRIM_TOKENS = {
    "se", "le", "xle", "xlt", "xl", "gt", "lx", "ex", "exl",
    "sxt", "srt", "sl", "sr", "trd", "awd", "fwd", "4wd", "4x4",
    "v6", "v8", "l4", "sport", "limited", "base", "premium",
    "platinum", "titanium", "touring", "ltz", "lt", "ls", "ss",
    "z71", "denali", "laramie", "bighorn", "rebel", "tradesman",
    "trailhawk", "overland", "rubicon", "sahara", "sport-s",
}


@dataclass
class ParsedListing:
    fb_listing_id: str
    year: int
    make: str
    model: str
    mileage: int          # integer miles
    price_cents: int
    city: str | None
    state: str | None
    is_sold: bool


def parse_mileage(mileage_raw: str | None) -> int | None:
    if not mileage_raw:
        return None
    m = re.search(r"(\d[\d,]*)([Kk])?", mileage_raw.replace(",", ""))
    if not m:
        return None
    value = int(m.group(1))
    return value * 1000 if m.group(2) else value


def parse_title(title: str) -> tuple[int | None, str | None, str | None]:
    """Return (year, make, model) from a FB Marketplace title. Drops trim."""
    core = title.split("·")[0].strip()

    # Year can appear anywhere (e.g. "Used 2016 Toyota Camry")
    year_match = re.search(r"\b(\d{4})\b", core)
    if not year_match:
        return None, None, None

    year = int(year_match.group(1))
    if not (1980 <= year <= 2030):
        return None, None, None

    remainder = core[year_match.end():].strip().split()
    if not remainder:
        return year, None, None

    make = remainder[0].title()

    # Collect model words: up to 2 words, stop at trim tokens or non-alpha starts
    model_words = []
    for word in remainder[1:]:
        if word.lower() in _TRIM_TOKENS:
            break
        if not word[0].isalpha():
            break
        if word.lower() == make.lower():
            break
        model_words.append(word.title())
        if len(model_words) == 2:
            break

    if not model_words:
        return year, None, None

    return year, make, " ".join(model_words)


def parse(listing: RawListing) -> ParsedListing | None:
    """Return a ParsedListing, or None if critical fields are missing."""
    if listing.price_cents is None:
        return None

    mileage = parse_mileage(listing.mileage_raw)
    if mileage is None:
        return None

    year, make, model = parse_title(listing.title)
    if not all([year, make, model]):
        return None

    return ParsedListing(
        fb_listing_id=listing.fb_listing_id,
        year=year,
        make=make,
        model=model,
        mileage=mileage,
        price_cents=listing.price_cents,
        city=listing.city,
        state=listing.state,
        is_sold=listing.is_sold,
    )


def parse_all(listings: list[RawListing]) -> list[ParsedListing]:
    return [p for listing in listings if (p := parse(listing)) is not None]
