"""
data/parser.py — Shared field normalization for auction sources (Copart, IAAI).
"""

import re
from dataclasses import dataclass


@dataclass
class AuctionListing:
    source: str                      # 'copart' or 'iaai'
    lot_number: str
    make: str
    model: str
    year: int
    mileage: int
    hammer_price: int                # cents
    retail_estimate: int | None      # cents; None for IAAI when not available
    condition_score: float | None    # 0–5 AutoGrade; None for IAAI
    damage_description: str | None
    city: str | None
    state: str | None


def parse_mileage(mileage_raw: str | None) -> int | None:
    """Parse strings like '78K miles', '78,000', '78000' → integer miles."""
    if not mileage_raw:
        return None
    m = re.search(r"(\d[\d,]*)([Kk])?", mileage_raw.replace(",", ""))
    if not m:
        return None
    value = int(m.group(1))
    return value * 1000 if m.group(2) else value


def parse_price(price_raw: str | None) -> int | None:
    """Parse price strings to cents. '$5,900.00' → 590000."""
    if not price_raw:
        return None
    cleaned = re.sub(r"[^\d.]", "", str(price_raw))
    try:
        return round(float(cleaned) * 100)
    except (ValueError, TypeError):
        return None
