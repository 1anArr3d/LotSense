"""
bot/handoff.py — Owner notification when a buyer confirms they're on their way.

Fires once per conversation when the buyer sends an arrival confirmation after
location has been given. No auto-reply goes out after this — owner takes over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable


# ── Arrival time parsing ────────────────────────────────────────────────────

_MINUTES_RE = re.compile(
    r"\bin\s+(about\s+)?(\d+|five|ten|fifteen|twenty|thirty)(\s+min(utes?)?)?\b",
    re.IGNORECASE,
)

_WORD_MIN = {"five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}


def _parse_eta(text: str) -> str | None:
    m = _MINUTES_RE.search(text)
    if not m:
        return None
    raw = m.group(2)
    minutes = _WORD_MIN.get(raw.lower()) or int(raw)
    t = datetime.now() + timedelta(minutes=minutes)
    hour = t.strftime("%I").lstrip("0") or "12"
    return f"~{hour}:{t.strftime('%M%p').lower()}"


# ── Summary ─────────────────────────────────────────────────────────────────

@dataclass
class HandoffSummary:
    listing_id: str
    car: str            # "2019 Toyota Camry"
    agreed_price: int
    eta: str            # "~3:45pm" or "en route / no ETA"
    buyer_message: str
    timestamp: datetime


def build_summary(
    listing_id: str,
    car_label: str,
    agreed_price: int,
    buyer_message: str,
) -> HandoffSummary:
    return HandoffSummary(
        listing_id=listing_id,
        car=car_label,
        agreed_price=agreed_price,
        eta=_parse_eta(buyer_message) or "en route / no ETA",
        buyer_message=buyer_message,
        timestamp=datetime.now(),
    )


# ── Notification ─────────────────────────────────────────────────────────────

def notify(
    summary: HandoffSummary,
    extra: Callable[[str], None] | None = None,
) -> None:
    """
    Print the handoff summary to stdout and attempt a desktop notification.
    Pass extra to hook in SMS, push, or any other delivery channel later.
    """
    title = f"BUYER INCOMING -- {summary.car}"
    body = (
        f"  Price    ${summary.agreed_price:,}\n"
        f"  Arrival  {summary.eta}\n"
        f'  Message  "{summary.buyer_message}"\n'
        f"  Listing  {summary.listing_id}\n"
        f"  At       {summary.timestamp.strftime('%I:%M %p').lstrip('0')}"
    )
    bar = "-" * 46

    print(f"\n{bar}\n  {title}\n{bar}\n{body}\n{bar}\n")

    # Desktop notification — best effort, no crash if plyer isn't installed
    try:
        from plyer import notification  # type: ignore
        notification.notify(
            title=title,
            message=f"${summary.agreed_price:,}  ·  Arriving {summary.eta}",
            timeout=12,
        )
    except Exception:
        pass

    if extra:
        extra(f"{title}\n{body}")
