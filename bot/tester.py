"""
bot/tester.py — Interactive tester for the conversation bot.

Simulates a buyer conversation in the terminal. No Facebook connection needed.
Uses the real Claude API if ANTHROPIC_API_KEY is set, otherwise falls back to
a mock that returns canned responses so you can test negotiation logic for free.

Usage:
    python -m bot.tester              # auto-detect API key
    python -m bot.tester --mock       # force mock mode
    python -m bot.tester --listed 7500 --floor 6500 --days 15

Commands mid-conversation:
    /sold       toggle listing as sold
    /incoming   toggle another buyer as incoming
    /state      show current conversation state
    /reset      restart the conversation
    /quit       exit
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass


# ── Mock LLM client ─────────────────────────────────────────────────────────

class _MockMessages:
    """Minimal stand-in for anthropic.Anthropic().messages so no API key is needed."""

    class _Content:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Response:
        def __init__(self, text: str) -> None:
            self.content = [_MockMessages._Content(text)]

    def create(self, *, model, max_tokens, system, messages, **_) -> "_MockMessages._Response":
        instruction = ""
        if isinstance(system, list):
            instruction = system[0].get("text", "")
        elif isinstance(system, str):
            instruction = system

        extra = ""
        if "For this response only:" in instruction:
            extra = instruction.split("For this response only:")[-1].strip()

        text = self._respond(extra or instruction, messages)
        return self._Response(text)

    @staticmethod
    def _respond(instruction: str, messages: list[dict]) -> str:
        i = instruction.lower()
        # Pull any dollar amount mentioned in the instruction
        m = re.search(r"\$([0-9,]+)", instruction)
        price = m.group(0) if m else ""

        if "accept" in i and "asking price" in i:
            return f"Works for me, {price} it is."
        if "matched your price" in i or "came up to your price" in i:
            return f"Deal at {price}."
        if "counter at" in i:
            return f"Best I can do is {price}."
        if "hold firm" in i:
            return f"Price is firm at {price}."
        if "lowest is" in i:
            return f"Lowest I can do is {price}."
        if "deal" in i and "confirm" in i:
            return f"Locked in at {price}. I'll send you the location."
        if "location is next" in i:
            return f"Done at {price}, I'll get you the address."
        last_msg = messages[-1]["content"] if messages else ""
        if "?" in last_msg:
            return "Yeah it runs good, no major issues."
        return "Still available, what do you need to know?"


class _MockClient:
    def __init__(self) -> None:
        self.messages = _MockMessages()


# ── Test fixtures ────────────────────────────────────────────────────────────

def _make_car(listed: float, floor: float, days_old: int) -> "CarContext":
    from bot.conversation import CarContext
    created = datetime.now(timezone.utc) - timedelta(days=days_old)
    return CarContext(
        listing_id="TEST-001",
        make="Honda",
        model="Accord",
        year=2018,
        mileage=87_000,
        listed_price=listed,
        floor_price=floor,
        market_value=listed * 0.97,
        listing_created_at=created,
        details={
            "drivetrain": "FWD",
            "transmission": "automatic",
            "color": "silver",
            "accident": "minor rear-end, 2021 — repaired at body shop",
        },
    )


def _make_schedule() -> "Schedule":
    from bot.conversation import Schedule
    return Schedule(
        today_open="10am",
        today_close="6pm",
        next_day="Saturday",
        next_open="10am",
        next_close="5pm",
        location="1234 Culebra Rd, San Antonio TX",
    )


# ── REPL ────────────────────────────────────────────────────────────────────

def _header(car, gate, use_mock: bool) -> None:
    from bot.conversation import _decay_floor
    floor = max(_decay_floor(car.listed_price, car.listing_created_at), int(car.floor_price))
    days = (datetime.now(timezone.utc) - car.listing_created_at.replace(tzinfo=timezone.utc)).days
    mode = "[MOCK]" if use_mock else "[LIVE]"
    print(f"\n{'=' * 52}")
    print(f"  LotSense Bot Tester  {mode}")
    print(f"{'=' * 52}")
    print(f"  Car      {car.year} {car.make} {car.model}, {car.mileage:,} mi")
    print(f"  Listed   ${car.listed_price:,.0f}  |  Floor  ${floor:,}  (day {days})")
    print(f"  Sold     {gate.is_sold}  |  Incoming  {gate.buyer_incoming}")
    print(f"{'-' * 52}")
    print("  Commands:  /sold  /incoming  /state  /reset  /quit")
    print(f"{'-' * 52}\n")


def run(listed: float, floor: float, days_old: int, use_mock: bool) -> None:
    from bot.classifier import IntentClassifier
    from bot.conversation import ConversationEngine, ListingGate
    from bot.handoff import build_summary, notify

    car = _make_car(listed, floor, days_old)
    schedule = _make_schedule()
    gate = ListingGate()

    classifier = IntentClassifier()  # keyword fallback — no weights needed for testing

    client = _MockClient() if use_mock else None  # type: ignore[arg-type]

    def _new_engine():
        return ConversationEngine(car, schedule, gate=gate, client=client)

    engine = _new_engine()
    _header(car, gate, use_mock)

    while True:
        try:
            raw = input("  [Buyer] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not raw:
            continue

        # Commands
        if raw.startswith("/"):
            cmd = raw.lower()
            if cmd in ("/quit", "/q", "/exit"):
                print("bye")
                break
            elif cmd == "/sold":
                gate.is_sold = not gate.is_sold
                print(f"  >is_sold = {gate.is_sold}")
            elif cmd == "/incoming":
                gate.buyer_incoming = not gate.buyer_incoming
                print(f"  >buyer_incoming = {gate.buyer_incoming}")
            elif cmd == "/state":
                print(f"  >state = {engine.state.value}  |  agreed = {engine.agreed_price}")
            elif cmd == "/reset":
                engine = _new_engine()
                print("  >conversation reset\n")
                _header(car, gate, use_mock)
            else:
                print(f"  unknown command: {raw}")
            continue

        # Classify
        all_buyer = [m["content"] for m in engine._history if m["role"] == "user"] + [raw]
        result = classifier.classify(all_buyer)
        flag = "auto" if result.auto_reply else "escalate"
        print(f"  [Intent] {result.label} ({result.confidence:.2f}) | {flag}")

        # Respond
        reply = engine.respond(raw, result)

        if reply is None:
            if engine.state.value == "handed_off":
                summary = build_summary(
                    car.listing_id,
                    f"{car.year} {car.make} {car.model}",
                    engine.agreed_price or int(car.listed_price),
                    raw,
                )
                notify(summary)
                print("  [Bot]    --- handed off, no further replies ---\n")
                engine = _new_engine()  # reset for next test run
            elif engine.state.value == "closed":
                print("  [Bot]    --- conversation closed ---\n")
                engine = _new_engine()
        else:
            print(f"  [Bot]    {reply}\n")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LotSense bot tester")
    parser.add_argument("--mock",   action="store_true", help="skip Claude API, use canned responses")
    parser.add_argument("--listed", type=float, default=9_500, help="listed price (default 9500)")
    parser.add_argument("--floor",  type=float, default=8_500, help="hard floor price (default 8500)")
    parser.add_argument("--days",   type=int,   default=0,     help="listing age in days (default 0)")
    args = parser.parse_args()

    use_mock = args.mock or not os.getenv("ANTHROPIC_API_KEY")
    if use_mock and not args.mock:
        print("  [note] ANTHROPIC_API_KEY not set — running in mock mode")

    run(args.listed, args.floor, args.days, use_mock)


if __name__ == "__main__":
    main()
