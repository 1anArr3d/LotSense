"""
bot/dispatcher.py — Main polling loop.

Maintains one ConversationEngine per active Facebook thread. On each poll cycle:
  1. Fetch new messages from the adapter
  2. Classify intent (3-message sliding window)
  3. Route through the matching engine
  4. Send the reply (or fire handoff notification on arrival confirmation)

Usage:
    from bot.adapter import PlaywrightAdapter
    from bot.dispatcher import BotDispatcher, load_config

    config = load_config("bot/listings.json")
    adapter = PlaywrightAdapter()
    dispatcher = BotDispatcher(adapter, config["cars"], config["schedule"])
    dispatcher.run()   # blocks; Ctrl-C to stop
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bot.adapter import FacebookAdapter, IncomingMessage
from bot.classifier import IntentClassifier
from bot.conversation import CarContext, ConversationEngine, ListingGate, Schedule
from bot.handoff import build_summary, notify


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(path: Path | str) -> dict[str, Any]:
    """
    Load listings.json. Returns:
      {
        "cars": {listing_id: CarContext},
        "schedule": Schedule,
      }
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    schedule_raw = raw["schedule"]
    schedule = Schedule(
        today_open=schedule_raw["today_open"],
        today_close=schedule_raw["today_close"],
        next_day=schedule_raw["next_day"],
        next_open=schedule_raw["next_open"],
        next_close=schedule_raw["next_close"],
        location=schedule_raw["location"],
    )

    cars: dict[str, CarContext] = {}
    for entry in raw["listings"]:
        lid = entry["listing_id"]
        cars[lid] = CarContext(
            listing_id=lid,
            make=entry["make"],
            model=entry["model"],
            year=entry["year"],
            mileage=entry["mileage"],
            listed_price=float(entry["listed_price"]),
            floor_price=float(entry["floor_price"]),
            market_value=float(entry.get("market_value", entry["listed_price"] * 0.97)),
            listing_created_at=datetime.fromisoformat(entry["listed_at"]).replace(tzinfo=timezone.utc),
            details=entry.get("details", {}),
        )

    return {"cars": cars, "schedule": schedule}


# ── Dispatcher ────────────────────────────────────────────────────────────────

class BotDispatcher:
    """
    poll_interval_s : seconds between inbox polls (default 30)
    cars            : dict mapping listing_id -> CarContext
    schedule        : shared Schedule for all listings
    extra_notify    : optional callback(str) for SMS/push on handoff
    """

    def __init__(
        self,
        adapter: FacebookAdapter,
        cars: dict[str, CarContext],
        schedule: Schedule,
        poll_interval_s: int = 30,
        extra_notify: Callable[[str], None] | None = None,
    ) -> None:
        self._adapter = adapter
        self._cars = cars
        self._schedule = schedule
        self._poll_interval = poll_interval_s
        self._extra_notify = extra_notify

        self._classifier = IntentClassifier()
        self._engines: dict[str, ConversationEngine] = {}   # thread_id -> engine
        self._gates: dict[str, ListingGate] = {}            # listing_id -> gate
        self._histories: dict[str, list[str]] = {}          # thread_id -> buyer messages

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        print(f"[dispatcher] started — polling every {self._poll_interval}s  (Ctrl-C to stop)")
        try:
            while True:
                for msg in self._adapter.poll():
                    self._handle(msg)
                time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            print("\n[dispatcher] stopped")
        finally:
            self._adapter.close()

    # ── Single message handler ───────────────────────────────────────────────

    def _handle(self, msg: IncomingMessage) -> None:
        tid = msg.thread_id

        engine = self._engines.get(tid)

        if engine is None:
            engine = self._init_engine(msg)
            if engine is None:
                return   # listing not found or no listing_id

        if engine.handed_off:
            return

        # Classify using sliding window of last 3 buyer messages
        history = self._histories[tid]
        history.append(msg.text)
        intent = self._classifier.classify(history)

        flag = "auto" if intent.auto_reply else "escalate"
        print(
            f"[{tid[:10]}] {intent.label} ({intent.confidence:.2f}) {flag}"
            f"  | {msg.sender_name}: {msg.text[:60]}"
        )

        if not intent.auto_reply:
            print(f"[{tid[:10]}] escalating — no reply sent")
            return

        reply = engine.respond(msg.text, intent)

        if reply is None:
            self._on_terminal(engine, msg, tid)
        else:
            self._adapter.send(tid, reply)

    def _init_engine(self, msg: IncomingMessage) -> ConversationEngine | None:
        tid = msg.thread_id

        if not msg.listing_id:
            print(f"[{tid[:10]}] no listing_id in message — skipping")
            return None

        car = self._cars.get(msg.listing_id)
        if car is None:
            print(f"[{tid[:10]}] listing {msg.listing_id} not in config — skipping")
            return None

        gate = self._gates.setdefault(msg.listing_id, ListingGate())
        engine = ConversationEngine(car, self._schedule, gate=gate)
        self._engines[tid] = engine
        self._histories[tid] = []
        return engine

    def _on_terminal(
        self, engine: ConversationEngine, msg: IncomingMessage, tid: str
    ) -> None:
        state = engine.state.value
        if state == "handed_off":
            car = engine.car
            summary = build_summary(
                car.listing_id,
                f"{car.year} {car.make} {car.model}",
                engine.agreed_price or int(car.listed_price),
                msg.text,
            )
            notify(summary, extra=self._extra_notify)

            # Flag listing so the next buyer gets the "someone's on their way" message
            gate = self._gates.get(car.listing_id)
            if gate:
                gate.buyer_incoming = True

            print(f"[{tid[:10]}] handed off")
            del self._engines[tid]

        elif state == "closed":
            print(f"[{tid[:10]}] conversation closed")
            del self._engines[tid]
