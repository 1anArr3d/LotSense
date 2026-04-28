"""
bot/conversation.py — Response generation and negotiation logic.

All negotiation decisions are made deterministically; Claude Haiku is only called
to render the final line in natural language. State machine:

    INITIAL → NEGOTIATING → PRICE_AGREED → LOCATION_GIVEN → HANDED_OFF
    Any state → CLOSED  (as-is dealbreaker, buyer ghosts after way-out counter)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import anthropic

from bot.classifier import IntentResult


# ── Data contracts ──────────────────────────────────────────────────────────

@dataclass
class CarContext:
    listing_id: str
    make: str
    model: str
    year: int
    mileage: int
    listed_price: float
    floor_price: float        # hard minimum from pricing module
    market_value: float       # mid estimate from pricing module
    listing_created_at: datetime
    details: dict             # drivetrain, features, accident, anything car-specific


@dataclass
class Schedule:
    today_open: str    # "2pm"
    today_close: str   # "6pm"
    next_day: str      # "Saturday"
    next_open: str     # "10am"
    next_close: str    # "4pm"
    location: str      # revealed only after price is agreed


@dataclass
class ListingGate:
    """Listing-level flags the caller keeps current (e.g. from SQLite)."""
    is_sold: bool = False         # car sold — stop all conversations immediately
    buyer_incoming: bool = False  # another buyer already confirmed they're on their way


class State(Enum):
    INITIAL = "initial"
    NEGOTIATING = "negotiating"
    PRICE_AGREED = "price_agreed"
    LOCATION_GIVEN = "location_given"
    HANDED_OFF = "handed_off"
    CLOSED = "closed"


# ── Negotiation math ────────────────────────────────────────────────────────

# (max_days, discount_pct) — first matching bucket wins
_DECAY: list[tuple[int, float]] = [
    (14,   0.050),   # days  0–13: listed - 5%
    (28,   0.075),   # days 14–27: listed - 7.5%
    (9999, 0.100),   # day  28+:  listed - 10% (cap)
]


def _round50(n: float) -> int:
    return int(round(n / 50) * 50)


def _decay_floor(listed: float, created_at: datetime) -> int:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    days = (datetime.now(timezone.utc) - created_at).days
    for cap, pct in _DECAY:
        if days < cap:
            return _round50(listed * (1 - pct))
    return _round50(listed * 0.90)


_DOLLAR_RE = re.compile(r"\$\s*([0-9][0-9,]{2,})")


def _extract_offer(text: str) -> float | None:
    m = _DOLLAR_RE.search(text)
    return float(m.group(1).replace(",", "")) if m else None


# ── Pattern detection ───────────────────────────────────────────────────────

_SPANISH_RE = re.compile(
    r"\b(el|la|los|las|de|en|que|está|tiene|puedo|acepta|disponible|"
    r"efectivo|precio|carro|cuánto|cómo|dónde|cuándo|todavía|millas)\b",
    re.IGNORECASE,
)
_LOCATION_RE = re.compile(
    r"\b(where|where'?s|location|address|dónde|dirección|how do i get there)\b",
    re.IGNORECASE,
)
_ARRIVAL_RE = re.compile(
    r"\b(heading over|on my way|be there|leaving now|omw|pulling up|"
    r"almost there|in \d+ min|voy para allá|ya voy|salgo ahorita|en camino|llegando)\b",
    re.IGNORECASE,
)
_AGREEMENT_RE = re.compile(
    r"\b(deal|okay|ok|sounds good|works for me|i.ll take it|let.s do it|"
    r"agreed|done|sold|trato|de acuerdo|está bien|lo tomo)\b",
    re.IGNORECASE,
)
_CARFAX_RE = re.compile(
    r"\b(carfax|autocheck|vehicle\s*history|car\s*report|history\s*report|"
    r"historial del (auto|carro|veh[ií]culo))\b",
    re.IGNORECASE,
)
_ACCIDENT_RE = re.compile(
    r"\b(accident|crash|collision|wreck|been\s*hit|"
    r"accidente|choque|chocado|golpe|lo\s*chocaron)\b",
    re.IGNORECASE,
)


def _lang(text: str) -> str:
    return "es" if _SPANISH_RE.search(text) else "en"


# ── Engine ──────────────────────────────────────────────────────────────────

class ConversationEngine:
    """
    Manages one buyer conversation end-to-end.
    Call respond() for each incoming message. Returns None when done.
    """

    def __init__(
        self,
        car: CarContext,
        schedule: Schedule,
        gate: ListingGate | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.car = car
        self.schedule = schedule
        self.gate = gate or ListingGate()
        self._client = client or anthropic.Anthropic()

        self._state = State.INITIAL
        self._counter_made = False
        self._report_asked = False
        self._agreed_price: int | None = None
        self._location_redirects = 0
        self._history: list[dict] = []

    @property
    def state(self) -> State:
        return self._state

    @property
    def handed_off(self) -> bool:
        return self._state in (State.HANDED_OFF, State.CLOSED)

    @property
    def agreed_price(self) -> int | None:
        return self._agreed_price

    # ── Main entry ──────────────────────────────────────────────────────────

    def respond(self, buyer_message: str, intent: IntentResult) -> str | None:
        """
        Process one buyer message. Returns the bot reply, or None when the
        conversation is closed or handed off — caller should stop at that point.
        """
        if self._state in (State.CLOSED, State.HANDED_OFF):
            return None

        # Listing-level kill switches — checked on every message
        if self.gate.is_sold:
            self._state = State.CLOSED
            return None

        lang = _lang(buyer_message)
        self._history.append({"role": "user", "content": buyer_message})

        reply = self._route(buyer_message, lang)

        if reply is not None:
            self._history.append({"role": "assistant", "content": reply})
        return reply

    # ── Routing ─────────────────────────────────────────────────────────────

    def _route(self, msg: str, lang: str) -> str | None:
        # Arrival confirmation → silent handoff, caller handles notification
        if self._state == State.LOCATION_GIVEN and _ARRIVAL_RE.search(msg):
            self._state = State.HANDED_OFF
            return None

        # Location request
        if _LOCATION_RE.search(msg):
            return self._handle_location(lang)

        # Report / Carfax (check before accident — "accident report" hits this first)
        if _CARFAX_RE.search(msg):
            return self._handle_report(lang)

        # Direct accident question
        if _ACCIDENT_RE.search(msg):
            return self._handle_accident(lang)

        # Dollar offer
        offer = _extract_offer(msg)
        if offer is not None:
            self._state = State.NEGOTIATING
            return self._handle_offer(offer, lang)

        # Price agreement (only meaningful once we've made a counter)
        if self._state == State.NEGOTIATING and self._counter_made and _AGREEMENT_RE.search(msg):
            self._state = State.PRICE_AGREED
            return self._llm(lang, f"Deal at ${self._agreed_price:,}. Confirm briefly, tell them location is next.")

        return self._llm(lang)

    # ── Negotiation ─────────────────────────────────────────────────────────

    def _handle_offer(self, offer: float, lang: str) -> str | None:
        listed = self.car.listed_price
        floor = max(_decay_floor(listed, self.car.listing_created_at), int(self.car.floor_price))

        # Accepted or over asking
        if offer >= listed:
            self._agreed_price = _round50(listed)
            self._state = State.PRICE_AGREED
            return self._llm(lang, f"They met asking price. Accept. Deal at ${self._agreed_price:,}.")

        # They matched or beat our standing counter
        if self._counter_made and self._agreed_price and offer >= self._agreed_price:
            self._state = State.PRICE_AGREED
            return self._llm(lang, f"They came up to your price. Deal at ${self._agreed_price:,}.")

        # Hold firm — counter already made
        if self._counter_made:
            return self._llm(lang, f"Hold firm at ${self._agreed_price:,}. One sentence, no apology.")

        # Within negotiable range — first counter
        if offer >= floor:
            counter = max(_round50((listed + offer) / 2), floor)
            self._agreed_price = counter
            self._counter_made = True
            return self._llm(lang, f"Counter at ${counter:,}. One sentence.")

        # Way out — bump counter at floor + 2%
        bump = _round50(floor * 1.02)
        self._agreed_price = bump
        self._counter_made = True
        return self._llm(lang, f"Their number doesn't work. Lowest is ${bump:,}. One sentence, no explanation.")

    # ── Canned responses ────────────────────────────────────────────────────

    def _handle_location(self, lang: str) -> str | None:
        if self._state == State.PRICE_AGREED:
            if self.gate.buyer_incoming:
                self._state = State.CLOSED
                if lang == "es":
                    return "Alguien ya viene en camino. Si no llegan, avíseme y le doy chance."
                return "Someone's already on their way. If they pass, reach out and I'll let you know."
            self._state = State.LOCATION_GIVEN
            return self._location_message(lang)

        self._location_redirects += 1
        if self._location_redirects > 2:
            self._state = State.CLOSED
            return None

        listed = int(self.car.listed_price)
        if lang == "es":
            return f"Primero el precio. ¿Está bien con ${listed:,}?"
        return f"Let's settle on a price first. You good with ${listed:,}?"

    def _handle_report(self, lang: str) -> str | None:
        if self._report_asked:
            self._state = State.CLOSED
            if lang == "es":
                return "Entendido, no es para todos. Suerte en su búsqueda."
            return "Understood, not for everyone. Good luck out there."
        self._report_asked = True
        if lang == "es":
            return "Se vende as-is, no hay reportes. El precio ya refleja eso."
        return "Sold as-is, no reports. Price reflects that."

    def _handle_accident(self, lang: str) -> str | None:
        accident = self.car.details.get("accident", "")
        if accident:
            if lang == "es":
                return f"Sí, accidente en el historial: {accident}. Se vende as-is."
            return f"Yes, accident on record: {accident}. Sold as-is."
        if lang == "es":
            return "Se vende as-is, sin reportes disponibles."
        return "Sold as-is. No reports available."

    def _location_message(self, lang: str) -> str:
        s = self.schedule
        try:
            close_hour = datetime.strptime(s.today_close.upper(), "%I%p").hour
        except ValueError:
            close_hour = 18
        now = datetime.now()
        if now.hour < close_hour:
            if lang == "es":
                return f"{s.location}. Hoy hasta las {s.today_close}."
            return f"{s.location}. Here until {s.today_close} today."
        if lang == "es":
            return (
                f"Ya cerramos por hoy. Siguiente: {s.next_day} "
                f"de {s.next_open} a {s.next_close}. {s.location}."
            )
        return f"Done for today. Next: {s.next_day}, {s.next_open}-{s.next_close}. {s.location}."

    # ── LLM ─────────────────────────────────────────────────────────────────

    def _system_prompt(self, lang: str) -> str:
        c = self.car
        floor = _decay_floor(c.listed_price, c.listing_created_at)
        accident = c.details.get("accident", "none on record")
        lines = [
            f"Car: {c.year} {c.make} {c.model}, {c.mileage:,} miles",
            f"Listed: ${c.listed_price:,.0f} | Floor: ${floor:,}",
            f"Details: {c.details}",
            f"Accident: \"{accident}\" — only bring up if directly asked about accidents.",
            "",
            "Rules:",
            "- Sold as-is. No vehicle history reports. Cash or CashApp only. No financing.",
            "- If they ask for a Carfax: as-is, no reports, price reflects it. If they push: close.",
            "- These cars have been in accidents. Don't be defensive, don't volunteer it.",
            "- Short responses only. No apologies. No filler. Numbers when relevant.",
            "- You're not here to make friends. Just move the car.",
        ]
        prefix = "Respond in Spanish.\n\n" if lang == "es" else ""
        return prefix + "\n".join(lines)

    def _llm(self, lang: str, extra_instruction: str = "") -> str:
        system = self._system_prompt(lang)
        if extra_instruction:
            system += f"\n\nFor this response only: {extra_instruction}"

        response = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=self._history,
        )
        return response.content[0].text.strip()
