"""
bot/adapter.py — Facebook message adapter layer.

FacebookAdapter is the abstract interface the dispatcher talks to.
MockAdapter replays a JSONL script file for integration testing without a browser.
PlaywrightAdapter polls facebook.com/messages with session cookies.

Selector notes (PlaywrightAdapter):
  Facebook's DOM changes periodically. When selectors break, open the inbox in
  Chrome DevTools, find the element, and update the constants below.
  Run with --debug to get screenshots saved as debug_*.png.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class IncomingMessage:
    thread_id: str
    sender_name: str
    text: str
    listing_id: str | None   # None when not determinable from the thread
    timestamp: datetime


# ── Abstract interface ────────────────────────────────────────────────────────

class FacebookAdapter(ABC):
    @abstractmethod
    def poll(self) -> list[IncomingMessage]:
        """Return any new messages since the last poll. May return []."""

    @abstractmethod
    def send(self, thread_id: str, text: str) -> None:
        """Send a reply to the given conversation thread."""

    @abstractmethod
    def close(self) -> None:
        """Release browser / resources."""


# ── Mock adapter ──────────────────────────────────────────────────────────────

class MockAdapter(FacebookAdapter):
    """
    Replays a JSONL script for repeatable integration tests.

    Script format — one JSON object per line:
        {"thread_id": "t1", "sender": "Maria", "text": "Is this available?", "listing_id": "TEST-001"}
        {"thread_id": "t1", "sender": "Maria", "text": "Would you take $7000?", "listing_id": "TEST-001"}

    Lines starting with # are ignored. Blank lines are skipped.
    """

    def __init__(self, script_path: Path | str) -> None:
        self._messages: list[IncomingMessage] = []
        self._pos = 0
        self._sent: list[tuple[str, str]] = []

        for line in Path(script_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            self._messages.append(IncomingMessage(
                thread_id=obj["thread_id"],
                sender_name=obj.get("sender", "Buyer"),
                text=obj["text"],
                listing_id=obj.get("listing_id"),
                timestamp=datetime.now(timezone.utc),
            ))

    def poll(self) -> list[IncomingMessage]:
        if self._pos >= len(self._messages):
            return []
        msg = self._messages[self._pos]
        self._pos += 1
        return [msg]

    def send(self, thread_id: str, text: str) -> None:
        print(f"  [Bot -> {thread_id}]  {text}")
        self._sent.append((thread_id, text))

    def close(self) -> None:
        pass

    @property
    def sent(self) -> list[tuple[str, str]]:
        return list(self._sent)

    def exhausted(self) -> bool:
        return self._pos >= len(self._messages)


# ── Playwright adapter ────────────────────────────────────────────────────────

# Facebook inbox — Marketplace messages appear here alongside regular messages.
_INBOX_URL = "https://www.facebook.com/messages/t/"

# Update these selectors when Facebook changes their DOM.
# Verify with DevTools: right-click element > Inspect > copy selector.
_SEL_THREAD_ROW = 'a[href*="/messages/t/"]'          # left-panel thread links
_SEL_MSG_BUBBLE = 'div[dir="auto"]'                  # message text nodes (stable: set by FB for RTL support)
_SEL_MSG_INPUT  = 'div[contenteditable="true"][role="textbox"]'  # reply box

_MARKETPLACE_LABEL = "Marketplace"   # text label present in Marketplace thread rows


class PlaywrightAdapter(FacebookAdapter):
    """
    Polls the Facebook Messenger inbox with stored browser session cookies.

    Requires:
      pip install playwright && playwright install chromium

    Requires env vars (copy from Chrome DevTools > Application > Cookies > facebook.com):
      FB_DATR, FB_SB, FB_C_USER, FB_FR, FB_XS

    Cookies expire roughly every 30 days — re-copy when auth fails.
    """

    def __init__(self, headless: bool = True, debug: bool = False) -> None:
        from dotenv import load_dotenv
        from playwright.sync_api import sync_playwright

        load_dotenv()
        self._debug = debug
        # Track (thread_id, message_text) pairs already routed to avoid double-delivery.
        self._seen: set[tuple[str, str]] = set()

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self._page = self._browser.new_page()
        self._page.set_extra_http_headers({"User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/147.0.0.0 Safari/537.36"
        )})

        self._inject_cookies()
        self._page.goto(_INBOX_URL, wait_until="domcontentloaded")
        self._check_auth()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _inject_cookies(self) -> None:
        required = ("FB_DATR", "FB_SB", "FB_C_USER", "FB_FR", "FB_XS")
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise EnvironmentError(
                f"Missing env vars: {', '.join(missing)}. "
                "Copy from Chrome DevTools > Application > Cookies > facebook.com"
            )
        self._page.context.add_cookies([
            {"name": "datr",   "value": os.environ["FB_DATR"],   "domain": ".facebook.com", "path": "/"},
            {"name": "sb",     "value": os.environ["FB_SB"],     "domain": ".facebook.com", "path": "/"},
            {"name": "c_user", "value": os.environ["FB_C_USER"], "domain": ".facebook.com", "path": "/"},
            {"name": "fr",     "value": os.environ["FB_FR"],     "domain": ".facebook.com", "path": "/"},
            {"name": "xs",     "value": os.environ["FB_XS"],     "domain": ".facebook.com", "path": "/"},
        ])

    def _check_auth(self) -> None:
        url = self._page.url
        if "login" in url or "checkpoint" in url:
            raise EnvironmentError(
                "Facebook session expired or invalid. "
                "Re-copy cookies from DevTools > Application > Cookies > facebook.com into .env"
            )

    def _screenshot(self, name: str) -> None:
        if self._debug:
            self._page.screenshot(path=f"debug_{name}.png")
            print(f"  [adapter] screenshot: debug_{name}.png")

    # ── Poll ─────────────────────────────────────────────────────────────────

    def poll(self) -> list[IncomingMessage]:
        try:
            self._page.goto(_INBOX_URL, wait_until="domcontentloaded")
            self._check_auth()
            self._screenshot("inbox")

            results: list[IncomingMessage] = []
            thread_links = self._page.query_selector_all(_SEL_THREAD_ROW)

            for link in thread_links:
                content = link.text_content() or ""
                if _MARKETPLACE_LABEL not in content:
                    continue

                href = link.get_attribute("href") or ""
                thread_id = href.rstrip("/").split("/")[-1]
                if not thread_id:
                    continue

                link.click()
                self._page.wait_for_load_state("domcontentloaded")
                self._screenshot(f"thread_{thread_id}")

                # Read the last 5 message bubbles — enough for context, not too many
                bubbles = self._page.query_selector_all(_SEL_MSG_BUBBLE)
                for bubble in bubbles[-5:]:
                    text = (bubble.text_content() or "").strip()
                    if not text:
                        continue
                    key = (thread_id, text)
                    if key in self._seen:
                        continue
                    self._seen.add(key)

                    results.append(IncomingMessage(
                        thread_id=thread_id,
                        sender_name=self._extract_sender(link),
                        text=text,
                        listing_id=self._extract_listing_id(),
                        timestamp=datetime.now(timezone.utc),
                    ))

                self._page.goto(_INBOX_URL, wait_until="domcontentloaded")

            return results

        except EnvironmentError:
            raise
        except Exception as e:
            if self._debug:
                print(f"  [adapter] poll error: {e}")
                self._screenshot("poll_error")
            return []

    # ── Send ─────────────────────────────────────────────────────────────────

    def send(self, thread_id: str, text: str) -> None:
        try:
            self._page.goto(f"{_INBOX_URL}{thread_id}", wait_until="domcontentloaded")
            box = self._page.wait_for_selector(_SEL_MSG_INPUT, timeout=6000)
            box.click()
            box.type(text, delay=25)   # slight delay mimics human typing
            box.press("Enter")
            time.sleep(0.8)            # wait for send to register
        except Exception as e:
            if self._debug:
                print(f"  [adapter] send error: {e}")
                self._screenshot("send_error")

    # ── Close ────────────────────────────────────────────────────────────────

    def close(self) -> None:
        try:
            self._browser.close()
            self._pw.stop()
        except Exception:
            pass

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _extract_sender(self, thread_link) -> str:
        try:
            for span in thread_link.query_selector_all("span"):
                t = (span.text_content() or "").strip()
                if t and t != _MARKETPLACE_LABEL:
                    return t
        except Exception:
            pass
        return "Buyer"

    def _extract_listing_id(self) -> str | None:
        """Look for /marketplace/item/XXXXXXX in any link on the open thread page."""
        try:
            for link in self._page.query_selector_all('a[href*="/marketplace/item/"]'):
                href = link.get_attribute("href") or ""
                parts = href.split("/marketplace/item/")
                if len(parts) > 1:
                    return parts[1].split("/")[0].split("?")[0]
        except Exception:
            pass
        return None
