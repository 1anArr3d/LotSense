"""
Inspection scraper — Playwright once to solve Turnstile, pure HTTP for all VINs.
Session is acquired once per batch and reused across all lookups.
"""
import re
import threading
from html.parser import HTMLParser

from curl_cffi import requests as cffi_requests
from playwright.sync_api import sync_playwright

from data.db import open_db

SEARCH_URL = "https://www.mytxcar.org/TXCar_Net/SearchVehicleTestHistory.aspx"
HISTORY_URL = "https://www.mytxcar.org/TXCar_Net/VehicleTestHistory.aspx"

class _SessionExpired(Exception):
    pass

_http_session: cffi_requests.Session | None = None
_session_lock = threading.Lock()
_lookup_lock = threading.Lock()


# ── Parsers ───────────────────────────────────────────────────────────────────

class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attrs = dict(attrs)
        onclick = attrs.get("onclick", "")
        aria = attrs.get("aria-label", "")
        if "DoSelect" in onclick and aria:
            m = re.search(r"DoSelect\('([^']*)',\s*'([^']*)',\s*'(\d+)',\s*'([^']*)'", onclick)
            if m:
                date = aria.split(" Click")[0].split(" ")[0]
                self.links.append({
                    "date": date,
                    "search_type": m.group(1),
                    "vin": m.group(2),
                    "row_id": m.group(3),
                    "tas_id": m.group(4),
                })


class _OdometerParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._next_is_value = False
        self.odometer = None

    def handle_data(self, data):
        if "Odometer" in data:
            self._next_is_value = True
        elif self._next_is_value and data.strip().replace(",", "").isdigit():
            self.odometer = int(data.strip().replace(",", ""))
            self._next_is_value = False


def _extract_hidden(html: str) -> dict:
    fields = {}
    for m in re.finditer(r'<input[^>]+name="(__[^"]+)"[^>]+value="([^"]*)"', html):
        fields[m.group(1)] = m.group(2)
    return fields


# ── Session management ────────────────────────────────────────────────────────

def _acquire_session() -> cffi_requests.Session:
    """Launch browser, solve Turnstile once, return HTTP session with cookie."""
    print("[inspection] Launching browser to solve Turnstile...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])
        page = browser.new_page()
        page.goto("https://www.mytxcar.org/TXCar_Net/VehicleTestDetail.aspx", timeout=60000)

        try:
            page.wait_for_function(
                "document.querySelector('[name=\"cf-turnstile-response\"]').value.length > 0",
                timeout=8000
            )
        except Exception:
            try:
                rect = page.evaluate("""
                    () => {
                        const el = document.querySelector('.cf-turnstile');
                        if (!el) return null;
                        const r = el.getBoundingClientRect();
                        return { x: r.left, y: r.top };
                    }
                """)
                if rect:
                    page.mouse.click(rect['x'] + 25, rect['y'] + 32)
            except Exception:
                pass
            page.wait_for_function(
                "document.querySelector('[name=\"cf-turnstile-response\"]').value.length > 0",
                timeout=30000
            )

        page.locator('input[type="submit"]').click()
        page.wait_for_selector("#txtVin", timeout=15000)

        cookies = {c["name"]: c["value"] for c in page.context.cookies()}
        session_id = cookies.get("ASP.NET_SessionId")
        browser.close()

    print(f"[inspection] Session acquired: {session_id}")
    return cffi_requests.Session(
        impersonate="chrome120",
        cookies={"ASP.NET_SessionId": session_id}
    )


def _get_session() -> cffi_requests.Session:
    global _http_session
    with _session_lock:
        if _http_session is None:
            _http_session = _acquire_session()
        return _http_session


def _reset_session():
    global _http_session
    with _session_lock:
        _http_session = None


# ── VIN lookup ────────────────────────────────────────────────────────────────

def _lookup_vin(vin: str, session: cffi_requests.Session) -> list[dict]:
    r = session.get(SEARCH_URL, timeout=15)
    if "txtVin" not in r.text:
        raise _SessionExpired()
    hidden = _extract_hidden(r.text)

    r = session.post(SEARCH_URL, data={**hidden, "txtVin": vin, "btnSearch": "Search"}, timeout=15)
    html = r.text

    parser = _LinkParser()
    parser.feed(html)
    if not parser.links:
        return []

    result_hidden = _extract_hidden(html)
    results = []
    for link in parser.links:
        detail_data = {
            **result_hidden,
            "hidAction":      "SelectSearch",
            "hidSearchType":  link["search_type"],
            "hidCode":        link["vin"],
            "hidSelectedRow": link["row_id"],
            "hidTasId":       link.get("tas_id", ""),
        }
        try:
            r = session.post(HISTORY_URL, data=detail_data, timeout=15)
            odo = _OdometerParser()
            odo.feed(r.text)
            if odo.odometer:
                results.append({"date": link["date"], "odometer": odo.odometer})
        except Exception:
            continue

    return results


# ── DB save ───────────────────────────────────────────────────────────────────

def _save_history(vin: str, results: list[dict]):
    if not results:
        return
    with open_db() as conn:
        for i, res in enumerate(results):
            conn.execute(
                """INSERT INTO odometer_history (row_id, vin, inspection_date, mileage)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT (row_id) DO UPDATE SET
                       vin             = excluded.vin,
                       inspection_date = excluded.inspection_date,
                       mileage         = excluded.mileage""",
                (f"{vin}_{i}", vin, res["date"], res["odometer"])
            )
        conn.commit()


# ── Public API ────────────────────────────────────────────────────────────────


def run_inspection_batch(vins: list[str], workers: int = 10):
    """Run inspections for a list of VINs in parallel, sharing one session."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    session = _get_session()

    def _process(vin: str):
        nonlocal session
        with _lookup_lock:
            try:
                results = _lookup_vin(vin, session)
                _save_history(vin, results)
                print(f"[inspection] {vin}: {len(results)} record(s)")
            except _SessionExpired:
                print("[inspection] Session expired — re-acquiring...")
                _reset_session()
                session = _get_session()
                try:
                    results = _lookup_vin(vin, session)
                    _save_history(vin, results)
                    print(f"[inspection] {vin}: {len(results)} record(s)")
                except Exception as e:
                    print(f"[inspection] Error for {vin} after re-auth: {e}")
            except Exception as e:
                print(f"[inspection] Error for {vin}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process, vin) for vin in vins]
        for future in as_completed(futures):
            future.result()
