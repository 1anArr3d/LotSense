"""
data/db.py — SQLite storage for RawListing objects.

Usage:
    from data.db import open_db, upsert_listings

    with open_db() as conn:
        upsert_listings(conn, listings)
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from data.collector import RawListing

_DB_PATH = Path(__file__).parent / "listings.db"

_DDL = """
CREATE TABLE IF NOT EXISTS listings (
    fb_listing_id   TEXT    PRIMARY KEY,
    title           TEXT    NOT NULL,
    price_cents     INTEGER,
    city            TEXT,
    state           TEXT,
    photo_url       TEXT,
    creation_time   INTEGER,
    mileage_raw     TEXT,
    is_sold         INTEGER NOT NULL DEFAULT 0,
    seller_name     TEXT,
    raw_json        TEXT,
    scraped_at      INTEGER NOT NULL
);
"""


@contextmanager
def open_db(path: Path = _DB_PATH):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_DDL)
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def upsert_listings(conn: sqlite3.Connection, listings: list[RawListing]) -> int:
    """
    Insert or update listings. Returns the number of rows written.
    Existing rows are updated if price, mileage, or sold status changed.
    """
    now = int(time.time())
    rows = [
        (
            l.fb_listing_id,
            l.title,
            l.price_cents,
            l.city,
            l.state,
            l.photo_url,
            l.creation_time,
            l.mileage_raw,
            int(l.is_sold),
            l.seller_name,
            json.dumps(l.raw),
            now,
        )
        for l in listings
    ]
    conn.executemany(
        """
        INSERT INTO listings (
            fb_listing_id, title, price_cents, city, state, photo_url,
            creation_time, mileage_raw, is_sold, seller_name, raw_json, scraped_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fb_listing_id) DO UPDATE SET
            price_cents   = excluded.price_cents,
            mileage_raw   = excluded.mileage_raw,
            is_sold       = excluded.is_sold,
            photo_url     = excluded.photo_url,
            scraped_at    = excluded.scraped_at
        """,
        rows,
    )
    conn.commit()
    return len(rows)


_STALE_DAYS = 30


def listing_count(conn: sqlite3.Connection) -> int:
    cutoff = int(time.time()) - _STALE_DAYS * 86400
    return conn.execute(
        "SELECT COUNT(*) FROM listings WHERE scraped_at >= ?", (cutoff,)
    ).fetchone()[0]


def purge_stale(conn: sqlite3.Connection) -> int:
    """Delete listings older than _STALE_DAYS. Returns number of rows deleted."""
    cutoff = int(time.time()) - _STALE_DAYS * 86400
    cursor = conn.execute("DELETE FROM listings WHERE scraped_at < ?", (cutoff,))
    conn.commit()
    return cursor.rowcount


def get_listings(conn: sqlite3.Connection) -> list[RawListing]:
    """Return all listings scraped within the last _STALE_DAYS days."""
    cutoff = int(time.time()) - _STALE_DAYS * 86400
    rows = conn.execute(
        "SELECT * FROM listings WHERE scraped_at >= ?", (cutoff,)
    ).fetchall()
    return [
        RawListing(
            fb_listing_id=row["fb_listing_id"],
            title=row["title"],
            price_cents=row["price_cents"],
            city=row["city"],
            state=row["state"],
            photo_url=row["photo_url"],
            creation_time=row["creation_time"],
            mileage_raw=row["mileage_raw"],
            is_sold=bool(row["is_sold"]),
            seller_name=row["seller_name"],
            raw=json.loads(row["raw_json"]) if row["raw_json"] else {},
        )
        for row in rows
    ]
