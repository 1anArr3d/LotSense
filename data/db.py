"""
data/db.py — SQLite storage for auction listings (Copart, IAAI).

Usage:
    from data.db import open_db, upsert_listings

    with open_db() as conn:
        upsert_listings(conn, listings)
"""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

_DB_PATH = Path(__file__).parent / "listings.db"

_STALE_DAYS = 90  # auction comps stay relevant longer than retail listings

_DDL = """
CREATE TABLE IF NOT EXISTS listings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source              TEXT    NOT NULL,
    lot_number          TEXT,
    make                TEXT,
    model               TEXT,
    year                INTEGER,
    mileage             INTEGER,
    hammer_price        INTEGER,
    retail_estimate     INTEGER,
    condition_score     REAL,
    damage_description  TEXT,
    city                TEXT,
    state               TEXT,
    scraped_at          INTEGER NOT NULL,
    UNIQUE(source, lot_number)
);

CREATE TABLE IF NOT EXISTS photos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id  INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    url         TEXT    NOT NULL,
    local_path  TEXT,
    is_lead     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS price_estimates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id  INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    low         INTEGER NOT NULL,
    mid         INTEGER NOT NULL,
    high        INTEGER NOT NULL,
    created_at  INTEGER NOT NULL
);
"""


@contextmanager
def open_db(path: Path = _DB_PATH):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_DDL)
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def upsert_listings(conn: sqlite3.Connection, listings: list) -> int:
    """
    Insert or update auction listings. Returns number of rows written.
    Accepts a list of AuctionListing objects.
    """
    now = int(time.time())
    rows = [
        (
            l.source,
            l.lot_number,
            l.make,
            l.model,
            l.year,
            l.mileage,
            l.hammer_price,
            l.retail_estimate,
            l.condition_score,
            l.damage_description,
            l.city,
            l.state,
            now,
        )
        for l in listings
    ]
    conn.executemany(
        """
        INSERT INTO listings (
            source, lot_number, make, model, year, mileage,
            hammer_price, retail_estimate, condition_score,
            damage_description, city, state, scraped_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, lot_number) DO UPDATE SET
            hammer_price       = excluded.hammer_price,
            retail_estimate    = excluded.retail_estimate,
            condition_score    = excluded.condition_score,
            damage_description = excluded.damage_description,
            mileage            = excluded.mileage,
            scraped_at         = excluded.scraped_at
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def get_listings(
    conn: sqlite3.Connection,
    source: str | None = None,
    make: str | None = None,
    model: str | None = None,
) -> list:
    """Return non-stale listings, optionally filtered by source/make/model."""
    from data.parser import AuctionListing

    cutoff = int(time.time()) - _STALE_DAYS * 86400
    clauses = ["scraped_at >= ?"]
    params: list = [cutoff]

    if source:
        clauses.append("source = ?")
        params.append(source)
    if make:
        clauses.append("LOWER(make) = LOWER(?)")
        params.append(make)
    if model:
        clauses.append("LOWER(model) = LOWER(?)")
        params.append(model)

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT * FROM listings WHERE {where}", params
    ).fetchall()

    return [
        AuctionListing(
            source=row["source"],
            lot_number=row["lot_number"],
            make=row["make"],
            model=row["model"],
            year=row["year"],
            mileage=row["mileage"],
            hammer_price=row["hammer_price"],
            retail_estimate=row["retail_estimate"],
            condition_score=row["condition_score"],
            damage_description=row["damage_description"],
            city=row["city"],
            state=row["state"],
        )
        for row in rows
    ]


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
