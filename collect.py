"""
collect.py — Collect SalvageBid listings + inspection history into SQLite.

Usage:
    python collect.py                     # collect all pages + inspect
    python collect.py --pages 10          # limit to N pages
    python collect.py --workers 10        # parallel inspection lookups (default: 10)
    python collect.py VIN1 VIN2 ...       # inspect specific VINs directly
"""
import argparse
import sys

from data.db import open_db, upsert_listings
from mileage.inspection_scrape import run_inspection_batch


def _collect_salvagebid(pages: int | None, workers: int) -> None:
    from data.salvagebid import SalvageBidCollector
    print(f"[salvagebid] Fetching {'all' if pages is None else pages} page(s)...")
    collector = SalvageBidCollector()
    listings = []
    vins = []
    seen: set[str] = set()

    for listing in collector.search(max_pages=pages, debug=True):
        if not listing.vin or listing.vin in seen:
            continue
        seen.add(listing.vin)
        listings.append(listing)
        vins.append(listing.vin)

    print(f"[salvagebid] {len(listings)} unique VINs collected")

    with open_db() as conn:
        n = upsert_listings(conn, listings)
        already_done = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT vin FROM odometer_history WHERE vin IS NOT NULL"
            ).fetchall()
        }
    print(f"[db] Saved {n} listings")

    new_vins = [v for v in vins if v not in already_done]
    print(f"[inspect] {len(new_vins)} new VINs to inspect ({len(vins) - len(new_vins)} already in DB), "
          f"using {workers} workers...")
    if not new_vins:
        print("[inspect] Nothing new to inspect.")
        return
    run_inspection_batch(new_vins, workers=workers)
    print(f"\n[collect] Done. Retrain: python -m pricing.mileage_model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vins", nargs="*", help="Specific VINs to inspect")
    parser.add_argument("--pages",   type=int, default=None, help="Pages to fetch from SalvageBid (default: all)")
    parser.add_argument("--workers", type=int, default=10,   help="Parallel inspection workers (default: 10)")
    args = parser.parse_args()

    if args.vins:
        run_inspection_batch(args.vins, workers=args.workers)
    elif not sys.stdin.isatty():
        print("Usage: python collect.py [--pages N] [--workers N]")
        print("       python collect.py VIN1 VIN2 ...")
        sys.exit(1)
    else:
        _collect_salvagebid(args.pages, args.workers)


if __name__ == "__main__":
    main()
