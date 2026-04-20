"""
main.py — LotSense CLI

Commands:
    collect   Scrape FB Marketplace and save listings to DB
    price     Estimate a fair price for a car you want to list
    stats     Show DB summary
"""

import argparse
import sys

from data.collector import FacebookAuthError, FacebookMarketplaceCollector
from data.db import get_listings, listing_count, open_db, purge_stale, upsert_listings
from data.parser import parse_all
from pricing.estimator import Estimator


# ------------------------------------------------------------------
# collect
# ------------------------------------------------------------------

def cmd_collect(args) -> None:
    try:
        collector = FacebookMarketplaceCollector()
    except EnvironmentError as e:
        print(f"Config error: {e}")
        sys.exit(1)

    query = f"{args.make} {args.model}"
    locations = args.locations or None

    if args.refresh:
        with open_db() as conn:
            purged = purge_stale(conn)
        if purged:
            print(f"Purged {purged} stale listing(s) (>30 days old).")

    print(f"Collecting '{query}' across {locations or ['default']} …")

    try:
        listings = list(collector.collect(
            max_pages=args.pages,
            query=query,
            debug=args.debug,
            min_year=args.min_year,
            max_year=args.max_year,
            min_price=args.min_price,
            max_price=args.max_price,
            locations=locations,
        ))
    except FacebookAuthError as e:
        print(f"\nAuth error: {e}")
        sys.exit(1)

    if not listings:
        print("No listings returned.")
        sys.exit(1)

    with open_db() as conn:
        written = upsert_listings(conn, listings)
        total = listing_count(conn)

    print(f"{written} listing(s) saved  ({total} total in DB).")


# ------------------------------------------------------------------
# price
# ------------------------------------------------------------------

def cmd_price(args) -> None:
    print(f"Loading comps for {args.year} {args.make} {args.model} …")

    try:
        est = Estimator.from_db(
            make=args.make,
            model=args.model,
            target_year=args.year,
        )
    except ValueError as e:
        print(f"\nNot enough data: {e}")
        print("Run `collect` first to gather more listings.")
        sys.exit(1)

    result = est.predict(year=args.year, mileage=args.mileage)

    confidence = result["confidence"]
    conf_label = {"high": "model estimate", "low": "rough estimate — collect more data"}[confidence]

    print(f"\n  {args.year} {args.make} {args.model} — {args.mileage:,} miles  [{conf_label}]")
    print(f"  ┌─────────────────────────────┐")
    print(f"  │  Low   ${result['low']:>8,.0f}            │")
    print(f"  │  Mid   ${result['mid']:>8,.0f}  ← list here │")
    print(f"  │  High  ${result['high']:>8,.0f}            │")
    print(f"  └─────────────────────────────┘")
    print(f"  Based on {result['n_comps']} comps (year-weighted).\n")


# ------------------------------------------------------------------
# stats
# ------------------------------------------------------------------

def cmd_stats(args) -> None:
    with open_db() as conn:
        total = listing_count(conn)
        raw = get_listings(conn)

    parsed = parse_all(raw)

    makes = {}
    for l in parsed:
        key = f"{l.make} {l.model}"
        makes[key] = makes.get(key, 0) + 1

    print(f"\n  DB: {total} total listings  ({len(parsed)} parseable)\n")
    print(f"  {'Make / Model':<28} {'Count':>5}")
    print(f"  {'-' * 35}")
    for name, count in sorted(makes.items(), key=lambda x: -x[1]):
        print(f"  {name:<28} {count:>5}")
    print()


# ------------------------------------------------------------------
# CLI wiring
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="lotsense", description="FB Marketplace price estimator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # collect
    p_collect = sub.add_parser("collect", help="Scrape listings and save to DB")
    p_collect.add_argument("make", help="Vehicle make (e.g. Toyota)")
    p_collect.add_argument("model", help="Vehicle model (e.g. Camry)")
    p_collect.add_argument("--pages", type=int, default=5)
    p_collect.add_argument("--min-year", type=int, default=None)
    p_collect.add_argument("--max-year", type=int, default=None)
    p_collect.add_argument("--min-price", type=int, default=None)
    p_collect.add_argument("--max-price", type=int, default=None)
    p_collect.add_argument("--locations", nargs="+", default=None, metavar="SLUG")
    p_collect.add_argument("--refresh", action="store_true", help="Purge stale listings (>30 days) before collecting")
    p_collect.add_argument("--debug", action="store_true")

    # price
    p_price = sub.add_parser("price", help="Estimate a fair listing price")
    p_price.add_argument("make", help="Vehicle make (e.g. Toyota)")
    p_price.add_argument("model", help="Vehicle model (e.g. Camry)")
    p_price.add_argument("year", type=int, help="Model year")
    p_price.add_argument("mileage", type=int, help="Odometer reading in miles")

    # stats
    sub.add_parser("stats", help="Show DB summary")

    args = parser.parse_args()

    if args.cmd == "collect":
        cmd_collect(args)
    elif args.cmd == "price":
        cmd_price(args)
    elif args.cmd == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
