"""
main.py — LotSense CLI

Commands:
    collect   Fetch auction listings from Copart and save to DB
    salvage   Fetch SalvageBid Texas listings and save to DB
    price     Estimate ACV price range for a vehicle (uses auction comps)
    mileage   Estimate mileage range or validate an odometer reading
    stats     Show DB summary
"""

import argparse
import sys

from data.db import get_listings, listing_count, open_db, upsert_listings


def cmd_collect(args) -> None:
    from data.copart import CopartCollector

    collector = CopartCollector()
    label = f"{args.make or ''} {args.model or ''}".strip() or "all vehicles"
    print(f"Collecting {label} from Copart …")

    min_year = args.min_year or (args.year - 1 if args.year else None)
    max_year = args.max_year or (args.year + 1 if args.year else None)

    listings = list(collector.search(
        make=args.make,
        model=args.model,
        min_year=min_year,
        max_year=max_year,
        min_mileage=args.min_mileage,
        max_mileage=args.max_mileage,
        max_pages=args.pages,
        debug=args.debug,
    ))

    if not listings:
        print("No listings returned.")
        sys.exit(1)

    with open_db() as conn:
        written = upsert_listings(conn, listings)
        total = listing_count(conn)

    print(f"{written} listing(s) saved  ({total} total in DB).")


def cmd_salvage(args) -> None:
    from data.salvagebid import SalvageBidCollector

    collector = SalvageBidCollector()
    print("Collecting Texas listings from SalvageBid …")

    listings = list(collector.search(
        max_pages=args.pages or None,
        actual_odometer_only=not args.all_odometer,
        debug=args.debug,
    ))

    if not listings:
        print("No listings returned.")
        sys.exit(1)

    with open_db() as conn:
        written = upsert_listings(conn, listings)
        total = listing_count(conn)

    vins = sum(1 for l in listings if l.vin)
    print(f"{written} listing(s) saved  ({vins} with VIN)  ({total} total in DB).")


def cmd_price(args) -> None:
    from pricing.estimator import Estimator

    try:
        est = Estimator.from_db(args.make, args.model, target_year=args.year)
    except ValueError as e:
        print(f"Not enough data: {e}")
        sys.exit(1)

    result = est.predict(year=args.year, mileage=args.mileage, market_discount=args.discount)
    print(f"\n  {args.year} {args.make} {args.model}  —  {args.mileage:,} miles")
    print(f"\n  Street price range  (ACV × {args.discount}):")
    print(f"    Low:  ${result['low']:,}")
    print(f"    Mid:  ${result['mid']:,}")
    print(f"    High: ${result['high']:,}")
    print(f"\n  Book value (ACV):  ${result['acv_mid']:,}")
    print(f"  Comps: {result['n_comps']}  Confidence: {result['confidence']}\n")


def cmd_mileage(args) -> None:
    from mileage.predict_vin import main as predict_main
    import sys
    argv = [args.vin]
    if args.stated:
        argv += ["--stated", str(args.stated)]
    sys.argv = ["predict_vin"] + argv
    predict_main()


def cmd_stats(args) -> None:
    with open_db() as conn:
        total = listing_count(conn)
        listings = get_listings(conn)

        vin_count = conn.execute(
            "SELECT COUNT(DISTINCT vin) FROM odometer_history WHERE vin IS NOT NULL"
        ).fetchone()[0]
        reading_count = conn.execute(
            "SELECT COUNT(*) FROM odometer_history"
        ).fetchone()[0]

    by_source: dict[str, int] = {}
    by_make_model: dict[str, int] = {}
    for l in listings:
        by_source[l.source] = by_source.get(l.source, 0) + 1
        if l.make and l.model:
            key = f"{l.make} {l.model}"
            by_make_model[key] = by_make_model.get(key, 0) + 1

    print(f"\n  Listings: {total:,}")
    print(f"  VINs with inspection history: {vin_count:,}  ({reading_count:,} readings)\n")

    if by_source:
        print(f"  {'Source':<16} {'Count':>6}")
        print(f"  {'-' * 24}")
        for src, count in sorted(by_source.items()):
            print(f"  {src:<16} {count:>6,}")
        print()

    if by_make_model:
        print(f"  {'Make / Model':<28} {'Count':>5}")
        print(f"  {'-' * 35}")
        for name, count in sorted(by_make_model.items(), key=lambda x: -x[1]):
            print(f"  {name:<28} {count:>5}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="lotsense", description="LotSense — auction price estimator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Fetch Copart auction listings and save to DB")
    p_collect.add_argument("--make",        default=None, help="Vehicle make (e.g. Toyota)")
    p_collect.add_argument("--model",       default=None, help="Vehicle model (e.g. Camry)")
    p_collect.add_argument("--year",        type=int, default=None, help="Target year — collects ±1 year automatically")
    p_collect.add_argument("--min-year",    type=int, default=None)
    p_collect.add_argument("--max-year",    type=int, default=None)
    p_collect.add_argument("--min-mileage", type=int, default=None)
    p_collect.add_argument("--max-mileage", type=int, default=None)
    p_collect.add_argument("--pages",       type=int, default=10)
    p_collect.add_argument("--debug",       action="store_true")

    p_salvage = sub.add_parser("salvage", help="Fetch SalvageBid Texas listings and save to DB")
    p_salvage.add_argument("--pages",        type=int, default=None)
    p_salvage.add_argument("--all-odometer", action="store_true", help="Include non-Actual odometer readings")
    p_salvage.add_argument("--debug",        action="store_true")

    p_price = sub.add_parser("price", help="Estimate ACV price range from auction comps")
    p_price.add_argument("make",    help="Vehicle make (e.g. Toyota)")
    p_price.add_argument("model",   help="Vehicle model (e.g. Camry)")
    p_price.add_argument("year",    type=int, help="Model year")
    p_price.add_argument("mileage", type=int, help="Odometer reading in miles")
    p_price.add_argument("--discount", type=float, default=0.70, help="Market discount vs ACV (default: 0.70)")

    p_mileage = sub.add_parser("mileage", help="Estimate current odometer reading for a VIN")
    p_mileage.add_argument("vin",            help="17-character VIN")
    p_mileage.add_argument("--stated", type=int, default=None, help="Stated mileage from listing (flags rollback if suspect)")

    sub.add_parser("stats", help="Show DB summary")

    args = parser.parse_args()

    if args.cmd == "collect":
        cmd_collect(args)
    elif args.cmd == "salvage":
        cmd_salvage(args)
    elif args.cmd == "price":
        cmd_price(args)
    elif args.cmd == "mileage":
        cmd_mileage(args)
    elif args.cmd == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
