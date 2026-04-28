"""
main.py — LotSense CLI

Commands:
    collect   Fetch auction listings from Copart and save to DB
    price     Estimate a fair price for a car (requires Copart data — coming soon)
    stats     Show DB summary
    bot       Run the automated buyer conversation bot
"""

import argparse
import sys
from pathlib import Path

from data.db import get_listings, listing_count, open_db, upsert_listings


# ------------------------------------------------------------------
# collect
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# price
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# stats
# ------------------------------------------------------------------

def cmd_stats(args) -> None:
    with open_db() as conn:
        total = listing_count(conn)
        listings = get_listings(conn)

    by_source: dict[str, int] = {}
    by_make_model: dict[str, int] = {}
    for l in listings:
        by_source[l.source] = by_source.get(l.source, 0) + 1
        if l.make and l.model:
            key = f"{l.make} {l.model}"
            by_make_model[key] = by_make_model.get(key, 0) + 1

    print(f"\n  DB: {total} listing(s)\n")

    if by_source:
        print(f"  {'Source':<12} {'Count':>5}")
        print(f"  {'-' * 19}")
        for src, count in sorted(by_source.items()):
            print(f"  {src:<12} {count:>5}")
        print()

    if by_make_model:
        print(f"  {'Make / Model':<28} {'Count':>5}")
        print(f"  {'-' * 35}")
        for name, count in sorted(by_make_model.items(), key=lambda x: -x[1]):
            print(f"  {name:<28} {count:>5}")
    print()


# ------------------------------------------------------------------
# bot
# ------------------------------------------------------------------

def cmd_bot(args) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Copy bot/listings_example.json to bot/listings.json and fill in your listing details.")
        sys.exit(1)

    if args.mock:
        if not args.script:
            print("--mock requires --script <path/to/script.jsonl>")
            sys.exit(1)
        from bot.adapter import MockAdapter
        from bot.dispatcher import BotDispatcher, load_config

        config = load_config(config_path)
        adapter = MockAdapter(args.script)
        dispatcher = BotDispatcher(
            adapter,
            config["cars"],
            config["schedule"],
            poll_interval_s=0,
        )
        print(f"[bot] mock mode — replaying {args.script}")
        while not adapter.exhausted():
            for msg in adapter.poll():
                dispatcher._handle(msg)
        print("[bot] script exhausted")

    else:
        try:
            from bot.adapter import PlaywrightAdapter
        except ImportError:
            print("playwright not installed. Run: pip install playwright && playwright install chromium")
            sys.exit(1)

        from bot.dispatcher import BotDispatcher, load_config

        config = load_config(config_path)
        try:
            adapter = PlaywrightAdapter(headless=not args.headed, debug=args.debug)
        except EnvironmentError as e:
            print(f"Auth error: {e}")
            sys.exit(1)

        dispatcher = BotDispatcher(
            adapter,
            config["cars"],
            config["schedule"],
            poll_interval_s=args.interval,
        )
        dispatcher.run()


# ------------------------------------------------------------------
# CLI wiring
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="lotsense", description="LotSense — auction price estimator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # collect
    p_collect = sub.add_parser("collect", help="Fetch Copart auction listings and save to DB")
    p_collect.add_argument("--make",         default=None,  help="Vehicle make (e.g. Toyota)")
    p_collect.add_argument("--model",        default=None,  help="Vehicle model (e.g. Camry)")
    p_collect.add_argument("--year",         type=int, default=None, help="Target year — collects ±2 years automatically")
    p_collect.add_argument("--min-year",     type=int, default=None)
    p_collect.add_argument("--max-year",     type=int, default=None)
    p_collect.add_argument("--min-mileage",  type=int, default=None)
    p_collect.add_argument("--max-mileage",  type=int, default=None)
    p_collect.add_argument("--pages",        type=int, default=10)
    p_collect.add_argument("--debug",        action="store_true")

    # price
    p_price = sub.add_parser("price", help="Estimate a fair listing price")
    p_price.add_argument("make", help="Vehicle make (e.g. Toyota)")
    p_price.add_argument("model", help="Vehicle model (e.g. Camry)")
    p_price.add_argument("year", type=int, help="Model year")
    p_price.add_argument("mileage", type=int, help="Odometer reading in miles")
    p_price.add_argument("--discount", type=float, default=0.75, help="Market discount vs ACV (default: 0.75)")

    # stats
    sub.add_parser("stats", help="Show DB summary")

    # bot
    p_bot = sub.add_parser("bot", help="Run the automated buyer conversation bot")
    p_bot.add_argument("--config",   default="bot/listings.json", help="Path to listings config (default: bot/listings.json)")
    p_bot.add_argument("--mock",     action="store_true",          help="Replay a JSONL script instead of connecting to Facebook")
    p_bot.add_argument("--script",   default=None,                 help="Path to mock script (required with --mock)")
    p_bot.add_argument("--interval", type=int,   default=30,       help="Poll interval in seconds (default: 30)")
    p_bot.add_argument("--headed",   action="store_true",          help="Show the browser window (useful for debugging)")
    p_bot.add_argument("--debug",    action="store_true",          help="Save screenshots on each poll step")

    args = parser.parse_args()

    if args.cmd == "collect":
        cmd_collect(args)
    elif args.cmd == "price":
        cmd_price(args)
    elif args.cmd == "stats":
        cmd_stats(args)
    elif args.cmd == "bot":
        cmd_bot(args)


if __name__ == "__main__":
    main()
