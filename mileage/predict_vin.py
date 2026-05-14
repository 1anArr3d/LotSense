"""
predict_vin.py — VIN → odometer estimate, end-to-end.

Usage:
    python -m mileage.predict_vin VIN
    python -m mileage.predict_vin VIN --stated 85000
"""
import argparse
from datetime import date, datetime

from .inspection_scrape import _acquire_session, _lookup_vin
from .mileage_model import features_from_readings, predict

TODAY = date.today()

_VIN_YEAR = {
    'A': 2010, 'B': 2011, 'C': 2012, 'D': 2013, 'E': 2014,
    'F': 2015, 'G': 2016, 'H': 2017, 'J': 2018, 'K': 2019,
    'L': 2020, 'M': 2021, 'N': 2022, 'P': 2023, 'R': 2024,
    'S': 2025, 'T': 2026,
    '1': 2001, '2': 2002, '3': 2003, '4': 2004, '5': 2005,
    '6': 2006, '7': 2007, '8': 2008, '9': 2009,
}


def _year_from_vin(vin: str) -> int | None:
    if len(vin) >= 10:
        return _VIN_YEAR.get(vin[9].upper())
    return None


def _parse_readings(raw: list[dict]) -> list[tuple[date, int]]:
    readings = []
    for r in raw:
        try:
            d = datetime.strptime(r["date"], "%m/%d/%Y").date()
            readings.append((d, r["odometer"]))
        except ValueError:
            continue
    return readings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vin",            help="17-character VIN")
    ap.add_argument("--stated", type=int, default=None, help="Stated mileage from listing")
    ap.add_argument("--year",   type=int, default=None, help="Model year (auto-decoded from VIN if omitted)")
    args = ap.parse_args()

    year = args.year or _year_from_vin(args.vin)
    if year is None:
        ap.error("Could not decode year from VIN — provide --year manually")

    print(f"\n[lookup] {args.vin}  year: {year}")

    from data.db import open_db
    with open_db() as conn:
        cached = conn.execute(
            "SELECT inspection_date, mileage FROM odometer_history "
            "WHERE vin = ? ORDER BY inspection_date",
            (args.vin,)
        ).fetchall()

    if cached:
        print(f"  [db] {len(cached)} cached record(s)")
        raw = [{"date": r["inspection_date"], "odometer": r["mileage"]} for r in cached]
    else:
        session = _acquire_session()
        raw     = _lookup_vin(args.vin, session)

    if not raw:
        print("  No inspection records found — cannot estimate.")
        return

    print(f"  {len(raw)} inspection record(s):")
    for r in raw:
        print(f"    {r['date']}  {r['odometer']:,} mi")

    readings = _parse_readings(raw)
    feats    = features_from_readings(readings, year, TODAY)

    if feats is None:
        print("\n  *** DETECTED ROLLBACK or insufficient data — cannot estimate. ***")
        return

    result = predict(
        last_inspection_mileage = feats["last_inspection_mileage"],
        days_since_inspection   = feats["days_since_inspection"],
        lifetime_annual_rate    = feats["lifetime_annual_rate"],
        year                    = year,
        recent_annual_rate      = feats["recent_annual_rate"],
        rate_delta              = feats["rate_delta"],
        n_records               = feats["n_records"],
        inspection_span_days    = feats["inspection_span_days"],
        stated_mileage          = args.stated,
        rate_std                = feats["rate_std"],
        rate_trend              = feats["rate_trend"],
        recent_vs_early         = feats["recent_vs_early"],
        max_interval_rate       = feats["max_interval_rate"],
        min_interval_rate       = feats["min_interval_rate"],
    )

    print(f"\n  Last inspection:  {feats['last_inspection_mileage']:,} mi")
    print(f"  Days since:       {feats['days_since_inspection']} days")
    print(f"  Annual rate:      {feats['lifetime_annual_rate']:,} mi/yr lifetime  |  {feats['recent_annual_rate']:,} mi/yr recent")
    print(f"  Readings:         {feats['n_records']} over {feats['inspection_span_days']} days")
    print(f"  Rate std:         {feats['rate_std']:,.0f} mi/yr  |  trend: {feats['rate_trend']:+,.0f} mi/yr²")
    print(f"  Interval range:   {feats['min_interval_rate']:,.0f} – {feats['max_interval_rate']:,.0f} mi/yr")

    if result["suspected_rollback"] and args.stated:
        print(f"\n  *** SUSPECTED ROLLBACK — inspection shows {feats['last_inspection_mileage']:,} mi,"
              f" stated is {args.stated:,} mi ***")

    print(f"\n  Estimated range:  {result['low']:,} – {result['high']:,} mi")
    print(f"  Midpoint:         {result['mid']:,} mi")

    if args.stated:
        diff = result["mid"] - args.stated
        print(f"  Stated:           {args.stated:,} mi  ({diff:+,} vs midpoint)")


if __name__ == "__main__":
    main()
