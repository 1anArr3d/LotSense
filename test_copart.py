"""
test_copart.py — Quick smoke test for the Copart fetcher.

Run: python test_copart.py
"""

from data.copart import CopartCollector


def main() -> None:
    collector = CopartCollector()

    print("Fetching Toyota Camry (2010–2016) salvage lots — AutoGraded only …")
    listings = list(collector.search(
        make="Toyota",
        model="Camry",
        min_year=2010,
        max_year=2016,
        graded_only=True,
        max_pages=1,
        debug=True,
    ))

    print(f"\nFetched {len(listings)} listing(s)\n")

    for l in listings[:5]:
        print(f"  [{l.lot_number}] {l.year} {l.make} {l.model}")
        print(f"    Mileage:    {l.mileage:,}")
        print(f"    ACV:        ${l.retail_estimate / 100:,.0f}" if l.retail_estimate else "    ACV:        —")
        print(f"    Hammer:     ${l.hammer_price / 100:,.0f}")
        print(f"    Condition:  {l.condition_score}")
        print(f"    Damage:     {l.damage_description}")
        print(f"    Location:   {l.city}, {l.state}")
        print()

    if not listings:
        print("No listings returned — check filters.")


if __name__ == "__main__":
    main()
