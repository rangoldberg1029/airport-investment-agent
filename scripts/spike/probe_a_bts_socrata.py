"""Probe A: BTS Socrata API (data.bts.gov) - which T-100 datasets exist and at what grain.

Run:  python scripts/spike/probe_a_bts_socrata.py
"""

import requests

TEST_AIRPORTS = ["BOS", "LAX", "SNA", "ANC", "SFO"]
CATALOG = "https://api.us.socrata.com/api/catalog/v1"
AIRPORT_DATASET = "r495-tyji"  # AFF - T100 Segment Summary By Origin Airport


def list_t100_datasets() -> None:
    """Search the Socrata catalog for T-100 datasets and print id, name and columns."""
    resp = requests.get(CATALOG, params={"domains": "data.bts.gov", "q": "T100", "limit": 50}, timeout=30)
    resp.raise_for_status()
    for item in resp.json()["results"]:
        res = item["resource"]
        print(f"\n{res['id']} | {res['name']} | type={res['type']} | updated={res['data_updated_at'][:10]}")
        if res["type"] == "dataset":
            print("  columns:", res["columns_field_name"])


def show_airport_dataset() -> None:
    """Print time coverage and the latest month for the test airports."""
    url = f"https://data.bts.gov/resource/{AIRPORT_DATASET}.json"
    span = requests.get(url, params={"$select": "min(reporting_month), max(reporting_month)"}, timeout=30).json()[0]
    print(f"\n=== {AIRPORT_DATASET} coverage: {span['min_reporting_month'][:7]} .. {span['max_reporting_month'][:7]}")

    latest = span["max_reporting_month"]
    airports = ",".join(f"'{a}'" for a in TEST_AIRPORTS)
    rows = requests.get(
        url,
        params={
            "$select": "origin_airport_code, reporting_month, total_departures, total_passengers, "
            "total_seats, total_load_factor, total_freight_lbs, domestic_passengers, outbound_international_1",
            "$where": f"origin_airport_code in({airports}) and reporting_month='{latest}'",
        },
        timeout=30,
    ).json()
    print("(outbound_international_1 = outbound international passengers)")
    for row in rows:
        print(" ", row)


if __name__ == "__main__":
    list_t100_datasets()
    show_airport_dataset()
