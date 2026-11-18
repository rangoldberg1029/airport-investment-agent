"""Probe E: OurAirports airports.csv and runways.csv (public domain metadata).

Run:  python scripts/spike/probe_e_ourairports.py
"""

import io

import pandas as pd
import requests

TEST_AIRPORTS = ["BOS", "LAX", "SNA", "ANC", "SFO"]
BASE = "https://davidmegginson.github.io/ourairports-data/"


def fetch(name: str) -> pd.DataFrame:
    resp = requests.get(BASE + name, timeout=120)
    resp.raise_for_status()
    print(f"\n=== {name}: last-modified {resp.headers.get('last-modified')}")
    return pd.read_csv(io.StringIO(resp.text), low_memory=False)


if __name__ == "__main__":
    airports = fetch("airports.csv")
    print("columns:", list(airports.columns))
    test = airports[airports.iata_code.isin(TEST_AIRPORTS)]
    print(test[["ident", "iata_code", "name", "type", "iso_region", "municipality",
                "latitude_deg", "longitude_deg", "scheduled_service"]].to_string(index=False))

    runways = fetch("runways.csv")
    print("columns:", list(runways.columns))
    open_rw = runways[runways.airport_ident.isin(test.ident) & (runways.closed == 0)]
    print("\nOpen runways per airport:")
    print(open_rw.groupby("airport_ident").agg(runways=("id", "count"), longest_ft=("length_ft", "max")).to_string())
