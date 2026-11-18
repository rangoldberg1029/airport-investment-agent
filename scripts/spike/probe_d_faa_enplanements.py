"""Probe D: FAA Passenger Boarding (Enplanement) data - commercial service airports.

Run:  python scripts/spike/probe_d_faa_enplanements.py
"""

import io

import pandas as pd
import requests

TEST_AIRPORTS = ["BOS", "LAX", "SNA", "ANC", "SFO"]
FILES = {
    "CY2025": "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/"
              "arp-cy2025-commercial-service-enplanements.xlsx",
    "CY2024": "https://www.faa.gov/sites/faa.gov/files/2026-09/arp-cy2024-commercial-service-enplanements.xlsx",
}

if __name__ == "__main__":
    for label, url in FILES.items():
        resp = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content))
        print(f"\n=== {label}: {len(df)} airports  {url}")
        print("columns:", list(df.columns))
        print(df[df["Locid"].isin(TEST_AIRPORTS)].to_string(index=False))
