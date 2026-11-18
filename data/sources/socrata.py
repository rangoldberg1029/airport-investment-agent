"""BTS Socrata 'AFF - T100 Segment Summary By Origin Airport' (r495-tyji) -> airport x month.

All carriers including all-cargo flights (so departures are inflated at cargo hubs like ANC).
Used as a cross-check / fallback for passengers and seats.
"""

from pathlib import Path

import pandas as pd
import requests

from data.sources.common import Provenance, load_with_fallback

SOURCE = "socrata_airport_months"
URL = "https://data.bts.gov/resource/r495-tyji.json"
ACCESS = "public API"
CACHE_FILE = "socrata_airport_months.parquet"
START = "2024-01-01T00:00:00"
PAGE_SIZE = 50_000

SELECT = "origin_airport_code, reporting_month, total_departures, total_passengers, total_seats, total_freight_lbs"


def fetch_records(start: str = START) -> list[dict]:
    """Page through the SODA API from `start` onwards."""
    records: list[dict] = []
    offset = 0
    while True:
        page = requests.get(
            URL,
            params={
                "$select": SELECT,
                "$where": f"reporting_month >= '{start}'",
                "$order": "reporting_month, origin_airport_code",
                "$limit": PAGE_SIZE,
                "$offset": offset,
            },
            timeout=120,
        )
        page.raise_for_status()
        batch = page.json()
        records.extend(batch)
        if len(batch) < PAGE_SIZE:
            return records
        offset += PAGE_SIZE


def parse_records(records: list[dict]) -> pd.DataFrame:
    """SODA returns strings; convert to typed columns at airport x month grain."""
    raw = pd.DataFrame.from_records(records)
    month = pd.to_datetime(raw["reporting_month"])
    out = pd.DataFrame({
        "year": month.dt.year.astype(int),
        "month": month.dt.month.astype(int),
        "airport": raw["origin_airport_code"].astype(str),
    })
    for src, dst in [("total_departures", "departures"), ("total_passengers", "passengers"),
                     ("total_seats", "seats"), ("total_freight_lbs", "freight_lbs")]:
        out[dst] = pd.to_numeric(raw.get(src), errors="coerce").fillna(0).astype("int64")
    return out


def load(mode: str, processed_dir: Path) -> tuple[pd.DataFrame, Provenance]:
    return load_with_fallback(
        SOURCE, URL, ACCESS,
        fetch=lambda: parse_records(fetch_records()),
        cache_path=processed_dir / CACHE_FILE,
        mode=mode,
    )
