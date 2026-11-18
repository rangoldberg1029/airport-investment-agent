"""FAA Passenger Boarding (Enplanement) data, commercial service airports -> airport x calendar year.

Each yearly file holds two years ("CY 25 Enplanements", "CY 24 Enplanements"). The newer file's
primary year wins; the oldest year comes only from a file's second column (no hub/rank for it).
"""

import io
import re
from pathlib import Path

import pandas as pd
import requests

from data.sources.common import Provenance, load_with_fallback

SOURCE = "faa_enplanements"
ACCESS = "direct download"
CACHE_FILE = "faa_enplanements.parquet"
FILES = [  # newest first
    "https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/"
    "arp-cy2025-commercial-service-enplanements.xlsx",
    "https://www.faa.gov/sites/faa.gov/files/2026-09/arp-cy2024-commercial-service-enplanements.xlsx",
]
URL = " + ".join(FILES)


def fetch_excel(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return pd.read_excel(io.BytesIO(resp.content))


def parse_enplanements(sheet: pd.DataFrame) -> pd.DataFrame:
    """Wide FAA sheet -> long rows (airport, year, enplanements, hub, state, rank, is_primary)."""
    year_cols = sorted(
        (c for c in sheet.columns if re.fullmatch(r"CY \d{2} Enplanements", str(c))),
        key=lambda c: int(str(c)[3:5]),
        reverse=True,
    )
    if len(year_cols) != 2:
        raise ValueError(f"expected two 'CY xx Enplanements' columns, got {year_cols}")
    sheet = sheet[sheet["Locid"].notna() & pd.to_numeric(sheet[year_cols[0]], errors="coerce").notna()]

    rows = []
    for col, primary in [(year_cols[0], True), (year_cols[1], False)]:
        rows.append(pd.DataFrame({
            "airport": sheet["Locid"].astype(str).str.strip(),
            "year": 2000 + int(str(col)[3:5]),
            "enplanements": pd.to_numeric(sheet[col], errors="coerce"),
            "hub": sheet["Hub"] if primary else None,
            "state": sheet["ST"],
            "rank": pd.to_numeric(sheet["Rank"], errors="coerce") if primary else float("nan"),
            "is_primary": primary,
        }))
    return pd.concat(rows, ignore_index=True).dropna(subset=["enplanements"])


def combine_years(parsed: list[pd.DataFrame]) -> pd.DataFrame:
    """Keep one row per airport-year: primary rows first, then the newest file."""
    allrows = pd.concat(parsed, ignore_index=True)
    allrows = allrows.sort_values("is_primary", ascending=False, kind="stable")
    out = allrows.drop_duplicates(["airport", "year"]).drop(columns="is_primary")
    out["enplanements"] = out["enplanements"].astype("int64")
    out["rank"] = out["rank"].astype("Int64")
    return out.sort_values(["year", "airport"]).reset_index(drop=True)


def fetch_all() -> pd.DataFrame:
    return combine_years([parse_enplanements(fetch_excel(u)) for u in FILES])


def load(mode: str, processed_dir: Path) -> tuple[pd.DataFrame, Provenance]:
    return load_with_fallback(SOURCE, URL, ACCESS, fetch_all, processed_dir / CACHE_FILE, mode)
