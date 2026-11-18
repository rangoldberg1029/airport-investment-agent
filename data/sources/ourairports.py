"""OurAirports airports.csv + runways.csv -> one row per US airport with scheduled service."""

import io
from pathlib import Path

import pandas as pd
import requests

from data.sources.common import Provenance, load_with_fallback

SOURCE = "ourairports_us"
BASE = "https://davidmegginson.github.io/ourairports-data/"
URL = BASE + "airports.csv + " + BASE + "runways.csv"
ACCESS = "direct download"
CACHE_FILE = "ourairports_us.parquet"


def fetch_csv(name: str) -> pd.DataFrame:
    resp = requests.get(BASE + name, timeout=120)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), low_memory=False)


def filter_us_commercial(airports: pd.DataFrame) -> pd.DataFrame:
    """US airports with an IATA code and scheduled service."""
    keep = (
        (airports["iso_country"] == "US")
        & airports["iata_code"].notna()
        & (airports["scheduled_service"] == "yes")
    )
    a = airports[keep]
    return pd.DataFrame({
        "iata": a["iata_code"].astype(str),
        "icao": a["icao_code"],
        "ident": a["ident"].astype(str),
        "name": a["name"],
        "city": a["municipality"],
        "state": a["iso_region"].astype(str).str.split("-").str[1],
        "lat": a["latitude_deg"],
        "lon": a["longitude_deg"],
        "type": a["type"],
    })


def summarize_runways(runways: pd.DataFrame) -> pd.DataFrame:
    """Open runways per airport ident: count and longest length."""
    open_rw = runways[runways["closed"] == 0]
    return (
        open_rw.groupby("airport_ident")
        .agg(runway_count=("id", "count"), longest_runway_ft=("length_ft", "max"))
        .reset_index()
        .rename(columns={"airport_ident": "ident"})
    )


def build(airports: pd.DataFrame, runways: pd.DataFrame) -> pd.DataFrame:
    out = filter_us_commercial(airports).merge(summarize_runways(runways), on="ident", how="left")
    out["runway_count"] = out["runway_count"].fillna(0).astype(int)
    return out.drop(columns="ident").drop_duplicates("iata").reset_index(drop=True)


def load(mode: str, processed_dir: Path) -> tuple[pd.DataFrame, Provenance]:
    return load_with_fallback(
        SOURCE, URL, ACCESS,
        fetch=lambda: build(fetch_csv("airports.csv"), fetch_csv("runways.csv")),
        cache_path=processed_dir / CACHE_FILE,
        mode=mode,
    )
