"""OurAirports airports.csv + runways.csv.

-> `ourairports_us`: one row per US airport with scheduled service (runway summary = all open runways)
-> `runways`: one row per open runway at those airports (length/surface), so scoring can apply
   its own minimum-length rule (scoring/config.py MIN_COMMERCIAL_RUNWAY_FT).
"""

import io
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from data.sources.common import Provenance, load_with_fallback

SOURCE = "ourairports_us"
BASE = "https://davidmegginson.github.io/ourairports-data/"
URL = BASE + "airports.csv + " + BASE + "runways.csv"
ACCESS = "direct download"
CACHE_FILE = "ourairports_us.parquet"
RUNWAYS_SOURCE = "runways"
RUNWAYS_URL = BASE + "runways.csv"
RUNWAYS_CACHE_FILE = "runways_us.parquet"


@lru_cache(maxsize=4)
def fetch_csv(name: str) -> pd.DataFrame:
    """Download once per process (both outputs use the same two files)."""
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


def build_runways(airports: pd.DataFrame, runways: pd.DataFrame) -> pd.DataFrame:
    """Open runways at US commercial airports, keyed by IATA code."""
    us = filter_us_commercial(airports)[["iata", "ident"]]
    open_rw = runways[runways["closed"] == 0].merge(us, left_on="airport_ident", right_on="ident")
    # label "LE/HE" (e.g. 07L/25R); helipads have only one end (e.g. H1), unnamed ones get their id
    le, he = open_rw["le_ident"], open_rw["he_ident"]
    label = (le + "/" + he).fillna(le).fillna("id " + open_rw["id"].astype(str))
    return pd.DataFrame({
        "iata": open_rw["iata"],
        "runway": label,
        "length_ft": open_rw["length_ft"],
        "width_ft": open_rw["width_ft"],
        "surface": open_rw["surface"],
    }).sort_values(["iata", "length_ft"], ascending=[True, False]).reset_index(drop=True)


def load_runways(mode: str, processed_dir: Path) -> tuple[pd.DataFrame, Provenance]:
    return load_with_fallback(
        RUNWAYS_SOURCE, RUNWAYS_URL, ACCESS,
        fetch=lambda: build_runways(fetch_csv("airports.csv"), fetch_csv("runways.csv")),
        cache_path=processed_dir / RUNWAYS_CACHE_FILE,
        mode=mode,
    )


def load(mode: str, processed_dir: Path) -> tuple[pd.DataFrame, Provenance]:
    return load_with_fallback(
        SOURCE, URL, ACCESS,
        fetch=lambda: build(fetch_csv("airports.csv"), fetch_csv("runways.csv")),
        cache_path=processed_dir / CACHE_FILE,
        mode=mode,
    )
