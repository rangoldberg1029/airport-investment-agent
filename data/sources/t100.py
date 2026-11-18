"""BTS T-100 Segment (All Carriers): manual TranStats zips -> passenger route x month table.

Raw zips (data/raw/t100_segment/*.zip) are only read with --from-raw; otherwise the committed
data/processed/t100_routes.parquet is used.
"""

import zipfile
from pathlib import Path

import pandas as pd

from data.db import month_span
from data.sources.common import Provenance

SOURCE = "t100_routes"
URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4"
ACCESS = "manual download"
CACHE_FILE = "t100_routes.parquet"

RAW_COLUMNS = [
    "YEAR", "MONTH", "ORIGIN", "DEST", "ORIGIN_STATE_ABR", "DEST_STATE_ABR",
    "ORIGIN_COUNTRY", "DEST_COUNTRY", "CLASS", "AIRCRAFT_CONFIG", "DISTANCE",
    "DEPARTURES_SCHEDULED", "DEPARTURES_PERFORMED", "SEATS", "PASSENGERS",
]
PASSENGER_CLASSES = ("F", "L")   # scheduled + non-scheduled passenger/cargo service
FREIGHTER_CONFIG = 2             # AIRCRAFT_CONFIG 2 = freighter
KEYS = ["year", "month", "origin", "dest", "origin_state", "dest_state", "origin_country", "dest_country"]


def read_raw_zips(raw_dir: Path) -> pd.DataFrame:
    """Read every T-100 zip in raw_dir (only the needed columns); duplicate rows are dropped."""
    paths = sorted(raw_dir.glob("*.zip"))
    if not paths:
        raise FileNotFoundError(f"no T-100 zips in {raw_dir} (see data/raw/README.md for manual download)")
    frames = []
    for path in paths:
        with zipfile.ZipFile(path) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            frames.append(pd.read_csv(zf.open(csv_name), usecols=RAW_COLUMNS, low_memory=False))
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def filter_passenger_service(raw: pd.DataFrame) -> pd.DataFrame:
    """Agreed filter: CLASS in (F, L) and AIRCRAFT_CONFIG != 2; US origins only."""
    keep = (
        raw["CLASS"].isin(PASSENGER_CLASSES)
        & (raw["AIRCRAFT_CONFIG"] != FREIGHTER_CONFIG)
        & (raw["ORIGIN_COUNTRY"] == "US")
    )
    return raw[keep]


def aggregate_routes(pax: pd.DataFrame) -> pd.DataFrame:
    """Sum carriers/aircraft into origin x dest x month.

    dep_scheduled_us / dep_performed_us only include rows with DEPARTURES_SCHEDULED > 0
    (foreign carriers report 0 scheduled), per the agreed completion-gap rule.
    """
    has_schedule = pax["DEPARTURES_SCHEDULED"] > 0
    df = pd.DataFrame({
        "year": pax["YEAR"].astype(int),
        "month": pax["MONTH"].astype(int),
        "origin": pax["ORIGIN"],
        "dest": pax["DEST"],
        "origin_state": pax["ORIGIN_STATE_ABR"],
        "dest_state": pax["DEST_STATE_ABR"],
        "origin_country": pax["ORIGIN_COUNTRY"],
        "dest_country": pax["DEST_COUNTRY"],
        "distance_mi": pax["DISTANCE"],
        "dep_performed": pax["DEPARTURES_PERFORMED"],
        "dep_scheduled_us": pax["DEPARTURES_SCHEDULED"].where(has_schedule, 0),
        "dep_performed_us": pax["DEPARTURES_PERFORMED"].where(has_schedule, 0),
        "seats": pax["SEATS"],
        "passengers": pax["PASSENGERS"],
    })
    out = (
        df.groupby(KEYS, dropna=False)
        .agg(
            distance_mi=("distance_mi", "max"),
            dep_performed=("dep_performed", "sum"),
            dep_scheduled_us=("dep_scheduled_us", "sum"),
            dep_performed_us=("dep_performed_us", "sum"),
            seats=("seats", "sum"),
            passengers=("passengers", "sum"),
        )
        .reset_index()
    )
    out = out[(out["dep_performed"] > 0) | (out["passengers"] > 0)]
    count_cols = ["dep_performed", "dep_scheduled_us", "dep_performed_us", "seats", "passengers"]
    out[count_cols] = out[count_cols].astype("int64")
    return out.reset_index(drop=True)


def load(mode: str, processed_dir: Path, raw_dir: Path) -> tuple[pd.DataFrame, Provenance]:
    """from_raw -> rebuild from zips and rewrite the cache; otherwise read the cache."""
    cache = processed_dir / CACHE_FILE
    if mode == "from_raw":
        df = aggregate_routes(filter_passenger_service(read_raw_zips(raw_dir)))
        processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False, compression="zstd")
        return df, Provenance(SOURCE, URL, ACCESS, "raw", month_span(df), len(df), cache_written=True)
    df = pd.read_parquet(cache)
    return df, Provenance(SOURCE, URL, ACCESS, "cache", month_span(df), len(df))
