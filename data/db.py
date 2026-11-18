"""Shared paths, the standard data window and a helper to write DataFrames into DuckDB."""

from pathlib import Path

import duckdb
import pandas as pd

from scoring.config import WINDOW

DATA_DIR = Path(__file__).resolve().parent
DB_PATH = DATA_DIR / "airports.duckdb"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# Standard trailing-12-month window lives in scoring/config.py (single source of truth).
WINDOW_START, WINDOW_END = WINDOW


def write_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    """Create (or replace) table `name` from a DataFrame."""
    con.register("_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
    con.unregister("_df")


def month_span(df: pd.DataFrame) -> str:
    """'YYYY-MM..YYYY-MM' for a frame with year/month columns, or 'YYYY..YYYY' with year only."""
    if df.empty:
        return "empty"
    if "month" in df.columns:
        ym = df["year"].astype(int) * 100 + df["month"].astype(int)
        lo, hi = int(ym.min()), int(ym.max())
        return f"{lo // 100}-{lo % 100:02d}..{hi // 100}-{hi % 100:02d}"
    if "year" in df.columns:
        return f"{int(df['year'].min())}..{int(df['year'].max())}"
    return "n/a"
