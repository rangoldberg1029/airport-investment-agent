"""Read-only access to data/airports.duckdb. The only place the scoring layer's input is loaded from disk."""

from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from data.db import DB_PATH

TABLES = ("t100_routes", "delay_cause", "socrata_airport_months", "faa_enplanements", "airports")


@dataclass
class Frames:
    """All tables as DataFrames. `cache` holds derived tables (e.g. the KPI table) per window."""

    t100: pd.DataFrame
    delay: pd.DataFrame
    socrata: pd.DataFrame
    faa: pd.DataFrame
    airports: pd.DataFrame
    cache: dict = field(default_factory=dict, repr=False)


def load_frames(db_path: Path = DB_PATH) -> Frames:
    if not Path(db_path).exists():
        raise FileNotFoundError(f"{db_path} not found - run `python -m data.etl --offline` first")
    with duckdb.connect(str(db_path), read_only=True) as con:
        t = {name: con.execute(f"SELECT * FROM {name}").df() for name in TABLES}
    return Frames(t["t100_routes"], t["delay_cause"], t["socrata_airport_months"],
                  t["faa_enplanements"], t["airports"])
