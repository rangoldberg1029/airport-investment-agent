"""Live-fetch-with-cache-fallback logic shared by all online sources."""

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from data.db import month_span

log = logging.getLogger("etl")

MODES = ("live", "offline", "from_raw")


@dataclass
class Provenance:
    """Where a table's data came from in this ETL run."""

    source: str
    url: str
    access: str          # "public API", "direct download", "manual download"
    mode: str            # "live", "cache" or "raw"
    data_period: str
    rows: int
    warning: str = ""
    cache_written: bool = field(default=False)
    built: str = field(default_factory=lambda: date.today().isoformat())


def load_with_fallback(
    source: str,
    url: str,
    access: str,
    fetch: Callable[[], pd.DataFrame],
    cache_path: Path,
    mode: str,
) -> tuple[pd.DataFrame, Provenance]:
    """Fetch live data; fall back to the processed cache on failure.

    mode="offline"  -> cache only.
    mode="live"     -> live, cache on failure, cache is NOT rewritten.
    mode="from_raw" -> live, cache on failure, cache IS rewritten on success.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")

    warning = ""
    if mode != "offline":
        try:
            df = fetch()
            if df.empty:
                raise ValueError("live fetch returned 0 rows")
            written = mode == "from_raw"
            if written:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path, index=False, compression="zstd")
            return df, Provenance(source, url, access, "live", month_span(df), len(df), cache_written=written)
        except Exception as exc:  # network errors, format changes, ...
            warning = f"live fetch failed ({type(exc).__name__}: {exc}); using cached copy"
            log.warning("%s: %s", source, warning)

    if not cache_path.exists():
        raise FileNotFoundError(f"{source}: no cached copy at {cache_path} and live fetch unavailable")
    df = pd.read_parquet(cache_path)
    return df, Provenance(source, url, access, "cache", month_span(df), len(df), warning=warning)
