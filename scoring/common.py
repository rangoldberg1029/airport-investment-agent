"""Shared helpers: the result envelope, window arithmetic and airport/region name resolution."""

import math
from datetime import date

import numpy as np
import pandas as pd

from scoring.config import AIRPORT_ALIASES, REGIONS

Month = tuple[int, int]
Window = tuple[Month, Month]


# ---------------------------------------------------------------- envelope

def to_builtin(obj):
    """Recursively convert numpy/pandas values to JSON-safe Python types (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else float(obj)
    if obj is pd.NA or obj is None:
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def envelope(data, sources: list[str], data_period: str,
             assumptions: list[str], caveats: list[str]) -> dict:
    """The standard tool result from CLAUDE.md."""
    return to_builtin({
        "data": data,
        "sources": list(dict.fromkeys(sources)),
        "data_period": data_period,
        "assumptions": list(dict.fromkeys(assumptions)),
        "caveats": list(dict.fromkeys(caveats)),
    })


def pct(x: float, digits: int = 2) -> float:
    """Fraction -> percent, rounded (NaN stays NaN)."""
    return round(100 * x, digits) if x == x else float("nan")


def rnd(x: float, digits: int = 3) -> float:
    return round(float(x), digits) if x == x else float("nan")


# ---------------------------------------------------------------- windows

def shift_month(m: Month, months: int) -> Month:
    idx = m[0] * 12 + (m[1] - 1) + months
    return idx // 12, idx % 12 + 1


def in_window(df: pd.DataFrame, window: Window) -> pd.Series:
    ym = df["year"] * 100 + df["month"]
    (y0, m0), (y1, m1) = window
    return ym.between(y0 * 100 + m0, y1 * 100 + m1)


def month_label(m: Month) -> str:
    return date(m[0], m[1], 1).strftime("%b %Y")


def window_label(window: Window) -> str:
    return f"{month_label(window[0])} - {month_label(window[1])}"


# ---------------------------------------------------------------- name resolution

def resolve_airport(name: str, airports: pd.DataFrame) -> tuple[str, str | None]:
    """Return (IATA, assumption-or-None). Order: IATA code, alias map, exact city/name match."""
    key = name.strip().lower().replace(".", "")
    codes = set(airports["iata"])
    if key.upper() in codes:
        return key.upper(), None
    if key in AIRPORT_ALIASES:
        iata, alternatives = AIRPORT_ALIASES[key]
        if alternatives:
            return iata, f"'{name}' interpreted as {iata} (other airports in the area: {', '.join(alternatives)})"
        return iata, f"'{name}' interpreted as {iata}"
    hits = airports[(airports["city"].str.lower() == key) | (airports["name"].str.lower() == key)]
    if len(hits) == 1:
        iata = hits.iloc[0]["iata"]
        return iata, f"'{name}' interpreted as {iata} ({hits.iloc[0]['name']})"
    if len(hits) > 1:
        options = ", ".join(f"{r.iata} ({r.name})" for r in hits.itertuples())
        raise ValueError(f"'{name}' is ambiguous: {options}. Please specify the airport code.")
    raise ValueError(f"unknown airport '{name}'. Use an IATA code such as BOS or LAX.")


def resolve_region(region: str) -> tuple[tuple[str, ...], str]:
    """Region name -> state codes and an assumption string stating the definition used."""
    key = region.strip().lower()
    if key not in REGIONS:
        raise ValueError(f"unknown region '{region}'. Known: {', '.join(sorted(REGIONS))}")
    states = REGIONS[key]
    return states, f"{region.title()} defined as states {', '.join(states)}"
