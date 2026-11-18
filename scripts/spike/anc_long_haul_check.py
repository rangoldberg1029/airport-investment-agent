"""Spike sanity check: long-haul share of ANC passenger departures from T-100 Segment (source B).

Not the final implementation. Passenger service = CLASS in (F, L) and AIRCRAFT_CONFIG != 2
(2 = freighter). Distance thresholds are assumptions, printed as a sensitivity table.

Run:  python scripts/spike/anc_long_haul_check.py
"""

import zipfile
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "t100_segment"
AIRPORT = "ANC"
THRESHOLDS_MI = [2500, 3000, 4000]
COLS = ["YEAR", "MONTH", "ORIGIN", "DEST", "DEST_COUNTRY", "CLASS", "AIRCRAFT_CONFIG",
        "DISTANCE", "DEPARTURES_PERFORMED", "SEATS", "PASSENGERS"]


def load_all() -> pd.DataFrame:
    """Read every zip once per distinct year (duplicate downloads are dropped)."""
    frames = []
    for path in sorted(RAW_DIR.glob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            frames.append(pd.read_csv(zf.open(csv_name), usecols=COLS, low_memory=False))
    return pd.concat(frames).drop_duplicates()


def long_haul_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS_MI:
        long = df[df.DISTANCE >= t]
        rows.append({
            "threshold_mi": t,
            "departures_share_%": 100 * long.DEPARTURES_PERFORMED.sum() / df.DEPARTURES_PERFORMED.sum(),
            "seat_weighted_share_%": 100 * long.SEATS.sum() / df.SEATS.sum(),
            "long_haul_departures": int(long.DEPARTURES_PERFORMED.sum()),
        })
    return pd.DataFrame(rows).round(2)


WINDOW = ((2025, 9), (2026, 8))  # standard trailing-12-month window (inclusive)


def in_window(df: pd.DataFrame) -> pd.Series:
    ym = df.YEAR * 100 + df.MONTH
    (y0, m0), (y1, m1) = WINDOW
    return ym.between(y0 * 100 + m0, y1 * 100 + m1)


if __name__ == "__main__":
    data = load_all()
    data = data[in_window(data)]
    months = sorted({(y, m) for y, m in zip(data.YEAR, data.MONTH)})
    print(f"Window: {months[0]} .. {months[-1]} ({len(months)} months)")

    pax = data[(data.ORIGIN == AIRPORT) & data.CLASS.isin(["F", "L"]) & (data.AIRCRAFT_CONFIG != 2)]
    print(f"\n=== {AIRPORT} passenger departures performed = {int(pax.DEPARTURES_PERFORMED.sum()):,}")
    print(long_haul_table(pax).to_string(index=False))
    top = (pax[pax.DISTANCE >= THRESHOLDS_MI[0]]
           .groupby(["DEST", "DEST_COUNTRY", "DISTANCE"]).DEPARTURES_PERFORMED.sum()
           .sort_values(ascending=False).head(8))
    print(f"Top destinations >= {THRESHOLDS_MI[0]} mi:\n{top.to_string()}")

    all_anc = data[data.ORIGIN == AIRPORT]
    cargo_share = 100 * (1 - pax.DEPARTURES_PERFORMED.sum() / all_anc.DEPARTURES_PERFORMED.sum())
    print(f"\nFor context: all-cargo / freighter departures = {cargo_share:.1f}% of all ANC departures (excluded above).")
