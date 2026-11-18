"""Data-quality checks run at the end of the ETL. Pure functions over DataFrames -> report lines."""

import pandas as pd

from data.db import WINDOW_END, WINDOW_START, month_span

CHECK_AIRPORTS = ["BOS", "LAX", "SNA", "ANC", "SFO"]
CROSS_CHECK_YEAR = 2025
MAX_DIFF = 0.05  # flag differences > 5%


def pct_diff(a: float, b: float) -> float:
    """Relative difference of a vs b (b is the reference)."""
    return (a - b) / b if b else float("nan")


def passenger_cross_check(
    t100: pd.DataFrame, socrata: pd.DataFrame, faa: pd.DataFrame,
    airports: list[str] = CHECK_AIRPORTS, year: int = CROSS_CHECK_YEAR,
) -> pd.DataFrame:
    """Calendar-year passengers per airport from three sources, with % differences vs FAA and flags."""
    t = t100[t100["year"] == year].groupby("origin")["passengers"].sum()
    s = socrata[socrata["year"] == year].groupby("airport")["passengers"].sum()
    f = faa[faa["year"] == year].set_index("airport")["enplanements"]
    out = pd.DataFrame({"t100_pax": t, "socrata_pax": s, "faa_enplanements": f}).reindex(airports)
    out["t100_vs_faa"] = [pct_diff(a, b) for a, b in zip(out.t100_pax, out.faa_enplanements)]
    out["socrata_vs_faa"] = [pct_diff(a, b) for a, b in zip(out.socrata_pax, out.faa_enplanements)]
    out["t100_vs_socrata"] = [pct_diff(a, b) for a, b in zip(out.t100_pax, out.socrata_pax)]
    diffs = out[["t100_vs_faa", "socrata_vs_faa", "t100_vs_socrata"]]
    out["flag"] = (diffs.abs() > MAX_DIFF).any(axis=1) | diffs.isna().any(axis=1)
    return out


def window_complete(df: pd.DataFrame, start: tuple[int, int] = WINDOW_START,
                    end: tuple[int, int] = WINDOW_END) -> list[tuple[int, int]]:
    """Return the months of the standard window that are missing from df (empty list = complete)."""
    have = set(zip(df["year"].astype(int), df["month"].astype(int)))
    missing, (y, m) = [], start
    while (y, m) <= end:
        if (y, m) not in have:
            missing.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return missing


def load_factor_violations(t100: pd.DataFrame, min_seats: int = 1000) -> int:
    """Airport-months (with >= min_seats seats) where passengers > seats. Tiny airports are
    excluded: a few report e.g. 18 passengers on 8 seats, which is reporting noise."""
    g = t100.groupby(["origin", "year", "month"])[["passengers", "seats"]].sum()
    g = g[g["seats"] >= min_seats]
    return int((g["passengers"] > g["seats"]).sum())


def delay_cause_sum_violations(delay: pd.DataFrame, max_rel_diff: float = 0.05) -> int:
    """Airport-months where the five cause columns differ from arr_delay_min by > max_rel_diff.
    Small source-side mismatches exist (spike: 0.013% of all minutes), so only material ones count."""
    causes = delay[["carrier_delay_min", "weather_delay_min", "nas_delay_min",
                    "security_delay_min", "late_aircraft_delay_min"]].sum(axis=1)
    total = delay["arr_delay_min"].where(delay["arr_delay_min"] > 0)
    return int(((causes - delay["arr_delay_min"]).abs() / total > max_rel_diff).sum())


def completion_violations(t100: pd.DataFrame, min_scheduled: int = 1000) -> list[str]:
    """Airports (>= min_scheduled US scheduled departures in total) with performed > scheduled * 1.05."""
    g = t100.groupby("origin")[["dep_scheduled_us", "dep_performed_us"]].sum()
    g = g[g["dep_scheduled_us"] >= min_scheduled]
    return sorted(g.index[g["dep_performed_us"] > g["dep_scheduled_us"] * 1.05])


def null_key_count(df: pd.DataFrame, keys: list[str]) -> int:
    return int(df[keys].isna().any(axis=1).sum())


TABLE_KEYS = {
    "t100_routes": ["year", "month", "origin", "dest"],
    "delay_cause": ["year", "month", "airport"],
    "socrata_airport_months": ["year", "month", "airport"],
    "faa_enplanements": ["year", "airport"],
    "airports": ["iata"],
}


def report(tables: dict[str, pd.DataFrame]) -> tuple[list[str], int]:
    """Build the printable report. Returns (lines, number_of_flags)."""
    lines, flags = ["", "=== DATA QUALITY REPORT ===", "", "Tables:"], 0
    for name, df in tables.items():
        nulls = null_key_count(df, TABLE_KEYS[name])
        lines.append(f"  {name:<24} {len(df):>9,} rows   period {month_span(df):<18} null keys: {nulls}")
        flags += nulls > 0

    win = f"{WINDOW_START[0]}-{WINDOW_START[1]:02d}..{WINDOW_END[0]}-{WINDOW_END[1]:02d}"
    lines.append(f"\nStandard window {win} complete?")
    for name in ["t100_routes", "delay_cause"]:
        missing = window_complete(tables[name])
        lines.append(f"  {name:<24} {'yes' if not missing else 'MISSING ' + str(missing)}")
        flags += bool(missing)

    xc = passenger_cross_check(tables["t100_routes"], tables["socrata_airport_months"], tables["faa_enplanements"])
    lines.append(f"\nPassenger cross-check CY{CROSS_CHECK_YEAR} (flag if any difference > {MAX_DIFF:.0%}):")
    lines.append(f"  {'airport':<8}{'T-100 pax':>13}{'Socrata pax':>13}{'FAA enpl.':>13}"
                 f"{'T100/FAA':>10}{'Soc/FAA':>10}{'T100/Soc':>10}  flag")
    for ap, r in xc.iterrows():
        lines.append(
            f"  {ap:<8}{r.t100_pax:>13,.0f}{r.socrata_pax:>13,.0f}{r.faa_enplanements:>13,.0f}"
            f"{r.t100_vs_faa:>+10.1%}{r.socrata_vs_faa:>+10.1%}{r.t100_vs_socrata:>+10.1%}  {'FLAG' if r.flag else 'ok'}"
        )
    flags += int(xc["flag"].sum())

    lf = load_factor_violations(tables["t100_routes"])
    ds = delay_cause_sum_violations(tables["delay_cause"])
    cv = completion_violations(tables["t100_routes"])
    lines += [
        "\nSanity checks:",
        f"  airport-months (>=1,000 seats) with pax > seats: {lf}",
        f"  delay-cause months, causes vs arr_delay > 5%:    {ds}",
        f"  airports with US performed > scheduled x 1.05:   {len(cv)} {cv[:10]}",
    ]
    flags += int(lf > 0) + int(ds > 0) + int(bool(cv))
    lines.append(f"\nFlags: {flags}")
    return lines, flags
