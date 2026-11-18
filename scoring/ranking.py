"""Percentile normalisation and composite indices: congestion, unmet demand, investment score.

Every index reports, per component: raw value -> percentile (0-100, vs the ranking universe)
-> weight -> contribution. Contributions add up to the score.
"""

import numpy as np
import pandas as pd

from data.repository import Frames
from scoring.common import envelope, pct, resolve_airport, resolve_region, rnd, window_label
from scoring.config import (
    CONGESTION_WEIGHTS, DEFAULT_PRESET, LOAD_FACTOR_SENSITIVITY, RANKING_UNIVERSE_HUBS,
    SOURCE_LABELS, TARGET_LOAD_FACTOR, UNMET_DEMAND_WEIGHTS, WEIGHT_PRESETS, WINDOW,
)
from scoring.metrics import (
    COMPLETION_CAVEAT, DELAY_CAVEAT, NAS_ASSUMPTION, OPS_ASSUMPTION, OPS_CAVEAT, PERIOD, RUNWAY_ASSUMPTION, hub_label, kpi_table, socrata_window,
)

UNIVERSE_TEXT = (f"Percentiles are computed against the ranking universe: FAA "
                 f"{'/'.join(RANKING_UNIVERSE_HUBS)} hubs (national, before any regional filter)")
ALL_SOURCES = [SOURCE_LABELS[s] for s in ("t100", "delay", "socrata", "faa", "ourairports")]


# ---------------------------------------------------------------- normalisation

def percentile_against(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Mid-rank percentile (0-100) of each value within the reference distribution.

    p(x) = (count(ref < x) + 0.5 * count(ref == x)) / n * 100. NaN stays NaN.
    """
    ref = np.sort(reference.dropna().to_numpy(dtype=float))
    if len(ref) == 0:
        return pd.Series(np.nan, index=values.index)
    v = values.to_numpy(dtype=float)
    lo = np.searchsorted(ref, v, side="left")
    hi = np.searchsorted(ref, v, side="right")
    out = (lo + 0.5 * (hi - lo)) / len(ref) * 100
    return pd.Series(np.where(np.isnan(v), np.nan, out), index=values.index)


def validate_weights(weights: dict | None, defaults: dict) -> tuple[dict, list[str]]:
    """Merge user overrides into defaults, check keys/signs, renormalise to sum 1."""
    if not weights:
        return dict(defaults), []
    unknown = set(weights) - set(defaults)
    if unknown:
        raise ValueError(f"unknown weight(s) {sorted(unknown)}; allowed: {sorted(defaults)}")
    merged = {**defaults, **{k: float(v) for k, v in weights.items()}}
    if any(v < 0 for v in merged.values()) or sum(merged.values()) <= 0:
        raise ValueError("weights must be >= 0 and not all zero")
    total = sum(merged.values())
    merged = {k: v / total for k, v in merged.items()}
    return merged, [f"User-overridden weights (renormalised to sum 1): "
                    + ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in merged.items())]


def preset_of(weights: dict) -> str | None:
    """Name of the preset these (normalised) weights equal, else None (custom weights)."""
    for name, preset in WEIGHT_PRESETS.items():
        if set(preset) == set(weights) and all(abs(weights[k] - preset[k]) < 1e-9 for k in preset):
            return name
    return None


def investment_weights(preset: str | None = None, weights: dict | None = None) -> tuple[dict, list[str]]:
    """Weights for the investment score: a named preset (default 'balanced'), optionally with explicit
    user-given weights on top (merged and renormalised)."""
    if preset is not None and preset not in WEIGHT_PRESETS:
        raise ValueError(f"unknown preset '{preset}'; options: {', '.join(WEIGHT_PRESETS)}")
    base = WEIGHT_PRESETS[preset or DEFAULT_PRESET]
    if weights:
        return validate_weights(weights, base)
    notes = ([f"Weight preset '{preset}': " + ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in base.items())]
             if preset else [])
    return dict(base), notes


def weighted_index(pcts: pd.DataFrame, weights: dict) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Score = sum(weight x percentile) over available components; weights of missing (NaN)
    components are redistributed. Returns (score, contributions, renormalised_flag)."""
    w = pd.Series(weights)[pcts.columns]
    available = pcts.notna()
    wsum = available.mul(w, axis=1).sum(axis=1)
    contrib = pcts.fillna(0).mul(w, axis=1).div(wsum.where(wsum > 0), axis=0)
    score = contrib.sum(axis=1).where(wsum > 0)
    return score, contrib, wsum < 1 - 1e-9


def component_table(raw: pd.DataFrame, pcts: pd.DataFrame, contrib: pd.DataFrame,
                    weights: dict, airport: str, raw_format: dict) -> list[dict]:
    rows = []
    for c in pcts.columns:
        rows.append({
            "component": c,
            "raw_value": raw_format.get(c, rnd)(raw.loc[airport, c]),
            "percentile": rnd(pcts.loc[airport, c], 1),
            "weight": weights[c],
            "contribution": rnd(contrib.loc[airport, c], 2),
        })
    return rows


def _universe(k: pd.DataFrame) -> pd.DataFrame:
    return k[k["in_universe"]]


RAW_FORMAT = {
    "delay_rate": pct, "nas_share": pct, "cancellation_rate": pct, "completion_gap": pct,
    "growth_yoy": pct, "growth": pct, "ops_per_runway": lambda x: rnd(x, 0),
    "scale": lambda x: rnd(x, 0), "passengers": lambda x: rnd(x, 0),
}
RAW_UNITS = ("raw values: rates/shares/growth in %, load factor as a fraction, "
             "ops_per_runway = annual operations per runway, scale = passengers")


# ---------------------------------------------------------------- congestion

def congestion_scores(frames: Frames) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Congestion index for every airport. Returns (raw, percentiles, contributions, score)."""
    key = ("congestion", WINDOW)
    if key not in frames.cache:
        k = kpi_table(frames)
        cols = list(CONGESTION_WEIGHTS)
        raw = k[cols]
        pcts = pd.DataFrame({c: percentile_against(raw[c], _universe(k)[c]) for c in cols})
        score, contrib, _ = weighted_index(pcts, CONGESTION_WEIGHTS)
        frames.cache[key] = (raw, pcts, contrib, score)
    return frames.cache[key]


def congestion_index(frames: Frames, names: list[str]) -> dict:
    """Congestion index (0-100) and its components for one or more airports (e.g. LAX vs SNA)."""
    k = kpi_table(frames)
    raw, pcts, contrib, score = congestion_scores(frames)
    uni_scores = score[k["in_universe"]].sort_values(ascending=False)
    assumptions, rows = [], []
    for name in names:
        iata, a = resolve_airport(name, frames.airports)
        if a:
            assumptions.append(a)
        rows.append({
            "airport": iata, "hub": hub_label(k.loc[iata, "hub"]),
            "congestion_index": rnd(score[iata], 1),
            "national_rank": int(uni_scores.index.get_loc(iata)) + 1 if iata in uni_scores.index else None,
            "ranked_airports": len(uni_scores),
            "components": component_table(raw, pcts, contrib, CONGESTION_WEIGHTS, iata, RAW_FORMAT),
        })
    sw = socrata_window(frames.socrata, WINDOW)
    return envelope(
        {"airports": rows, "weights": CONGESTION_WEIGHTS, "units": RAW_UNITS},
        [SOURCE_LABELS[s] for s in ("delay", "socrata", "ourairports", "faa")],
        f"Delay & traffic: {window_label(WINDOW)}; Operations (Socrata): {window_label(sw)}",
        assumptions + [
            "Congestion index = weighted percentiles of delay rate, NAS delay share, operations per runway, "
            "cancellation rate (weights shown)",
            UNIVERSE_TEXT, NAS_ASSUMPTION, OPS_ASSUMPTION, RUNWAY_ASSUMPTION,
        ],
        [DELAY_CAVEAT, OPS_CAVEAT,
         f"Uses Socrata's latest 12 months ({window_label(sw)}), one month earlier than the standard window",
         "The index is relative (percentiles), not an absolute capacity measure"],
    )


# ---------------------------------------------------------------- unmet demand

def capacity_gap(passengers: float, seats: float, departures: float, target_lf: float) -> dict:
    """Extra seats/departures needed so that passengers / seats <= target_lf (demand held fixed)."""
    needed = passengers / target_lf
    extra_seats = max(0.0, needed - seats)
    seats_per_dep = seats / departures if departures else float("nan")
    return {
        "target_load_factor": target_lf,
        "current_load_factor": rnd(passengers / seats if seats else float("nan")),
        "extra_seats": round(extra_seats),
        "extra_seats_pct": pct(extra_seats / seats) if seats else None,
        "extra_departures": round(extra_seats / seats_per_dep) if seats_per_dep == seats_per_dep else None,
        "avg_seats_per_departure": rnd(seats_per_dep, 1),
    }


def capacity_gap_interpretation(target: float, window: dict, peak: dict, peak_month: str) -> str:
    """Plain-language reading of a capacity-gap result (deterministic, so the LLM does not infer it)."""
    if window["extra_seats"] == 0 and peak["extra_seats"] == 0:
        return (f"0 at target {target:.2f}: both the 12-month average load factor ({window['current_load_factor']}) "
                f"and the peak-month load factor ({peak['current_load_factor']}) are at or below the target; no seat "
                f"shortage by this measure")
    if window["extra_seats"] == 0:
        return (f"0 for the 12-month window at target {target:.2f} means the average load factor "
                f"({window['current_load_factor']}) is below the target; the shortage appears only in the peak month "
                f"{peak_month} (load factor {peak['current_load_factor']}). Delay-based pressure is shown separately "
                f"in the components")
    return (f"The 12-month average load factor ({window['current_load_factor']}) is above target {target:.2f}: "
            f"extra seats are needed across the year, more so in the peak month {peak_month}")


def _gap_text(g: dict) -> str:
    if g["extra_seats"] == 0:
        return f"0 extra seats (load factor {g['current_load_factor']})"
    return (f"+{g['extra_seats']:,} seats (+{g['extra_seats_pct']}%, ~{g['extra_departures']:,} departures; "
            f"load factor {g['current_load_factor']})")


def capacity_gap_headline(capacity_gap: dict) -> str:
    """One unambiguous sentence: every target x both periods (12-month window and peak month)."""
    parts = []
    for target, entry in capacity_gap.items():
        lf = target.split("_")[1]
        parts.append(f"at target load factor {lf}: 12-month window {_gap_text(entry['window'])}; "
                     f"peak month {entry['peak_month']['month']} {_gap_text(entry['peak_month'])}")
    return "Capacity gap (extra seats needed to bring load factor down to the target) - " + " | ".join(parts)


def unmet_demand(frames: Frames, name: str) -> dict:
    """Unmet-demand proxy (0-100), drivers vs FAA hub peers, and a capacity-gap estimate."""
    iata, a = resolve_airport(name, frames.airports)
    k = kpi_table(frames)
    if iata not in k.index:
        raise ValueError(f"{iata} has no passenger service in T-100 for {window_label(WINDOW)}")
    r = k.loc[iata]
    cols = list(UNMET_DEMAND_WEIGHTS)
    pcts = pd.DataFrame({c: percentile_against(k[c], _universe(k)[c]) for c in cols})
    score, contrib, renorm = weighted_index(pcts, UNMET_DEMAND_WEIGHTS)

    peers = _universe(k)[_universe(k)["hub"] == r["hub"]]
    drivers = []
    for c in cols:
        med = peers[c].median()
        if r[c] == r[c] and r[c] > med:
            fmt = RAW_FORMAT.get(c, rnd)
            drivers.append({"component": c, "value": fmt(r[c]), "peer_median": fmt(med),
                            "percentile": rnd(pcts.loc[iata, c], 1)})

    gaps = {}
    for lf in (TARGET_LOAD_FACTOR, *LOAD_FACTOR_SENSITIVITY):
        window = capacity_gap(r["passengers"], r["seats"], r["departures"], lf)
        peak = capacity_gap(r["peak_passengers"], r["peak_seats"], r["peak_departures"], lf)
        gaps[f"target_{lf:.2f}"] = {
            "window": window,
            "peak_month": {"month": r["lf_peak_month"], **peak},
            "interpretation": capacity_gap_interpretation(lf, window, peak, r["lf_peak_month"]),
        }

    caveats = [
        "True unmet (spilled) demand is NOT observable in public data; this is a proxy built from "
        "load factor, growth, delays and cancellations",
        "Capacity gap assumes passenger demand stays the same when capacity is added, and ignores "
        "slot, gate, runway and airline-network constraints",
        DELAY_CAVEAT, COMPLETION_CAVEAT,
    ]
    if renorm[iata]:
        caveats.append("Some components are missing for this airport; their weight was redistributed")
    return envelope(
        {
            "airport": iata, "hub": hub_label(r["hub"]),
            "unmet_demand_score": rnd(score[iata], 1),
            "components": component_table(k[cols], pcts, contrib, UNMET_DEMAND_WEIGHTS, iata, RAW_FORMAT),
            "drivers_above_peer_median": drivers,
            "peer_group": f"{hub_label(r['hub'])}s in the ranking universe (n={len(peers)})",
            "capacity_gap": gaps,
            "capacity_gap_headline": capacity_gap_headline(gaps),
            "units": RAW_UNITS,
        },
        ALL_SOURCES[:2] + [SOURCE_LABELS["faa"]], PERIOD,
        ([a] if a else []) + [
            f"Capacity gap target load factor = {TARGET_LOAD_FACTOR:.2f} (industry 'effectively full' level); "
            f"sensitivity at {', '.join(f'{x:.2f}' for x in LOAD_FACTOR_SENSITIVITY)} (large-hub median load factor)",
            "Drivers = components where the airport is above the median of its FAA hub-size peers",
            UNIVERSE_TEXT,
        ],
        caveats,
    )


# ---------------------------------------------------------------- investment score

def investment_table(frames: Frames, weights: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Investment score for every airport: (raw, percentiles, contributions, score)."""
    k = kpi_table(frames)
    _, _, _, cong = congestion_scores(frames)
    raw = pd.DataFrame({
        "demand_pressure": k["lf_avg"], "growth": k["growth_yoy"],
        "congestion": cong, "scale": k["passengers"],
    })
    uni = raw[k["in_universe"]]
    pcts = pd.DataFrame({c: percentile_against(raw[c], uni[c]) for c in raw.columns})
    score, contrib, _ = weighted_index(pcts, weights)
    return raw, pcts, contrib, score


def _select(k: pd.DataFrame, region: str | None, states, airports) -> tuple[pd.Index, list[str]]:
    assumptions = []
    if region:
        states, a = resolve_region(region)
        assumptions.append(a)
    if states:
        return k.index[k["state"].isin(states)], assumptions
    if airports:
        return pd.Index(airports), assumptions
    return k.index, assumptions


def investment_scores(frames: Frames, region: str | None = None, states: list[str] | None = None,
                      airports: list[str] | None = None, weights: dict | None = None,
                      top_n: int | None = None, preset: str | None = None) -> dict:
    """Investment score 0-100 for a region / state list / airport list (default: all ranked airports).
    Weights: a named preset (default 'balanced') and/or explicit user weights (see investment_weights)."""
    w, w_assumptions = investment_weights(preset, weights)
    k = kpi_table(frames)
    raw, pcts, contrib, score = investment_table(frames, w)
    selected, sel_assumptions = _select(k, region, states, airports)
    national = score[k["in_universe"]].sort_values(ascending=False)

    ranked = [a for a in score.loc[selected].sort_values(ascending=False).index if k.loc[a, "in_universe"]]
    rows = []
    for i, a in enumerate(ranked[:top_n] if top_n else ranked, start=1):
        rows.append({
            "rank": i, "airport": a, "state": k.loc[a, "state"], "hub": hub_label(k.loc[a, "hub"]),
            "investment_score": rnd(score[a], 1),
            "national_rank": int(national.index.get_loc(a)) + 1,
            "components": component_table(raw, pcts, contrib, w, a, RAW_FORMAT | {"demand_pressure": rnd,
                                                                                  "congestion": lambda x: rnd(x, 1)}),
        })
    outside = [
        {"airport": a, "state": k.loc[a, "state"], "hub": hub_label(k.loc[a, "hub"]),
         "passengers": int(k.loc[a, "passengers"])}
        for a in selected if a in k.index and not k.loc[a, "in_universe"] and k.loc[a, "passengers"] >= 50_000
    ]
    return envelope(
        {"ranking": rows, "ranked_nationally": len(national), "weights": w, "preset": preset_of(w),
         "outside_ranking_universe": sorted(outside, key=lambda x: -x["passengers"]),
         "units": "demand_pressure = load factor; growth in %; congestion = congestion index 0-100; scale = passengers"},
        ALL_SOURCES, PERIOD,
        sel_assumptions + w_assumptions + [
            "Investment score = weighted percentiles of demand pressure (load factor), passenger growth, "
            "congestion index and scale (passengers)",
            UNIVERSE_TEXT,
            "Higher score = stronger demand-vs-capacity case for terminal/capacity investment",
        ],
        ["Score is relative (percentiles), not a financial return estimate",
         "Does not include costs, airport finances, land, regulation or airline strategy",
         DELAY_CAVEAT],
    )


def rank_stability(frames: Frames, region: str | None = None, states: list[str] | None = None,
                   top_n: int = 3, airports: list[str] | None = None) -> dict:
    """Re-rank under each weight preset and report whether the top-N picks change."""
    k = kpi_table(frames)
    selected, assumptions = _select(k, region, states, airports)
    selected = [a for a in selected if a in k.index and k.loc[a, "in_universe"]]
    rankings = {}
    for name, w in WEIGHT_PRESETS.items():
        _, _, _, score = investment_table(frames, w)
        rankings[name] = list(score.loc[selected].sort_values(ascending=False).index[:top_n])
    sets = {frozenset(v) for v in rankings.values()}
    orders = {tuple(v) for v in rankings.values()}
    return envelope(
        {"top_n": top_n, "rankings": rankings, "presets": WEIGHT_PRESETS,
         "same_top_n_set": len(sets) == 1, "same_top_n_order": len(orders) == 1},
        ALL_SOURCES, PERIOD,
        assumptions + ["Weight presets: " + ", ".join(
            f"{n} (default)" if n == DEFAULT_PRESET else n for n in WEIGHT_PRESETS)],
        ["Stability across presets increases confidence but does not validate the weights themselves"],
    )
