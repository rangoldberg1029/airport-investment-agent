"""Tool definitions (JSON schema) and dispatch to the deterministic scoring layer.

Each tool returns a ToolCallRecord with:
  envelope  - the full standard envelope {data, sources, data_period, assumptions, caveats} (for the UI)
  llm_view  - a compact JSON string sent to the LLM (short source ids, rounded numbers, trimmed lists,
              precomputed differences so the LLM never has to calculate)
No numbers are computed here except differences between values that the scoring layer returned.
"""

import json
from dataclasses import dataclass, field
from typing import Callable

from data.repository import Frames
from data.sources import nas_status
from scoring import metrics, ranking
from scoring.common import envelope, month_label, resolve_airport, window_label
from scoring.config import (
    CONGESTION_WEIGHTS, DEFAULT_LONG_HAUL_MI, DEFAULT_PRESET, INVESTMENT_WEIGHTS, LOAD_FACTOR_SENSITIVITY,
    LONG_HAUL_THRESHOLDS_MI, MIN_COMMERCIAL_RUNWAY_FT, PRIOR_WINDOW, RANKING_UNIVERSE_HUBS,
    REGIONS, SOURCE_LABELS, TARGET_LOAD_FACTOR, UNMET_DEMAND_WEIGHTS, WEIGHT_PRESETS, WINDOW,
)

SOURCE_IDS = {
    SOURCE_LABELS["t100"]: "T100", SOURCE_LABELS["delay"]: "DELAY", SOURCE_LABELS["socrata"]: "SOC",
    SOURCE_LABELS["faa"]: "FAA", SOURCE_LABELS["ourairports"]: "OA", "FAA NAS Status (live)": "NAS",
}


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    envelope: dict
    llm_view: str
    error: str | None = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers

def _round(x, digits: int = 1):
    return None if x is None else round(x, digits)


def diff(a, b, digits: int = 2):
    """a - b, or None if either is missing. The only arithmetic in this module."""
    if a is None or b is None:
        return None
    return round(a - b, digits)


def airport_name(frames: Frames, iata: str) -> str:
    """Official airport name from the airports table (OurAirports); the code itself if unknown."""
    match = frames.airports.loc[frames.airports["iata"] == iata, "name"]
    return str(match.iloc[0]) if len(match) else iata


def labelled(frames: Frames, iata: str) -> str:
    """'BOS Boston Logan International Airport': code + official name, as the LLM must quote it."""
    name = airport_name(frames, iata)
    return iata if name == iata else f"{iata} {name}"


def weights_label(weights: dict) -> str:
    """'preset growth-led (demand pressure 25%, growth 45%, congestion 15%, scale 15%)',
    'preset balanced (default) (...)' or 'custom weights (...)'."""
    name = ranking.preset_of(weights)
    head = "custom weights" if name is None else f"preset {name}" + (" (default)" if name == DEFAULT_PRESET else "")
    return f"{head} (" + ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in weights.items()) + ")"


def merge(parts: list[dict], data: dict, period: str | None = None) -> dict:
    """Combine several envelopes into one (union of sources/assumptions/caveats)."""
    return envelope(
        data,
        [s for p in parts for s in p["sources"]],
        period or parts[0]["data_period"],
        [a for p in parts for a in p["assumptions"]],
        [c for p in parts for c in p["caveats"]],
    )


# Short LLM-facing forms of the standard assumption/caveat texts (prefix match). None = omit for the LLM
# (still shown in the UI). Name resolutions, region definitions and unknown texts are kept verbatim.
SHORT_TEXT = [
    ("Delay data covers", "DELAY covers reporting carriers, domestic arrivals only"),
    ("Operations per runway is a rough proxy", "ops/runway is a rough proxy"),
    ("Uses Socrata's latest 12 months", "ops/runway period ends 1 month earlier (SOC)"),
    ("The index is relative", "index is relative (percentiles), not absolute capacity"),
    ("T-100 counts passengers per flight segment", "T100 counts passengers per flight segment"),
    ("Percentiles are computed against", f"percentiles vs FAA {'/'.join(RANKING_UNIVERSE_HUBS)} hubs nationally"),
    ("NAS share = ", "NAS share = NAS minutes / sum of 5 delay-cause minutes"),
    ("Operations = 2 x departures", "operations = 2 x all-carrier departures (incl. cargo)"),
    ("Runways counted = ", f"only runways >= {MIN_COMMERCIAL_RUNWAY_FT:,} ft counted"),
    ("Delay rate = ", "delay rate = arrivals >= 15 min late / arrivals"),
    ("Load factor = ", "load factor = passengers / seats"),
    ("Growth = ", "growth = window vs prior 12 months (T100)"),
    ("Congestion index = ", "congestion index = weighted percentiles of delay rate, NAS share, ops/runway, cancellations"),
    ("Investment score = ", "investment score = weighted percentiles of load factor, growth, congestion, scale"),
    ("Higher score = ", None),
    ("Score is relative", "score is relative, not a financial return estimate"),
    ("Does not include costs", "excludes costs, finances, regulation, airline strategy"),
    ("Completion gap covers US carriers only", "completion gap covers US carriers only"),
    ("Completion gap = ", None),
    ("Stability across presets", None),
    ("Weight presets", None),
    ("True unmet (spilled) demand is NOT observable", "true spilled demand is not observable; this is a proxy"),
    ("Capacity gap assumes", "capacity gap assumes fixed demand; ignores slot/gate/runway limits"),
    ("Capacity gap target load factor",
     f"capacity gap target LF {TARGET_LOAD_FACTOR:.2f}; sensitivity {', '.join(f'{x:.2f}' for x in LOAD_FACTOR_SENSITIVITY)}"),
    ("Drivers = ", "drivers = components above FAA hub-size peer median"),
    ("Passenger service only", "passenger service only, freighters excluded"),
    ("Distance is the flight segment", "distance = flight segment, not full journey"),
    ("All-cargo flights are excluded", "all-cargo flights excluded"),
]


def shorten(texts: list[str]) -> list[str]:
    out = []
    for t in texts:
        short = next((s for prefix, s in SHORT_TEXT if t.startswith(prefix)), t)
        if short and short not in out:
            out.append(short)
    return out


def compact(env: dict, view_data: dict) -> str:
    """LLM-facing JSON: short source ids, period, short assumptions/caveats, compact data."""
    view = {
        "src": [SOURCE_IDS.get(s, s) for s in env["sources"]],
        "period": env["data_period"],
        "data": view_data,
        "assumptions": shorten(env["assumptions"]),
        "caveats": shorten(env["caveats"]),
    }
    return json.dumps(view, separators=(",", ":"), ensure_ascii=False)


# Units attached to every value the LLM sees, so it cannot drop or confuse them.
UNITS = {
    "lf_avg": "LF", "lf_peak": "LF", "demand_pressure": "LF", "load_factor_avg": "LF", "load_factor_peak": "LF",
    "growth_yoy": "%", "growth": "%", "delay_rate": "%", "nas_share": "%", "cancellation_rate": "%",
    "completion_gap": "%", "delay_rate_pct": "%", "nas_share_pct": "%", "cancellation_rate_pct": "%",
    "ops_per_runway": "ops/runway/yr", "operations_per_runway": "ops/runway/yr",
    "scale": "passengers", "passengers": "passengers", "congestion": "index", "congestion_index": "index",
    "commercial_runways": "runways",
}
DIFF_UNITS = {"%": "pp", "index": "index pts"}
SIGNED_UNITS = ("pp", "index pts", "pts")


def u(value, unit: str, signed: bool = False):
    """Value with its unit as a string: 18.7 -> '18.7%', 0.817 -> '0.817 LF', 133594 -> '133,594 ops/runway/yr'.
    Differences ('pp', 'pts', or signed=True) always carry an explicit sign."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    sign = "+" if (signed or unit in SIGNED_UNITS) and value > 0 else ""
    if unit == "%":
        return f"{sign}{value}%"
    if isinstance(value, (int, float)) and abs(value) >= 1000:
        return f"{sign}{value:,.0f} {unit}"
    return f"{sign}{value} {unit}"


def uc(component: str, value):
    """Component value with its unit."""
    return u(value, UNITS.get(component, ""))


def ud(component: str, value):
    """Difference of a component with its difference unit (pp for %, index pts for indices)."""
    unit = UNITS.get(component, "")
    return u(value, DIFF_UNITS.get(unit, unit), signed=True)


def comp_rows(components: list[dict]) -> list[list]:
    """[component, value with unit, percentile (1 decimal), weight, contribution in pts]."""
    return [[c["component"], uc(c["component"], c["raw_value"]),
             None if c["percentile"] is None else round(c["percentile"], 1),
             c["weight"], f"{c['contribution']} pts"] for c in components]


COMP_HEADER = ["component", "value", "pctile", "weight", "contribution"]

UNMET_MEANINGS = {
    "lf_avg": "12-month average load factor (passengers / seats); higher = fuller flights",
    "lf_peak": "highest monthly load factor in the window",
    "growth_yoy": "passenger growth vs the prior 12 months",
    "delay_rate": "share of arrivals >= 15 min late",
    "nas_share": "share of delay minutes caused by the airport/airspace system (capacity, ATC)",
    "completion_gap": "share of US-carrier scheduled departures that were not flown (cancellations); "
                      "it is independent of load factor",
}
INDEPENDENCE_NOTE = ("Each component is scored independently against other airports; components do not cause or "
                     "explain each other. The score is the sum of the contributions.")


def _ft(label: str) -> int:
    """'7L/25R (12,894 ft)' -> 12894 (parsing a label produced by scoring.metrics)."""
    return int(label.rsplit("(", 1)[1].split(" ft")[0].replace(",", ""))


def contribution_differences(contrib_base: dict, contrib_other: dict) -> list[tuple[str, float]]:
    """(component, base - other) in index points, largest absolute difference first."""
    diffs = {k: round(contrib_base[k] - contrib_other[k], 2) for k in contrib_base}
    return sorted(diffs.items(), key=lambda kv: -abs(kv[1]))


def difference_drivers(base: str, other: str, contrib_base: dict, contrib_other: dict,
                       index_base: float, index_other: float) -> dict:
    """Rank the component contribution differences (base - other) by size and say whose index each raises."""
    ranked = contribution_differences(contrib_base, contrib_other)
    net = round(sum(v for _, v in ranked), 2)
    higher = base if net > 0 else other
    parts = ", ".join(f"{k} {v:+.2f} pts" for k, v in ranked)
    return {
        "ranked": [f"{k} {v:+.2f} pts (raises {base if v > 0 else other if v < 0 else 'neither'})" for k, v in ranked],
        "net_difference": f"{net:+.2f} pts",
        "summary": (f"{higher} has the higher congestion index ({base} {index_base} vs {other} {index_other}). "
                    f"Contribution differences ({base} minus {other}), largest first: {parts}; net {net:+.2f} pts. "
                    f"Positive values raise {base}'s index, negative values raise {other}'s. "
                    f"Largest single driver: {ranked[0][0]} ({ranked[0][1]:+.2f} pts)."),
    }


# ---------------------------------------------------------------- tools

def get_airport_profile(frames: Frames, airport: str) -> tuple[dict, dict]:
    prof = metrics.airport_profile(frames, airport)
    iata = prof["data"]["airport"]
    dly = metrics.delay_profile(frames, iata)
    gap = metrics.completion_gap(frames, iata)
    ops = metrics.ops_per_runway(frames, iata)
    inv = ranking.investment_scores(frames, airports=[iata])
    cong = ranking.congestion_index(frames, [iata])
    p, d, o = prof["data"], dly["data"], ops["data"]
    inv_row = inv["data"]["ranking"][0] if inv["data"]["ranking"] else None
    n = inv["data"]["ranked_nationally"]
    pct3 = ("delay_rate_pct", "nas_share_pct", "cancellation_rate_pct")
    data = {
        "airport": iata, "name": airport_name(frames, iata), "hub": p["hub"], "state": p["state"],
        "passengers": u(p["passengers"], "passengers"), "seats": u(p["seats"], "seats"),
        "departures": u(p["departures"], "departures"),
        "load_factor_avg": u(p["load_factor_avg"], "LF"), "load_factor_peak": u(p["load_factor_peak_month"], "LF"),
        "peak_month": p["peak_month"], "growth_yoy": u(p["growth_yoy_pct"], "%"),
        **{k.replace("_pct", ""): u(d[k], "%") for k in pct3},
        "national": {k.replace("_pct", ""): u(d["national"][k], "%") for k in pct3},
        "hub_peer_median": {k.replace("_pct", ""): u(d["hub_peer_median"][k], "%") for k in pct3},
        "delay_rate_minus_national": u(diff(d["delay_rate_pct"], d["national"]["delay_rate_pct"]), "pp"),
        "nas_share_minus_national": u(diff(d["nas_share_pct"], d["national"]["nas_share_pct"]), "pp"),
        "delay_rate_minus_peer_median": u(diff(d["delay_rate_pct"], d["hub_peer_median"]["delay_rate_pct"]), "pp"),
        "completion_gap_us_carriers": u(gap["data"]["completion_gap_pct"], "%"),
        "operations_per_runway": u(o["operations_per_runway"], "ops/runway/yr"),
        "commercial_runways": u(int(o["runway_count"]), "runways"),
        "runway_rule_interpretation": o["interpretation"],
        "congestion_index": u(cong["data"]["airports"][0]["congestion_index"], "index"),
        "congestion_national_rank": f"{cong['data']['airports'][0]['national_rank']} of {n}",
        "investment_score": u(inv_row["investment_score"], "/ 100") if inv_row else None,
        "investment_national_rank": f"{inv_row['national_rank']} of {n}" if inv_row else None,
        "in_ranking_universe": inv_row is not None,
    }
    full = merge([prof, dly, gap, ops, inv, cong],
                 {"profile": p, "delay": d, "completion_gap": gap["data"], "ops": o,
                  "investment": inv["data"], "congestion": cong["data"]},
                 f"{prof['data_period']}; operations: {o['period']}")
    return full, data


def rank_airports(frames: Frames, region: str | None = None, states: list[str] | None = None,
                  airports: list[str] | None = None, weights: dict | None = None, top_n: int = 5,
                  preset: str | None = None) -> tuple[dict, dict]:
    notes = []
    if airports:
        resolved = []
        for a in airports:
            iata, note = resolve_airport(a, frames.airports)
            resolved.append(iata)
            if note:
                notes.append(note)
        airports = resolved
    inv = ranking.investment_scores(frames, region=region, states=states, airports=airports, weights=weights,
                                    preset=preset)
    stab = ranking.rank_stability(frames, region=region, states=states, airports=airports, top_n=3)
    rows = inv["data"]["ranking"]
    table = []
    for i, r in enumerate(rows[:top_n]):
        raw = {c["component"]: c["raw_value"] for c in r["components"]}
        nxt = rows[i + 1]["investment_score"] if i + 1 < len(rows) else None
        table.append([r["rank"], r["airport"], airport_name(frames, r["airport"]), r["state"], r["hub"],
                      r["investment_score"], r["national_rank"],
                      uc("demand_pressure", raw["demand_pressure"]), uc("growth", raw["growth"]),
                      uc("congestion", raw["congestion"]), uc("scale", raw["scale"]),
                      [f"{c['contribution']} pts" for c in r["components"]],
                      u(diff(r["investment_score"], nxt, 1), "pts")])
    headline = (f"Ranking with {weights_label(inv['data']['weights'])}: "
                + "; ".join(f"{r['rank']}. {labelled(frames, r['airport'])} {r['investment_score']}"
                            for r in rows[:top_n])) if rows else "No airport in this selection is ranked."
    data = {
        "headline": headline,
        "header": ["rank", "airport", "name", "state", "hub", "score", "natl_rank", "load_factor", "growth",
                   "congestion", "passengers", "contribution[demand,growth,congestion,scale]", "score_gap_to_next"],
        "rows": table,
        "n_more": max(0, len(rows) - top_n),
        "ranked_nationally": inv["data"]["ranked_nationally"],
        "preset": inv["data"]["preset"] or "custom",
        "weights": inv["data"]["weights"],
        "outside_ranking_universe": [labelled(frames, o["airport"]) for o in inv["data"]["outside_ranking_universe"]],
        "stability_top3": stab["data"]["rankings"],
        "stability_same_set": stab["data"]["same_top_n_set"],
        "interpretation": "score = sum of the four contributions; score_gap_to_next = score minus the next airport's score",
    }
    full = merge([inv, stab], {"ranking": inv["data"], "stability": stab["data"]})
    full["assumptions"] = notes + full["assumptions"]
    return full, data


COMPARE_METRICS = [  # (label, unit, getter)
    ("congestion_index", "index", lambda c, p, d, o: c["congestion_index"]),
    ("delay_rate", "%", lambda c, p, d, o: d["delay_rate_pct"]),
    ("nas_share", "%", lambda c, p, d, o: d["nas_share_pct"]),
    ("cancellation_rate", "%", lambda c, p, d, o: d["cancellation_rate_pct"]),
    ("operations_per_runway", "ops/runway/yr", lambda c, p, d, o: o["operations_per_runway"]),
    ("commercial_runways", "runways", lambda c, p, d, o: int(o["runway_count"])),
    ("load_factor_avg", "LF", lambda c, p, d, o: p["load_factor_avg"]),
    ("passengers", "passengers", lambda c, p, d, o: p["passengers"]),
]


def compare_congestion(frames: Frames, airports: list[str]) -> tuple[dict, dict]:
    if not 2 <= len(airports) <= 5:
        raise ValueError("compare_congestion needs 2 to 5 airports")
    cong = ranking.congestion_index(frames, airports)
    rows = cong["data"]["airports"]
    codes = [r["airport"] for r in rows]
    profs = [metrics.airport_profile(frames, a) for a in codes]
    dlys = [metrics.delay_profile(frames, a) for a in codes]
    opss = [metrics.ops_per_runway(frames, a) for a in codes]
    raw = {label: {a: get(rows[i], profs[i]["data"], dlys[i]["data"], opss[i]["data"]) for i, a in enumerate(codes)}
           for label, _, get in COMPARE_METRICS}
    units = {label: unit for label, unit, _ in COMPARE_METRICS}
    base = codes[0]
    contrib = {r["airport"]: {c["component"]: c["contribution"] for c in r["components"]} for r in rows}
    index = {r["airport"]: r["congestion_index"] for r in rows}
    data = {
        "airports": codes,
        "names": {a: airport_name(frames, a) for a in codes},
        "hubs": {r["airport"]: r["hub"] for r in rows},
        "congestion_national_rank": {r["airport"]: f"{r['national_rank']} of {r['ranked_airports']}" for r in rows},
        "values": {label: {a: u(raw[label][a], units[label]) for a in codes} for label in raw},
        "national": {k.replace("_pct", ""): u(v, "%") for k, v in dlys[0]["data"]["national"].items() if k != "arrivals"},
        "differences": {
            f"{base}_minus_{o}": {label: u(diff(raw[label][base], raw[label][o], 3),
                                           DIFF_UNITS.get(units[label], units[label]), signed=True) for label in raw}
            for o in codes[1:]
        },
        "index_components_header": ["component", "pctile", "weight", "contribution"],
        "index_components": {r["airport"]: [[c[0], c[2], c[3], c[4]] for c in comp_rows(r["components"])]
                             for r in rows},
        "drivers_of_index_difference": {
            f"{base}_vs_{o}": difference_drivers(base, o, contrib[base], contrib[o], index[base], index[o])
            for o in codes[1:]
        },
        "runway_lengths_ft_counted": {o["data"]["airport"]: [_ft(x) for x in o["data"]["runways_counted"]] for o in opss},
        "runway_lengths_ft_excluded_short": {o["data"]["airport"]: [_ft(x) for x in o["data"]["runways_excluded_short"]]
                                             for o in opss},
        "interpretation": [
            f"{base}_minus_X > 0 means {base} is higher than X on that metric; < 0 means lower",
            "To explain why one index is higher, use drivers_of_index_difference (contributions), not raw values",
            *[o["data"]["interpretation"] for o in opss],
        ],
    }
    full = merge([cong, *profs, *dlys, *opss], {"congestion": cong["data"], "raw_values": raw},
                 f"Delay & traffic: {window_label(WINDOW)}; Operations (Socrata): {opss[0]['data']['period']}")
    return full, data


def long_haul_share(frames: Frames, airport: str, threshold_mi: int | None = None) -> tuple[dict, dict]:
    default = int(threshold_mi) if threshold_mi else DEFAULT_LONG_HAUL_MI
    env = metrics.long_haul_share(frames, airport, default=default)
    d = env["data"]
    key = f"top_routes_at_or_above_{default}_mi"
    data = {
        "airport": d["airport"], "name": airport_name(frames, d["airport"]), "threshold": f"{default:,} mi",
        "departure_share": u(d["long_haul_departure_share_pct"], "%"),
        "seat_share": u(d["long_haul_seat_share_pct"], "%"),
        "total_passenger_departures": u(d["total_passenger_departures"], "departures"),
        "sensitivity": [[f"{s['threshold_mi']:,} mi", u(s["departure_share_pct"], "%"), u(s["seat_share_pct"], "%")]
                        for s in d["sensitivity"]],
        "sensitivity_header": ["threshold", "departure_share", "seat_share"],
        "top_routes": [[r["dest"], airport_name(frames, r["dest"]), r["dest_country"], f"{int(r['distance_mi']):,} mi",
                        u(int(r["departures"]), "departures"), u(int(r["seats"]), "seats")] for r in d[key]],
        "top_routes_header": ["dest", "name", "country", "distance", "departures", "seats"],
        "interpretation": "A lower threshold counts more routes as long-haul, so the share rises as the threshold falls",
    }
    return env, data


def unmet_demand(frames: Frames, airport: str) -> tuple[dict, dict]:
    um = ranking.unmet_demand(frames, airport)
    d = um["data"]
    dly = metrics.delay_profile(frames, d["airport"])
    dd = dly["data"]

    # The headline already states every capacity-gap number (both targets x both periods); the LLM view adds only
    # the interpretation per target. The full table is in the envelope (Evidence tab).
    gap_meanings = {target: entry["interpretation"] for target, entry in d["capacity_gap"].items()}
    data = {
        "headline": d["capacity_gap_headline"],
        "airport": d["airport"], "name": airport_name(frames, d["airport"]), "hub": d["hub"],
        "unmet_demand_score": u(d["unmet_demand_score"], "/ 100"),
        "components_header": COMP_HEADER, "components": comp_rows(d["components"]),
        "component_meanings": UNMET_MEANINGS,
        "note": INDEPENDENCE_NOTE,
        "drivers_above_peer_median": [
            [x["component"], uc(x["component"], x["value"]), uc(x["component"], x["peer_median"]),
             ud(x["component"], diff(x["value"], x["peer_median"], 3)), round(x["percentile"], 1)]
            for x in d["drivers_above_peer_median"]],
        "drivers_header": ["component", "value", "peer_median", "value_minus_median", "pctile"],
        "peer_group": d["peer_group"],
        "vs_national": {
            "delay_rate": {"airport": u(dd["delay_rate_pct"], "%"), "national": u(dd["national"]["delay_rate_pct"], "%"),
                           "difference": u(diff(dd["delay_rate_pct"], dd["national"]["delay_rate_pct"]), "pp")},
            "nas_share": {"airport": u(dd["nas_share_pct"], "%"), "national": u(dd["national"]["nas_share_pct"], "%"),
                          "difference": u(diff(dd["nas_share_pct"], dd["national"]["nas_share_pct"]), "pp")},
        },
        "capacity_gap_interpretation": gap_meanings,
    }
    return merge([um, dly], {"unmet_demand": d, "delay_profile": dd}), data


def live_airport_status(frames: Frames, airport: str,
                        fetch: Callable[[], tuple[list[dict], str]] = nas_status.fetch_feed) -> tuple[dict, dict]:
    iata, note = resolve_airport(airport, frames.airports)
    caveats = ["Live FAA NAS Status: context only, not used in any score",
               "Feed is public but not a formally documented API; format may change"]
    try:
        events, stamp = fetch()
    except Exception as exc:
        data = {"airport": iata, "name": airport_name(frames, iata), "status": "live FAA status unavailable right now",
                "error": type(exc).__name__}
        return envelope(data, ["FAA NAS Status (live)"], "live", [note] if note else [], caveats), data
    found = nas_status.parse_airport_events(events, iata)
    data = {
        "airport": iata, "name": airport_name(frames, iata), "feed_timestamp": stamp,
        "status": "active FAA events" if found else "no active FAA delay events reported right now",
        "events": found,
        "airports_with_events_nationally": len(events),
    }
    env = envelope(data, ["FAA NAS Status (live)"], f"live, feed updated {stamp}", [note] if note else [], caveats)
    return env, data


def _source_inventory(frames: Frames) -> list[dict]:
    from data.db import month_span
    return [
        {"id": "T100", "name": SOURCE_LABELS["t100"], "access": "manual download (TranStats web form)",
         "period": month_span(frames.t100),
         "feeds": ["passengers", "seats", "departures", "load factor (avg, peak)", "growth", "completion gap",
                   "long-haul share"]},
        {"id": "DELAY", "name": SOURCE_LABELS["delay"],
         "access": "public file download (TranStats delay-cause page, scripted)", "period": month_span(frames.delay),
         "feeds": ["delay rate", "NAS delay share", "cancellation rate"]},
        {"id": "SOC", "name": SOURCE_LABELS["socrata"], "access": "public API (Socrata SODA)",
         "period": month_span(frames.socrata),
         "feeds": ["operations per runway (all-carrier departures)", "passenger cross-check"]},
        {"id": "FAA", "name": SOURCE_LABELS["faa"], "access": "public file download (FAA .xlsx)",
         "period": month_span(frames.faa), "feeds": ["hub size (ranking universe, peer groups)", "CY growth cross-check"]},
        {"id": "OA", "name": SOURCE_LABELS["ourairports"], "access": "public file download (CSV)",
         "period": "current snapshot", "feeds": ["airport names/locations", "commercial runway count"]},
        {"id": "NAS", "name": "FAA NAS Status (live)", "access": "public API (live, undocumented)",
         "period": "real time", "feeds": ["live delays/ground stops (context only, not scored)"]},
    ]


def explain_methodology(frames: Frames, topic: str = "all") -> tuple[dict, dict]:
    topics = {
        "investment_score": {
            "formula": "sum(weight x percentile) over demand_pressure (load factor), growth (YoY passengers), "
                       "congestion (congestion index), scale (passengers); 0-100",
            "weights": INVESTMENT_WEIGHTS, "default_preset": DEFAULT_PRESET, "presets": WEIGHT_PRESETS,
            "presets_use": "named presets give deterministic re-rankings and are used to test rank stability",
            "ranking_universe": f"FAA hubs {'/'.join(RANKING_UNIVERSE_HUBS)}; percentiles national",
            "override": "explicit user-given weights override the preset; renormalised to sum 1",
        },
        "congestion": {
            "formula": "sum(weight x percentile) of delay rate, NAS delay share, operations per runway, "
                       "cancellation rate; 0-100",
            "weights": CONGESTION_WEIGHTS,
            "runways_counted": f"open runways >= {MIN_COMMERCIAL_RUNWAY_FT:,} ft",
            "nas_share": "NAS minutes / sum of the five delay-cause minutes",
        },
        "unmet_demand": {
            "formula": "sum(weight x percentile); drivers = components above the FAA hub-size peer median",
            "weights": UNMET_DEMAND_WEIGHTS,
            "capacity_gap": f"extra seats = passengers / target LF - seats; target {TARGET_LOAD_FACTOR}, "
                            f"sensitivity {LOAD_FACTOR_SENSITIVITY}",
            "limitation": "true spilled demand is not observable in public data",
        },
        "long_haul": {
            "definition": "share of passenger departures (and seats) with segment distance >= threshold",
            "default_threshold_mi": DEFAULT_LONG_HAUL_MI, "sensitivity_mi": LONG_HAUL_THRESHOLDS_MI,
            "filter": "passenger service only (CLASS F/L, freighters excluded)",
        },
        "load_factor": {
            "definition": "passengers / seats on departing passenger segments",
            "window": window_label(WINDOW), "prior_window": window_label(PRIOR_WINDOW),
            "growth": "passengers in window vs prior window",
        },
        "data_sources": _source_inventory(frames),
        "regions": {k: list(v) for k, v in REGIONS.items()},
    }
    if topic not in topics and topic != "all":
        raise ValueError(f"unknown topic '{topic}'. Options: {', '.join(topics)}, all")
    data = topics if topic == "all" else {topic: topics[topic]}
    env = envelope(data, [], f"standard window {window_label(WINDOW)}",
                   ["Values read from scoring/config.py (deterministic)"], [])
    return env, data


# ---------------------------------------------------------------- schemas + dispatch

def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


AIRPORT = {"type": "string", "description": "IATA code or name, e.g. SFO, 'Boston', 'LA'"}

TOOL_SCHEMAS = [
    _fn("get_airport_profile",
        "All KPIs for ONE US airport: traffic, load factor, growth, delays, NAS share, runways, congestion "
        "index and investment score with national rank.", {"airport": AIRPORT}, ["airport"]),
    _fn("rank_airports",
        "Rank US airports by investment score (terminal/capacity expansion case) for a region, states or "
        "airport list, with rank stability across weight presets.",
        {"region": {"type": "string", "enum": sorted(REGIONS)},
         "states": {"type": "array", "items": {"type": "string"}, "description": "2-letter state codes"},
         "airports": {"type": "array", "items": {"type": "string"}},
         "preset": {"type": "string", "enum": list(WEIGHT_PRESETS),
                    "description": "Use for re-ranking requests, e.g. 'weight growth higher' -> growth-led, "
                                   "'default weights' -> balanced"},
         "weights": {"type": "object", "description": "ONLY if the user gives explicit numbers; partial override",
                     "properties": {k: {"type": "number"} for k in INVESTMENT_WEIGHTS}},
         "top_n": {"type": "integer", "default": 5}}, []),
    _fn("compare_congestion",
        "Compare congestion of 2-5 airports: congestion index, delays, NAS share, operations per runway, "
        "with precomputed differences.",
        {"airports": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5}}, ["airports"]),
    _fn("long_haul_share", "Share of long-haul passenger departures/seats from an airport, with threshold sensitivity.",
        {"airport": AIRPORT, "threshold_mi": {"type": "integer", "description": "default 3000"}}, ["airport"]),
    _fn("unmet_demand",
        "Unmet-demand proxy for an airport: score, drivers vs peers, capacity-gap estimate.",
        {"airport": AIRPORT}, ["airport"]),
    _fn("live_airport_status", "Current FAA delays/ground stops at an airport (live, context only).",
        {"airport": AIRPORT}, ["airport"]),
    _fn("explain_methodology", "Formulas, weights, thresholds, data sources and region definitions.",
        {"topic": {"type": "string", "enum": ["investment_score", "congestion", "unmet_demand", "long_haul",
                                              "load_factor", "data_sources", "regions", "all"]}}, ["topic"]),
]

TOOLS: dict[str, Callable] = {
    "get_airport_profile": get_airport_profile, "rank_airports": rank_airports,
    "compare_congestion": compare_congestion, "long_haul_share": long_haul_share,
    "unmet_demand": unmet_demand, "live_airport_status": live_airport_status,
    "explain_methodology": explain_methodology,
}


def execute_tool(name: str, arguments: str | dict, frames: Frames,
                 overrides: dict[str, Callable] | None = None) -> ToolCallRecord:
    """Run one tool call. Never raises: errors come back as an error payload for the LLM."""
    try:
        args = json.loads(arguments or "{}") if isinstance(arguments, str) else dict(arguments or {})
    except json.JSONDecodeError as exc:
        return _error(name, {}, f"invalid JSON arguments: {exc}")
    fn = (overrides or {}).get(name) or TOOLS.get(name)
    if fn is None:
        return _error(name, args, f"unknown tool '{name}'. Available: {', '.join(TOOLS)}")
    try:
        env, view = fn(frames, **args)
    except (ValueError, KeyError, TypeError) as exc:
        return _error(name, args, str(exc))
    return ToolCallRecord(name, args, env, compact(env, view))


def _error(name: str, args: dict, message: str) -> ToolCallRecord:
    payload = {"error": message, "hint": "ask the user to clarify, or retry with valid arguments"}
    return ToolCallRecord(name, args, envelope(payload, [], "n/a", [], []),
                          json.dumps(payload, ensure_ascii=False), error=message)
