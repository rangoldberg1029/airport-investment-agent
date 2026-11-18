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
    CONGESTION_WEIGHTS, DEFAULT_LONG_HAUL_MI, INVESTMENT_WEIGHTS, LOAD_FACTOR_SENSITIVITY,
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


def comp_rows(components: list[dict]) -> list[list]:
    """[component, raw, percentile, weight, contribution] with integer percentiles."""
    return [[c["component"], c["raw_value"], None if c["percentile"] is None else round(c["percentile"]),
             c["weight"], c["contribution"]] for c in components]


COMP_HEADER = ["component", "raw", "pctile", "weight", "contrib"]


def _ft(label: str) -> int:
    """'7L/25R (12,894 ft)' -> 12894 (parsing a label produced by scoring.metrics)."""
    return int(label.rsplit("(", 1)[1].split(" ft")[0].replace(",", ""))


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
    data = {
        "airport": iata, "hub": p["hub"], "state": p["state"],
        "passengers": p["passengers"], "seats": p["seats"], "departures": p["departures"],
        "load_factor_avg": p["load_factor_avg"], "load_factor_peak": p["load_factor_peak_month"],
        "peak_month": p["peak_month"], "growth_yoy_pct": p["growth_yoy_pct"],
        "delay_rate_pct": d["delay_rate_pct"], "nas_share_pct": d["nas_share_pct"],
        "cancellation_rate_pct": d["cancellation_rate_pct"],
        "national": d["national"], "hub_peer_median": d["hub_peer_median"],
        "delay_rate_minus_national_pp": diff(d["delay_rate_pct"], d["national"]["delay_rate_pct"]),
        "nas_share_minus_national_pp": diff(d["nas_share_pct"], d["national"]["nas_share_pct"]),
        "delay_rate_minus_peer_median_pp": diff(d["delay_rate_pct"], d["hub_peer_median"]["delay_rate_pct"]),
        "completion_gap_pct_us_carriers": gap["data"]["completion_gap_pct"],
        "operations_per_runway": o["operations_per_runway"], "commercial_runways": o["runway_count"],
        "congestion_index": cong["data"]["airports"][0]["congestion_index"],
        "congestion_national_rank": cong["data"]["airports"][0]["national_rank"],
        "investment_score": inv_row["investment_score"] if inv_row else None,
        "investment_national_rank": inv_row["national_rank"] if inv_row else None,
        "ranked_airports": inv["data"]["ranked_nationally"],
        "in_ranking_universe": inv_row is not None,
    }
    full = merge([prof, dly, gap, ops, inv, cong], data,
                 f"{prof['data_period']}; operations: {o['period']}")
    return full, data


def rank_airports(frames: Frames, region: str | None = None, states: list[str] | None = None,
                  airports: list[str] | None = None, weights: dict | None = None, top_n: int = 5) -> tuple[dict, dict]:
    notes = []
    if airports:
        resolved = []
        for a in airports:
            iata, note = resolve_airport(a, frames.airports)
            resolved.append(iata)
            if note:
                notes.append(note)
        airports = resolved
    inv = ranking.investment_scores(frames, region=region, states=states, airports=airports, weights=weights)
    stab = ranking.rank_stability(frames, region=region, states=states, airports=airports, top_n=3)
    rows = inv["data"]["ranking"]
    table = []
    for i, r in enumerate(rows[:top_n]):
        raw = {c["component"]: c["raw_value"] for c in r["components"]}
        nxt = rows[i + 1]["investment_score"] if i + 1 < len(rows) else None
        table.append([r["rank"], r["airport"], r["state"], r["hub"], r["investment_score"], r["national_rank"],
                      raw["demand_pressure"], raw["growth"], raw["congestion"], raw["scale"],
                      [c["contribution"] for c in r["components"]], diff(r["investment_score"], nxt, 1)])
    data = {
        "header": ["rank", "airport", "state", "hub", "score", "natl_rank", "load_factor", "growth_pct",
                   "congestion_idx", "passengers", "contrib[demand,growth,congestion,scale]", "gap_to_next"],
        "rows": table,
        "n_more": max(0, len(rows) - top_n),
        "ranked_nationally": inv["data"]["ranked_nationally"],
        "weights": inv["data"]["weights"],
        "outside_ranking_universe": [o["airport"] for o in inv["data"]["outside_ranking_universe"]],
        "stability_top3": stab["data"]["rankings"],
        "stability_same_set": stab["data"]["same_top_n_set"],
    }
    full = merge([inv, stab], {"ranking": inv["data"], "stability": stab["data"]})
    full["assumptions"] = notes + full["assumptions"]
    return full, data


COMPARE_METRICS = [  # (label, unit for differences, getter)
    ("congestion_index", "index pts", lambda c, p, d, o: c["congestion_index"]),
    ("delay_rate_pct", "pp", lambda c, p, d, o: d["delay_rate_pct"]),
    ("nas_share_pct", "pp", lambda c, p, d, o: d["nas_share_pct"]),
    ("cancellation_rate_pct", "pp", lambda c, p, d, o: d["cancellation_rate_pct"]),
    ("operations_per_runway", "operations", lambda c, p, d, o: o["operations_per_runway"]),
    ("commercial_runways", "runways", lambda c, p, d, o: o["runway_count"]),
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
    values = {}
    for label, unit, get in COMPARE_METRICS:
        values[label] = {a: get(rows[i], profs[i]["data"], dlys[i]["data"], opss[i]["data"]) for i, a in enumerate(codes)}
    base = codes[0]
    differences = {
        f"{base}_minus_{other}": {label: [diff(values[label][base], values[label][other], 3), unit]
                                  for label, unit, _ in COMPARE_METRICS}
        for other in codes[1:]
    }
    contrib = {a: {c[0]: c[4] for c in comp_rows(rows[i]["components"])} for i, a in enumerate(codes)}
    contrib_diff = {f"{base}_minus_{o}": {k: diff(contrib[base][k], contrib[o][k]) for k in contrib[base]}
                    for o in codes[1:]}
    data = {
        "airports": codes,
        "hubs": {r["airport"]: r["hub"] for r in rows},
        "national_rank_congestion": {r["airport"]: r["national_rank"] for r in rows},
        "ranked_airports": rows[0]["ranked_airports"],
        "values": values,
        "national": {k: v for k, v in dlys[0]["data"]["national"].items() if k != "arrivals"},
        "differences": differences,
        "index_components_header": ["component", "pctile", "weight", "contrib"],
        "index_components": {r["airport"]: [[c[0], c[2], c[3], c[4]] for c in comp_rows(r["components"])]
                             for r in rows},
        "contribution_differences": contrib_diff,
        "runway_lengths_ft_counted": {o["data"]["airport"]: [_ft(x) for x in o["data"]["runways_counted"]] for o in opss},
        "runway_lengths_ft_excluded_short": {o["data"]["airport"]: [_ft(x) for x in o["data"]["runways_excluded_short"]]
                                             for o in opss},
    }
    full = merge([cong, *profs, *dlys, *opss], data,
                 f"Delay & traffic: {window_label(WINDOW)}; Operations (Socrata): {opss[0]['data']['period']}")
    return full, data


def long_haul_share(frames: Frames, airport: str, threshold_mi: int | None = None) -> tuple[dict, dict]:
    default = int(threshold_mi) if threshold_mi else DEFAULT_LONG_HAUL_MI
    env = metrics.long_haul_share(frames, airport, default=default)
    d = env["data"]
    key = f"top_routes_at_or_above_{default}_mi"
    data = {
        "airport": d["airport"], "threshold_mi": default,
        "departure_share_pct": d["long_haul_departure_share_pct"], "seat_share_pct": d["long_haul_seat_share_pct"],
        "total_passenger_departures": d["total_passenger_departures"],
        "sensitivity": [[s["threshold_mi"], s["departure_share_pct"], s["seat_share_pct"]] for s in d["sensitivity"]],
        "sensitivity_header": ["threshold_mi", "departure_share_pct", "seat_share_pct"],
        "top_routes": [[r["dest"], r["dest_country"], int(r["distance_mi"]), int(r["departures"]), int(r["seats"])]
                       for r in d[key]],
        "top_routes_header": ["dest", "country", "distance_mi", "departures", "seats"],
    }
    return env, data


def unmet_demand(frames: Frames, airport: str) -> tuple[dict, dict]:
    u = ranking.unmet_demand(frames, airport)
    d = u["data"]
    dly = metrics.delay_profile(frames, d["airport"])
    dd = dly["data"]
    gaps = []
    for target, parts in d["capacity_gap"].items():
        for period, g in parts.items():
            gaps.append([g["target_load_factor"], period, g["current_load_factor"], g["extra_seats"],
                         g["extra_seats_pct"], g["extra_departures"]])
    data = {
        "airport": d["airport"], "hub": d["hub"], "unmet_demand_score": d["unmet_demand_score"],
        "components_header": COMP_HEADER, "components": comp_rows(d["components"]),
        "drivers": [[x["component"], x["value"], x["peer_median"],
                     diff(x["value"], x["peer_median"], 3), round(x["percentile"])]
                    for x in d["drivers_above_peer_median"]],
        "drivers_header": ["component", "value", "peer_median", "value_minus_median", "pctile"],
        "peer_group": d["peer_group"],
        "vs_national": {
            "delay_rate_pct": [dd["delay_rate_pct"], dd["national"]["delay_rate_pct"],
                               diff(dd["delay_rate_pct"], dd["national"]["delay_rate_pct"])],
            "nas_share_pct": [dd["nas_share_pct"], dd["national"]["nas_share_pct"],
                              diff(dd["nas_share_pct"], dd["national"]["nas_share_pct"])],
            "header": ["airport", "national", "difference_pp"],
        },
        "capacity_gap_header": ["target_lf", "period", "current_lf", "extra_seats", "extra_seats_pct", "extra_departures"],
        "capacity_gap": gaps,
    }
    return merge([u, dly], {"unmet_demand": d, "delay_profile": dd}), data


def live_airport_status(frames: Frames, airport: str,
                        fetch: Callable[[], tuple[list[dict], str]] = nas_status.fetch_feed) -> tuple[dict, dict]:
    iata, note = resolve_airport(airport, frames.airports)
    caveats = ["Live FAA NAS Status: context only, not used in any score",
               "Feed is public but not a formally documented API; format may change"]
    try:
        events, stamp = fetch()
    except Exception as exc:
        data = {"airport": iata, "status": "live FAA status unavailable right now", "error": type(exc).__name__}
        return envelope(data, ["FAA NAS Status (live)"], "live", [note] if note else [], caveats), data
    found = nas_status.parse_airport_events(events, iata)
    data = {
        "airport": iata, "feed_timestamp": stamp,
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
            "weights": INVESTMENT_WEIGHTS, "presets_for_stability": WEIGHT_PRESETS,
            "ranking_universe": f"FAA hubs {'/'.join(RANKING_UNIVERSE_HUBS)}; percentiles national",
            "override": "user can override any weight; weights are renormalised to sum 1",
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
        "airport list, with rank stability across weight presets. Use weights to re-rank.",
        {"region": {"type": "string", "enum": sorted(REGIONS)},
         "states": {"type": "array", "items": {"type": "string"}, "description": "2-letter state codes"},
         "airports": {"type": "array", "items": {"type": "string"}},
         "weights": {"type": "object", "description": "partial override, keys: demand_pressure, growth, congestion, scale",
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
