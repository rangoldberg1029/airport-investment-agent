"""'Key facts (computed, not AI-generated)': deterministic sentences built from tool outputs.

Pure functions. The text comes from the tool envelopes and compact views (both produced by Python code),
never from the LLM.
"""

import json

from agent.tools import ToolCallRecord


def _view(record: ToolCallRecord) -> dict:
    try:
        return json.loads(record.llm_view).get("data", {})
    except (json.JSONDecodeError, AttributeError):
        return {}


def facts_rank(record: ToolCallRecord) -> list[str]:
    ranking = record.envelope["data"]["ranking"]
    stab = record.envelope["data"]["stability"]
    out = []
    for r in ranking["ranking"][:3]:
        out.append(f"#{r['rank']} {r['airport']} ({r['state']}, {r['hub']}): investment score "
                   f"{r['investment_score']} / 100, national rank {r['national_rank']} of {ranking['ranked_nationally']}")
    if not ranking["ranking"]:
        out.append("No airport in this selection is in the ranking universe (FAA large/medium/small hubs).")
    top = ", ".join(f"{k}: {', '.join(v)}" for k, v in stab["rankings"].items())
    same = "unchanged" if stab["same_top_n_set"] else "changes"
    out.append(f"Top {stab['top_n']} across weight presets ({same}): {top}")
    if ranking["outside_ranking_universe"]:
        names = ", ".join(o["airport"] for o in ranking["outside_ranking_universe"])
        out.append(f"Shown but not ranked (FAA non-hubs): {names}")
    return out


def facts_compare(record: ToolCallRecord) -> list[str]:
    view = _view(record)
    rows = record.envelope["data"]["congestion"]["airports"]
    out = [f"{r['airport']} ({r['hub']}): congestion index {r['congestion_index']} / 100, "
           f"national rank {r['national_rank']} of {r['ranked_airports']}" for r in rows]
    out += [d["summary"] for d in view.get("drivers_of_index_difference", {}).values()]
    out += [i for i in view.get("interpretation", []) if "runway" in i]
    return out


def facts_long_haul(record: ToolCallRecord) -> list[str]:
    d = record.envelope["data"]
    sens = "; ".join(f">= {s['threshold_mi']:,} mi: {s['departure_share_pct']}% of departures, "
                     f"{s['seat_share_pct']}% of seats" for s in d["sensitivity"])
    return [
        f"{d['airport']}: {d['long_haul_departure_share_pct']}% of {d['total_passenger_departures']:,} passenger "
        f"departures fly >= {d['default_threshold_mi']:,} statute miles ({d['long_haul_seat_share_pct']}% of seats)",
        f"Threshold sensitivity: {sens}",
    ]


def facts_unmet(record: ToolCallRecord) -> list[str]:
    d = record.envelope["data"]["unmet_demand"]
    out = [f"{d['airport']} ({d['hub']}): unmet-demand proxy score {d['unmet_demand_score']} / 100"]
    if d["drivers_above_peer_median"]:
        drivers = ", ".join(f"{x['component']} {x['value']} (peer median {x['peer_median']})"
                            for x in d["drivers_above_peer_median"])
        out.append(f"Above the {d['peer_group'].split(' in ')[0]} median: {drivers}")
    out += [f"Target LF {target.split('_')[1]}: {entry['interpretation']}" for target, entry in d["capacity_gap"].items()]
    return out


def facts_profile(record: ToolCallRecord) -> list[str]:
    v = _view(record)
    out = [f"{v['airport']} ({v['hub']}, {v['state']}): {v['passengers']}, load factor {v['load_factor_avg']} "
           f"(peak {v['load_factor_peak']} in {v['peak_month']}), growth {v['growth_yoy']}",
           f"Delay rate {v['delay_rate']} (national {v['national']['delay_rate']}); NAS share {v['nas_share']} "
           f"(national {v['national']['nas_share']})",
           f"Congestion index {v['congestion_index']} (rank {v['congestion_national_rank']}); "
           f"investment score {v['investment_score'] or 'n/a (outside ranking universe)'}"
           + (f" (rank {v['investment_national_rank']})" if v["investment_national_rank"] else ""),
           v["runway_rule_interpretation"]]
    return out


def facts_live(record: ToolCallRecord) -> list[str]:
    d = record.envelope["data"]
    out = [f"{d['airport']}: {d['status']}" + (f" (FAA feed updated {d['feed_timestamp']})" if d.get("feed_timestamp") else "")]
    for e in d.get("events", []):
        details = ", ".join(f"{k} {v}" for k, v in e.items() if k != "type")
        out.append(f"{e['type']}: {details}" if details else e["type"])
    return out


FACTS = {
    "rank_airports": facts_rank, "compare_congestion": facts_compare, "long_haul_share": facts_long_haul,
    "unmet_demand": facts_unmet, "get_airport_profile": facts_profile, "live_airport_status": facts_live,
}


def key_facts(records: list[ToolCallRecord]) -> list[str]:
    """All key-fact lines for one answer, in tool-call order. Errors are reported, never raised."""
    out = []
    for r in records:
        if r.error:
            out.append(f"Tool {r.name} returned an error: {r.error}")
            continue
        fn = FACTS.get(r.name)
        if fn is None:
            continue
        try:
            out += fn(r)
        except (KeyError, TypeError, IndexError):
            out.append(f"Key facts unavailable for {r.name} (see Evidence)")
    return out
