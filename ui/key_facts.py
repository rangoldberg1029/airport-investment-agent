"""'Key facts (computed, not AI-generated)': deterministic sentences built from tool outputs.

Pure functions. Text comes from the tool envelopes (produced by Python code), never from the LLM.
Labels are human-readable and every value carries its unit.
"""

from agent.tools import ToolCallRecord, contribution_differences, weights_label

LABELS = {
    "lf_avg": "Average load factor", "lf_peak": "Peak-month load factor", "growth_yoy": "Passenger growth (YoY)",
    "delay_rate": "Arrival delay rate", "nas_share": "NAS share of delay minutes",
    "completion_gap": "Completion gap (US carriers)", "cancellation_rate": "Cancellation rate",
    "ops_per_runway": "Operations per runway", "demand_pressure": "Demand pressure (load factor)",
    "growth": "Passenger growth", "congestion": "Congestion index", "scale": "Scale (passengers)",
}
PERCENT = {"growth_yoy", "growth", "delay_rate", "nas_share", "completion_gap", "cancellation_rate"}


def label(component: str) -> str:
    return LABELS.get(component, component.replace("_", " ").capitalize())


def value_text(component: str, value) -> str:
    """Value with its unit, e.g. 33.39 -> '33.39%', 133594 -> '133,594 operations/runway/year'."""
    if value is None:
        return "n/a"
    if component in PERCENT:
        return f"{value}%"
    if component == "ops_per_runway":
        return f"{value:,.0f} operations/runway/year"
    if component in ("scale", "passengers"):
        return f"{value:,.0f} passengers"
    if component == "congestion":
        return f"{value} / 100"
    return f"{value}"            # load factors are ratios (passengers / seats)


def _view(record: ToolCallRecord) -> dict:
    return record.view or {}


MAX_UNRANKED = 5      # "Shown but not ranked": list at most this many, then "and N more"


def _gap_line(target: str, entry: dict) -> str:
    def part(g: dict) -> str:
        if g["extra_seats"] == 0:
            return f"0 extra seats (load factor {g['current_load_factor']})"
        return (f"+{g['extra_seats']:,} seats (+{g['extra_seats_pct']}%, about {g['extra_departures']:,} departures; "
                f"load factor {g['current_load_factor']})")
    lf = target.split("_")[1]
    return (f"Capacity gap at target load factor {lf}: 12-month window {part(entry['window'])}; "
            f"peak month {entry['peak_month']['month']} {part(entry['peak_month'])}")


def facts_rank(record: ToolCallRecord) -> list[str]:
    ranking = record.envelope["data"]["ranking"]
    stab = record.envelope["data"]["stability"]
    weights = weights_label(ranking["weights"])
    out = [weights[:1].upper() + weights[1:]]
    for r in ranking["ranking"][:3]:
        raw = {c["component"]: c["raw_value"] for c in r["components"]}
        out.append(f"#{r['rank']} {r['airport']} ({r['state']}, {r['hub']}): investment score "
                   f"{r['investment_score']} / 100, national rank {r['national_rank']} of {ranking['ranked_nationally']} "
                   f"(load factor {raw['demand_pressure']}, growth {raw['growth']}%, "
                   f"congestion index {raw['congestion']} / 100)")
    if not ranking["ranking"]:
        out.append("No airport in this selection is in the ranking universe (FAA large/medium/small hubs).")
    changes = record.envelope["data"].get("rank_change_vs_balanced")
    if changes:
        moved = [c for c in changes if not c.endswith("(same)")]
        out.append("Rank change vs the balanced preset: "
                   + ("; ".join(c.replace(" -> ", " → ") for c in moved) if moved else "no changes"))
    presets = "; ".join(f"{k}: {', '.join(v)}" for k, v in stab["rankings"].items())
    verdict = "is the same under all weight presets" if stab["same_top_n_set"] else "changes with the weight presets"
    out.append(f"Top {stab['top_n']} {verdict} ({presets})")
    outside = ranking["outside_ranking_universe"]
    if outside:
        names = ", ".join(f"{o['airport']} ({o['passengers']:,} passengers)" for o in outside[:MAX_UNRANKED])
        more = f", and {len(outside) - MAX_UNRANKED} more" if len(outside) > MAX_UNRANKED else ""
        out.append(f"Shown but not ranked (FAA non-hubs): {names}{more}")
    return out


def facts_compare(record: ToolCallRecord) -> list[str]:
    rows = record.envelope["data"]["congestion"]["airports"]
    out = [f"{r['airport']} ({r['hub']}): congestion index {r['congestion_index']} / 100, "
           f"national rank {r['national_rank']} of {r['ranked_airports']}" for r in rows]
    base = rows[0]
    contrib = {r["airport"]: {c["component"]: c["contribution"] for c in r["components"]} for r in rows}
    for other in rows[1:]:
        a, b = base["airport"], other["airport"]
        diffs = contribution_differences(contrib[a], contrib[b])
        parts = "; ".join(f"{label(k)} raises {a if v > 0 else b} by {abs(v):.2f} pts" for k, v in diffs if v)
        out.append(f"Why the indices differ ({a} vs {b}, largest first): {parts}")
    view = _view(record)
    out += [i for i in view.get("interpretation", []) if "runway" in i]
    return out


def facts_long_haul(record: ToolCallRecord) -> list[str]:
    d = record.envelope["data"]
    sens = "; ".join(f"at {s['threshold_mi']:,} mi: {s['departure_share_pct']}% of departures, "
                     f"{s['seat_share_pct']}% of seats" for s in d["sensitivity"])
    return [
        f"{d['airport']}: {d['long_haul_departure_share_pct']}% of {d['total_passenger_departures']:,} passenger "
        f"departures fly {d['default_threshold_mi']:,} statute miles or more ({d['long_haul_seat_share_pct']}% of seats)",
        f"Threshold sensitivity: {sens}",
    ]


def facts_unmet(record: ToolCallRecord) -> list[str]:
    d = record.envelope["data"]["unmet_demand"]
    out = [f"{d['airport']} ({d['hub']}): unmet-demand proxy score {d['unmet_demand_score']} / 100"]
    if d["drivers_above_peer_median"]:
        drivers = "; ".join(f"{label(x['component'])} {value_text(x['component'], x['value'])} "
                            f"(peer median {value_text(x['component'], x['peer_median'])})"
                            for x in d["drivers_above_peer_median"])
        out.append(f"Above the {d['peer_group'].split(' in ')[0]} median: {drivers}")
    out += [_gap_line(target, entry) for target, entry in d["capacity_gap"].items()]
    return out


def facts_profile(record: ToolCallRecord) -> list[str]:
    v = _view(record)
    rank = f" (national rank {v['investment_national_rank']})" if v["investment_national_rank"] else ""
    return [
        f"{v['airport']} ({v['hub']}, {v['state']}): {v['passengers']}; average load factor "
        f"{v['load_factor_avg'].replace(' LF', '')} (peak {v['load_factor_peak'].replace(' LF', '')} in {v['peak_month']}); "
        f"passenger growth {v['growth_yoy']}",
        f"Arrival delay rate {v['delay_rate']} (national {v['national']['delay_rate']}); "
        f"NAS share of delay minutes {v['nas_share']} (national {v['national']['nas_share']})",
        f"Congestion index {v['congestion_index'].replace(' index', ' / 100')} (national rank "
        f"{v['congestion_national_rank']}); investment score "
        f"{v['investment_score'] or 'n/a (outside the ranking universe)'}{rank}",
        v["runway_rule_interpretation"],
    ]


def facts_live(record: ToolCallRecord) -> list[str]:
    d = record.envelope["data"]
    stamp = f" (FAA feed updated {d['feed_timestamp']})" if d.get("feed_timestamp") else ""
    out = [f"{d['airport']}: {d['status']}{stamp}"]
    out += d.get("event_descriptions", [])           # readable lines; events with nothing to show are dropped
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
            out.append(f"Out of scope: {r.error}" if r.meta.get("out_of_scope")
                       else f"Tool {r.name} returned an error: {r.error}")
            continue
        fn = FACTS.get(r.name)
        if fn is None:
            continue
        try:
            out += fn(r)
        except (KeyError, TypeError, IndexError, AttributeError):
            out.append(f"Key facts unavailable for {r.name} (see Evidence)")
    return out
