"""Deterministic answer headlines: the "Answer" line of every reply is built by code from the tool output.

The LLM writes only the "Why / drivers" bullets under it. So the core answer (who ranks first, which index is
higher, what share, what score) is always exactly what the scoring layer computed. Pure functions.
"""

import json
import re

from agent.tools import ToolCallRecord, contribution_differences, weights_label
from scoring.common import window_label
from scoring.config import (
    CONGESTION_WEIGHTS, DEFAULT_LONG_HAUL_MI, INVESTMENT_WEIGHTS, LONG_HAUL_THRESHOLDS_MI, RANKING_UNIVERSE_HUBS,
    REGIONS, TARGET_LOAD_FACTOR, UNMET_DEMAND_WEIGHTS, WINDOW,
)

PERIOD = window_label(WINDOW)
TOP_N = 3
MAX_BULLETS = 4
NEAR_EQUAL_POINTS = 1.0       # congestion indices closer than this are called "within 1 point", not "higher"

LABELS = {
    "demand_pressure": "demand pressure (load factor)", "growth": "growth", "congestion": "congestion",
    "scale": "scale", "delay_rate": "arrival delay rate", "nas_share": "NAS share of delay minutes",
    "ops_per_runway": "operations per runway", "cancellation_rate": "cancellation rate",
    "lf_avg": "average load factor", "lf_peak": "peak-month load factor", "growth_yoy": "passenger growth",
    "completion_gap": "completion gap",
}


def _view(record: ToolCallRecord) -> dict:
    return json.loads(record.llm_view)["data"]


def _weights(weights: dict) -> str:
    return ", ".join(f"{LABELS.get(k, k)} {v:.0%}" for k, v in weights.items())


# ---------------------------------------------------------------- one headline per tool

def rank_headline(record: ToolCallRecord) -> str:
    v, args = _view(record), record.args
    rows = v["rows"][:TOP_N]
    if not rows:
        return f"No airport in this selection is in the ranking universe (FAA {'/'.join(RANKING_UNIVERSE_HUBS)} hubs)."
    if args.get("region"):
        scope = f"{str(args['region']).title()} candidates"
    elif args.get("states"):
        scope = f"candidates in {'/'.join(s.upper() for s in args['states'])}"
    elif args.get("airports"):
        scope = "candidates among the selected airports"
    else:
        scope = "US candidates"
    items = "; ".join(f"{r[0]}. {r[1]} {r[2]} ({r[5]})" for r in rows)
    text = (f"Top {len(rows)} {scope} for terminal expansion (investment score 0-100, "
            f"{weights_label(v['weights'])}, {PERIOD}): {items}.")
    if len(v["rows"]) > TOP_N:
        text += (f" The top {TOP_N} is the same under all weight presets." if v["stability_same_set"]
                 else f" The top {TOP_N} changes with the weight presets.")
    return text


def compare_headline(record: ToolCallRecord) -> str:
    rows = record.envelope["data"]["congestion"]["airports"]
    names = _view(record)["names"]
    parts = " vs ".join(f"{r['airport']} {names[r['airport']]} {r['congestion_index']} "
                        f"(national rank {r['national_rank']} of {r['ranked_airports']})" for r in rows)
    text = f"Congestion index (0-100, higher = more congested, {PERIOD}): {parts}."
    ordered = sorted(rows, key=lambda r: -r["congestion_index"])
    top, second = ordered[0], ordered[1]
    if top["congestion_index"] == second["congestion_index"]:
        text += " The highest indices are tied."
    elif top["congestion_index"] - second["congestion_index"] < NEAR_EQUAL_POINTS:
        text += (f" The indices are within {NEAR_EQUAL_POINTS:g} point of each other "
                 f"({top['airport']} marginally higher).")
    else:
        text += f" {top['airport']} is the most congested." if len(rows) > 2 else \
            f" {top['airport']} is the more congested."
    if len(rows) == 2:
        a, b = rows
        contrib = {r["airport"]: {c["component"]: c["contribution"] for c in r["components"]} for r in rows}
        component, value = contribution_differences(contrib[a["airport"]], contrib[b["airport"]])[0]
        raises = a["airport"] if value > 0 else b["airport"]
        text += (f" Largest single driver of the difference: {LABELS.get(component, component)} "
                 f"({abs(value):.2f} index points, raises {raises}).")
    return text


def long_haul_headline(record: ToolCallRecord) -> str:
    v = _view(record)
    departures = v["total_passenger_departures"].replace(" departures", " passenger departures")
    text = (f"{v['airport']} {v['name']}: {v['departure_share']} of {departures} fly {v['threshold']} or more "
            f"({v['seat_share']} of seats, {PERIOD}).")
    others = [f"at {t}: {d}" for t, d, _ in v["sensitivity"] if t != v["threshold"]]
    if others:
        text += " Departure share " + "; ".join(others) + "."
    return text


def unmet_headline(record: ToolCallRecord) -> str:
    v = _view(record)
    d = record.envelope["data"]["unmet_demand"]
    text = (f"{v['airport']} {v['name']} ({v['hub']}): unmet-demand proxy score {v['unmet_demand_score']} "
            f"({PERIOD}; a proxy - true spilled demand is not observable).")
    drivers = sorted(v["drivers_above_peer_median"], key=lambda x: -x[4])[:2]
    if drivers:
        peers = d["peer_group"].split(" in ")[0]
        text += " Strongest pressure vs the median of " + peers + ": " + "; ".join(
            f"{LABELS.get(c, c)} {value} (median {median})" for c, value, median, _, _ in drivers) + "."
    gap = d["capacity_gap"].get(f"target_{TARGET_LOAD_FACTOR:.2f}")
    if gap:
        def seats(g: dict) -> str:
            return "0 extra seats" if g["extra_seats"] == 0 else f"+{g['extra_seats']:,} seats (+{g['extra_seats_pct']}%)"
        text += (f" Capacity gap at target load factor {TARGET_LOAD_FACTOR:.2f}: {seats(gap['window'])} over the "
                 f"12 months; {seats(gap['peak_month'])} in the peak month ({gap['peak_month']['month']}).")
    return text


def profile_headline(record: ToolCallRecord) -> str:
    v = _view(record)
    score = (f"investment score {v['investment_score']} (national rank {v['investment_national_rank']})"
             if v["investment_score"] else "outside the investment ranking universe (FAA non-hub)")
    return (f"{v['airport']} {v['name']} ({v['hub']}, {v['state']}, {PERIOD}): {v['passengers']}, average load "
            f"factor {v['load_factor_avg'].replace(' LF', '')}, passenger growth {v['growth_yoy']}, congestion "
            f"index {v['congestion_index'].replace(' index', '')} (national rank {v['congestion_national_rank']}), "
            f"{score}.")


def live_headline(record: ToolCallRecord) -> str:
    d = record.envelope["data"]
    stamp = f", feed updated {d['feed_timestamp']}" if d.get("feed_timestamp") else ""
    return (f"{d['airport']} {d['name']}: {d['status']} (FAA NAS Status, live{stamp}). "
            "Live context only; not used in any score.")


def methodology_headline(record: ToolCallRecord) -> str:
    topic = record.args.get("topic", "all")
    lines = {
        "investment_score": "Investment score (0-100) = weighted national percentiles vs FAA "
                            f"{'/'.join(RANKING_UNIVERSE_HUBS)} hubs: {_weights(INVESTMENT_WEIGHTS)}.",
        "congestion": f"Congestion index (0-100) = weighted percentiles: {_weights(CONGESTION_WEIGHTS)}.",
        "unmet_demand": f"Unmet-demand proxy (0-100) = weighted percentiles: {_weights(UNMET_DEMAND_WEIGHTS)}. "
                        "True spilled demand is not observable.",
        "long_haul": f"Long-haul = passenger departures with segment distance >= {DEFAULT_LONG_HAUL_MI:,} statute "
                     f"miles (sensitivity {' / '.join(f'{t:,}' for t in LONG_HAUL_THRESHOLDS_MI)} mi).",
        "load_factor": f"Load factor = passengers / seats on departing passenger segments ({PERIOD}).",
        "data_sources": "Data sources: BTS T-100, BTS delay causes, BTS Socrata, FAA enplanements, OurAirports, "
                        "FAA NAS Status (live); details in Evidence.",
        "regions": "Regions: " + "; ".join(f"{k.title()} = {', '.join(v)}" for k, v in REGIONS.items()) + ".",
    }
    return lines.get(topic, "Methodology: investment score, congestion index, unmet-demand proxy, long-haul share "
                            "and data sources, all set in scoring/config.py (details in Evidence).")


HEADLINES = {
    "rank_airports": rank_headline, "compare_congestion": compare_headline, "long_haul_share": long_haul_headline,
    "unmet_demand": unmet_headline, "get_airport_profile": profile_headline, "live_airport_status": live_headline,
    "explain_methodology": methodology_headline,
}


def interpretation_notes(records: list[ToolCallRecord]) -> list[str]:
    """Airport-name interpretations stated by the tools, e.g. "'LA' interpreted as LAX (...)"."""
    notes = []
    for r in records:
        for a in r.envelope.get("assumptions", []) if not r.error else []:
            if " interpreted as " in a and a not in notes:
                notes.append(a)
    return notes


def answer_headline(records: list[ToolCallRecord]) -> str:
    """The deterministic Answer text for one turn (empty if no tool call succeeded)."""
    lines = []
    for r in records:
        fn = HEADLINES.get(r.name)
        if r.error or fn is None:
            continue
        try:
            line = fn(r)
        except (KeyError, TypeError, IndexError, ValueError, AttributeError, json.JSONDecodeError):
            continue
        if line not in lines:
            lines.append(line)
    if lines:
        lines += [f"Note: {n}." for n in interpretation_notes(records)]
    return "  \n".join(lines)


# ---------------------------------------------------------------- combine with the LLM's explanation

BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
WHY_HEADING_RE = re.compile(r"^[#*_\s]*why\s*(?:/|and)\s*drivers?[*_:\s]*$", re.I)
ANSWER_RE = re.compile(r"^[#*_\s]*answer\b", re.I)


def why_bullets(text: str, max_bullets: int = MAX_BULLETS) -> str:
    """The LLM's explanation without an "Answer" paragraph or "Why / drivers" heading of its own (the app adds
    both), keeping at most `max_bullets` bullets."""
    out, bullets, skipping = [], 0, False
    for line in text.strip().splitlines():
        if ANSWER_RE.match(line):
            skipping = True                      # drop the model's own Answer paragraph
            continue
        if WHY_HEADING_RE.match(line):
            skipping = False
            continue
        is_bullet = bool(BULLET_RE.match(line))
        if skipping and (is_bullet or not line.strip()):
            skipping = False
        if skipping:
            continue
        if is_bullet:
            bullets += 1
            if bullets > max_bullets:
                continue
        out.append(line)
    return "\n".join(out).strip()


def compose(headline: str, explanation: str, why_heading: bool = True) -> str:
    """Markdown shown to the user: computed Answer line, then the LLM's Why / drivers bullets."""
    if not headline:
        return explanation
    text = f"**Answer** - {headline}"
    if explanation:
        text += ("\n\n**Why / drivers**\n" if why_heading else "\n\n") + explanation
    return text
