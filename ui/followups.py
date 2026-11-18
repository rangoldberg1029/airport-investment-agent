"""Suggested follow-up questions under each answer, chosen deterministically from the tool that was used.

Pure functions: input = the tool-call records of one answer, output = 2-3 question strings.
"""

from agent.tools import ToolCallRecord
from scoring.config import DEFAULT_LONG_HAUL_MI, INVESTMENT_WEIGHTS, LONG_HAUL_THRESHOLDS_MI

# Large hubs used to suggest "What about X?" (first one that is not the airport just shown).
PEER_HUBS = ["SFO", "BOS", "LAX", "JFK", "ORD", "DEN", "ATL", "SEA"]
MAX_SUGGESTIONS = 3


def _peer(exclude: set[str]) -> str:
    return next(code for code in PEER_HUBS if code not in exclude)


def _rank(record: ToolCallRecord) -> list[str]:
    ranking = record.envelope["data"]["ranking"]
    out = []
    weights = ranking["weights"]
    if weights.get("growth", 0) <= INVESTMENT_WEIGHTS["growth"]:
        out.append("Rank them again with growth weighted higher")
    else:
        out.append("Rank them again with default weights")
    if ranking["ranking"]:
        top = ranking["ranking"][0]["airport"]
        out += [f"Show the profile of {top}", f"What is the unmet demand at {top}?"]
    return out


def _compare(record: ToolCallRecord) -> list[str]:
    rows = record.envelope["data"]["congestion"]["airports"]
    ordered = sorted(rows, key=lambda r: r["congestion_index"])
    lower, higher = ordered[0]["airport"], ordered[-1]["airport"]
    return [f"Why is {lower} lower?", f"What is the unmet demand at {higher}?", f"Show the profile of {lower}"]


def _unmet(record: ToolCallRecord) -> list[str]:
    airport = record.envelope["data"]["unmet_demand"]["airport"]
    return [f"What about {_peer({airport})}?", f"Show the profile of {airport}",
            f"Compare {airport} and {_peer({airport})} congestion levels"]


def _long_haul(record: ToolCallRecord) -> list[str]:
    d = record.envelope["data"]
    other = min(LONG_HAUL_THRESHOLDS_MI) if d["default_threshold_mi"] == DEFAULT_LONG_HAUL_MI else DEFAULT_LONG_HAUL_MI
    return [f"What if long-haul starts at {other:,} miles?", f"Show the profile of {d['airport']}"]


def _profile(record: ToolCallRecord) -> list[str]:
    airport = record.envelope["data"]["profile"]["airport"]
    peer = _peer({airport})
    return [f"What is the unmet demand at {airport}?", f"Compare {airport} and {peer} congestion levels",
            f"Are there FAA delays at {airport} right now?"]


def _live(record: ToolCallRecord) -> list[str]:
    airport = record.envelope["data"]["airport"]
    return [f"Show the profile of {airport}", f"What is the unmet demand at {airport}?"]


RULES = {"rank_airports": _rank, "compare_congestion": _compare, "unmet_demand": _unmet,
         "long_haul_share": _long_haul, "get_airport_profile": _profile, "live_airport_status": _live}


def suggestions(records: list[ToolCallRecord]) -> list[str]:
    """2-3 follow-up questions based on the last successful tool call (empty if none applies)."""
    for record in reversed(records or []):
        rule = RULES.get(record.name)
        if record.error or rule is None:
            continue
        try:
            return list(dict.fromkeys(rule(record)))[:MAX_SUGGESTIONS]
        except (KeyError, TypeError, IndexError, StopIteration):
            return []
    return []
