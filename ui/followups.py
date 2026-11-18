"""Suggested follow-up questions under each answer, chosen deterministically from the tool that was used.

Pure functions: input = the tool-call records of one answer, output = 2-3 suggestions. Each suggestion carries
the exact tool call that answers it, so a click runs the tool directly (fast path, no LLM tool choice).
"""

import re
from dataclasses import dataclass, field
from typing import Iterable

from agent.tools import ToolCallRecord
from scoring.config import DEFAULT_LONG_HAUL_MI, LONG_HAUL_THRESHOLDS_MI

# Large hubs used to suggest "What about X?" (first one that is not the airport just shown).
PEER_HUBS = ["SFO", "BOS", "LAX", "JFK", "ORD", "DEN", "ATL", "SEA"]
MAX_SUGGESTIONS = 3
GROWTH_PRESET = "growth-led"     # "Rank them again with growth weighted higher" -> named preset (deterministic)


@dataclass(frozen=True)
class Suggestion:
    question: str
    tool: str
    args: dict = field(default_factory=dict, hash=False)


def _peer(exclude: set[str]) -> str:
    return next(code for code in PEER_HUBS if code not in exclude)


def _profile(airport: str) -> Suggestion:
    return Suggestion(f"Show the profile of {airport}", "get_airport_profile", {"airport": airport})


def _unmet(airport: str, question: str | None = None) -> Suggestion:
    return Suggestion(question or f"What is the unmet demand at {airport}?", "unmet_demand", {"airport": airport})


def _compare(a: str, b: str) -> Suggestion:
    return Suggestion(f"Compare {a} and {b} congestion levels", "compare_congestion", {"airports": [a, b]})


def from_rank(record: ToolCallRecord) -> list[Suggestion]:
    ranking = record.envelope["data"]["ranking"]
    scope = {k: v for k, v in record.args.items() if k in ("region", "states", "airports")}
    if ranking.get("preset") != GROWTH_PRESET:
        out = [Suggestion("Rank them again with growth weighted higher", "rank_airports",
                          {**scope, "preset": GROWTH_PRESET})]
    else:
        out = [Suggestion("Rank them again with default weights", "rank_airports", scope)]
    if ranking["ranking"]:
        top = ranking["ranking"][0]["airport"]
        out += [_profile(top), _unmet(top)]
        if len(ranking["ranking"]) > 1:                      # backup if one of the above was already asked
            out.append(_compare(top, ranking["ranking"][1]["airport"]))
    return out


def from_compare(record: ToolCallRecord) -> list[Suggestion]:
    rows = record.envelope["data"]["congestion"]["airports"]
    ordered = sorted(rows, key=lambda r: r["congestion_index"])
    lower, higher = ordered[0]["airport"], ordered[-1]["airport"]
    codes = [r["airport"] for r in rows]
    return [Suggestion(f"Why is {lower} lower?", "compare_congestion", {"airports": codes}),
            _unmet(higher), _profile(lower), _profile(higher), _unmet(lower)]


def from_unmet(record: ToolCallRecord) -> list[Suggestion]:
    airport = record.envelope["data"]["unmet_demand"]["airport"]
    peer = _peer({airport})
    return [_unmet(peer, f"What about {peer}?"), _profile(airport), _compare(airport, peer),
            Suggestion(f"Are there FAA delays at {airport} right now?", "live_airport_status", {"airport": airport})]


def from_long_haul(record: ToolCallRecord) -> list[Suggestion]:
    d = record.envelope["data"]
    other = min(LONG_HAUL_THRESHOLDS_MI) if d["default_threshold_mi"] == DEFAULT_LONG_HAUL_MI else DEFAULT_LONG_HAUL_MI
    return [Suggestion(f"What if long-haul starts at {other:,} miles?", "long_haul_share",
                       {"airport": d["airport"], "threshold_mi": other}),
            _profile(d["airport"]), _unmet(d["airport"])]


def from_profile(record: ToolCallRecord) -> list[Suggestion]:
    airport = record.envelope["data"]["profile"]["airport"]
    return [_unmet(airport), _compare(airport, _peer({airport})),
            Suggestion(f"Are there FAA delays at {airport} right now?", "live_airport_status", {"airport": airport})]


def from_live(record: ToolCallRecord) -> list[Suggestion]:
    airport = record.envelope["data"]["airport"]
    return [_profile(airport), _unmet(airport), _compare(airport, _peer({airport}))]


RULES = {"rank_airports": from_rank, "compare_congestion": from_compare, "unmet_demand": from_unmet,
         "long_haul_share": from_long_haul, "get_airport_profile": from_profile, "live_airport_status": from_live}


def normalize(question: str) -> str:
    """'Why is SNA lower?' == 'why is sna lower': lower-case, single spaces, no trailing punctuation."""
    return re.sub(r"\s+", " ", question.strip().lower()).rstrip(" ?.!")


def suggestions(records: list[ToolCallRecord], asked: Iterable[str] = ()) -> list[Suggestion]:
    """Up to 3 follow-ups based on the last successful tool call, never repeating a question already asked in
    this conversation (`asked`). Empty if none applies."""
    done = {normalize(q) for q in asked}
    for record in reversed(records or []):
        rule = RULES.get(record.name)
        if record.error or rule is None:
            continue
        try:
            out, seen = [], set(done)
            for s in rule(record):
                if normalize(s.question) not in seen:
                    seen.add(normalize(s.question))
                    out.append(s)
            return out[:MAX_SUGGESTIONS]
        except (KeyError, TypeError, IndexError, StopIteration):
            return []
    return []
