"""Number check: every number in the final answer must be traceable to a tool output.

Deterministic. Returns the numbers that could not be matched; the UI shows them as a warning.
"""

import re
from dataclasses import dataclass

from scoring.config import (
    CONGESTION_WEIGHTS, DEFAULT_LONG_HAUL_MI, INVESTMENT_WEIGHTS, LOAD_FACTOR_SENSITIVITY,
    LONG_HAUL_THRESHOLDS_MI, MIN_COMMERCIAL_RUNWAY_FT, TARGET_LOAD_FACTOR, UNMET_DEMAND_WEIGHTS,
    WEIGHT_PRESETS,
)

NUMBER_RE = re.compile(
    r"(?<![\w.\-/‐‑‒–—])"   # not part of a word, code or range like T-100 (any hyphen)
    r"(?<!\d,)"                                       # never start inside 30,802
    r"([+\-−‐‑‒–]?)"         # sign: ASCII/unicode minus and hyphens
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"           # 1,234,567 | 1234 | 12.5
    r"(?:st|nd|rd|th)?"                               # ordinals: 63.7th percentile
    r"(?:\s?(%|pp\b|percentage points?))?"            # percent markers
    r"(?:\s?(million|billion|thousand|[MBK]\b|k\b))?" # scale words
    r"(?![\w]|\.\d)",                                 # never stop in the middle of "63.7"
    re.IGNORECASE,
)
SCALE = {"million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9, "thousand": 1e3, "k": 1e3}


@dataclass
class Number:
    text: str
    value: float          # scaled value (36.5 million -> 36,500,000)
    decimals: int
    scale: float
    is_percent: bool
    context: str


def extract_numbers(text: str) -> list[Number]:
    out = []
    for m in NUMBER_RE.finditer(text or ""):
        sign, whole, frac, pct, scale_word = m.groups()
        value = float(whole.replace(",", "") + ("." + frac if frac else ""))
        if sign and sign != "+":
            value = -value
        scale = SCALE.get((scale_word or "").lower(), 1.0)
        start, end = max(0, m.start() - 40), min(len(text), m.end() + 40)
        out.append(Number(m.group(0).strip(), value * scale, len(frac or ""), scale, bool(pct),
                          text[start:end].replace("\n", " ")))
    return out


def _walk(obj, sink: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        sink.append(float(obj))
    elif isinstance(obj, str):
        sink.extend(n.value for n in extract_numbers(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk(k, sink)
            _walk(v, sink)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, sink)


def allowed_numbers(tool_outputs: list) -> set[float]:
    """All numbers in the tool outputs, plus x100 / /100 (fraction <-> percent) and absolute values."""
    raw: list[float] = []
    _walk(tool_outputs, raw)
    allowed = set()
    for y in raw:
        for v in (y, y * 100, y / 100):
            allowed.add(round(v, 6))
            allowed.add(round(abs(v), 6))
    return allowed


def config_constants() -> set[float]:
    """Assumption values from scoring/config.py (thresholds, targets, weights) that answers may quote."""
    values = {*LONG_HAUL_THRESHOLDS_MI, DEFAULT_LONG_HAUL_MI, MIN_COMMERCIAL_RUNWAY_FT, TARGET_LOAD_FACTOR,
              *LOAD_FACTOR_SENSITIVITY, 12, 15}
    for weights in (INVESTMENT_WEIGHTS, CONGESTION_WEIGHTS, UNMET_DEMAND_WEIGHTS, *WEIGHT_PRESETS.values()):
        values |= set(weights.values())
    return {round(v * f, 6) for v in values for f in (1, 100)}


def _ignored(n: Number, extra: set[float]) -> bool:
    if n.decimals == 0 and n.scale == 1 and not n.is_percent:
        if 1990 <= n.value <= 2100 or 0 <= abs(n.value) <= 10:
            return True
    return round(abs(n.value), 6) in extra


def _matches(n: Number, allowed: set[float]) -> bool:
    x = abs(n.value)
    tol = 0.5 * 10 ** (-n.decimals) * n.scale + 1e-9           # half a unit of the shown precision
    for y in allowed:
        if abs(x - y) <= tol:
            return True
        if x >= 1000 and y > 0 and abs(x - y) / y <= 0.005:      # rounded large numbers (abs values are in `allowed`)
            return True
    return False


def untraceable_numbers(answer: str, tool_outputs: list, user_text: str = "") -> list[dict]:
    """Numbers in `answer` that do not appear (within rounding) in any tool output."""
    allowed = allowed_numbers(tool_outputs)
    extra = config_constants() | {round(abs(n.value), 6) for n in extract_numbers(user_text)}
    flagged = []
    for n in extract_numbers(answer):
        if not _ignored(n, extra) and not _matches(n, allowed):
            flagged.append({"value": n.text, "context": n.context})
    return flagged


# ---------------------------------------------------------------- grounding check
# Catches answers that were not produced from this turn's tool calls (e.g. an invented re-ranking).

CODE_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{3}(?![A-Za-z0-9])")
# Upper-case words that are also US airport codes or that we print ourselves: never treated as airport mentions.
NON_AIRPORT_WORDS = {"FAA", "BTS", "NAS", "ATC", "DOT", "USA", "YOY", "CEO", "API", "KPI", "AND", "THE", "FOR", "ANY",
                     "ALL", "NOT", "TOP", "AVG", "MAX", "MIN", "PCT", "NET", "OPS", "SOC", "DFT"}
ACTION_RE = re.compile(r"\b(re-?rank\w*|rank\w*|compar\w*|compute\w*|calculat\w*|recalculat\w*|weight\w*)\b", re.I)
# A clear scope refusal needs no tool call (e.g. "Construction costs are out of scope").
REFUSAL_RE = re.compile(r"\bout of (?:the )?scope\b|\boutside (?:the |our |my |this )?(?:US )?scope\b|"
                        r"\bnot in (?:the )?scope\b|\bcovers US (?:commercial )?airports only\b|"
                        r"\bonly covers US\b|\bcan(?:no|')t help with\b", re.I)


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer or ""))


def airport_codes(text: str, known_codes: set[str]) -> set[str]:
    """Upper-case 3-letter tokens in text that are known US airport codes (excluding common acronyms)."""
    return {c for c in CODE_RE.findall(text or "") if c in known_codes and c not in NON_AIRPORT_WORDS}


def grounding_issues(answer: str, user_text: str, tools_called_now: list[str], tool_names: list[str],
                     grounded_text: str, known_codes: set[str]) -> list[str]:
    """Deterministic checks that the answer is grounded in tool calls. Returns human-readable issues.

    a. every airport code in the answer appears in this conversation's tool outputs or user messages
       (`grounded_text` = all tool outputs + all user messages so far, as text);
    b. the answer does not claim a tool that was not called in this turn: its identifier ("rank_airports") or
       "<name> tool" ("rank airports tool"). Plain topic words such as "unmet demand" are not tool claims;
    c. a request to rank / re-rank / compare / compute / change weights has at least one tool call this turn,
       unless the answer is a clear scope refusal.
    `tools_called_now` includes tool calls made by code on the fast path (example cards, follow-up buttons).
    """
    issues = []
    unknown = sorted(airport_codes(answer, known_codes) - set(CODE_RE.findall(grounded_text)))
    if unknown:
        issues.append(f"airport codes not found in any tool output or question: {', '.join(unknown)}")
    lowered = (answer or "").lower()
    claimed = [t for t in tool_names if t not in tools_called_now
               and (t in lowered or re.search(rf"\b{re.escape(t.replace('_', ' '))}\s+tool\b", lowered))]
    if claimed:
        issues.append(f"answer names tool(s) not called in this turn: {', '.join(claimed)}")
    if ACTION_RE.search(user_text or "") and not tools_called_now and not is_refusal(answer):
        issues.append("the request asks to rank/compare/compute or change weights, but no tool was called")
    return issues
