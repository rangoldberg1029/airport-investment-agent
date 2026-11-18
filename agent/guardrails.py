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
    r"(?<![\w.\-/])"                                  # not part of a word, code or range like T-100, 07L/25R
    r"([+\-−]?)"                                 # sign (incl. unicode minus)
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"           # 1,234,567 | 1234 | 12.5
    r"(?:\s?(%|pp\b|percentage points?))?"            # percent markers
    r"(?:\s?(million|billion|thousand|[MBK]\b|k\b))?" # scale words
    r"(?![\w])",
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
        if sign in ("-", "−"):
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
