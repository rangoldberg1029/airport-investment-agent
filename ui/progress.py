"""Progress reporting inside st.status: tool steps and a countdown while the LLM provider is at capacity."""

import json
import math
import time

ANALYZING = "Analyzing question..."
COMPOSING = "Facts ready · writing explanation…"
RETRYING = "Answer failed the grounding check - retrying with a tool call..."


def tool_label(name: str, args: str | dict) -> str:
    """'compare_congestion', '{"airports": ["LAX","SNA"]}' -> 'Running tool: compare_congestion (LAX, SNA)'."""
    try:
        parsed = json.loads(args) if isinstance(args, str) else dict(args or {})
    except json.JSONDecodeError:
        parsed = {}
    parts = []
    for key, value in parsed.items():
        if isinstance(value, list):
            parts.append(", ".join(str(v) for v in value))
        elif isinstance(value, dict):
            parts.append(", ".join(f"{k}={v}" for k, v in value.items()))
        elif key in ("airport", "region", "topic"):
            parts.append(str(value))
        else:
            parts.append(f"{key}={value}")
    return f"Running tool: {name}" + (f" ({'; '.join(parts)})" if parts else "")


def done_line(result, seconds: float, primary: str) -> str:
    """Status label after a turn; names the model(s) that answered, e.g. 'Done · 1 tool call · 2,682 tokens · 3s ·
    gpt-oss-120b (failover: gemini server error (HTTP 503), after 1 retry) · auto-verified (1 correction)'.
    The last part appears when a first draft was corrected by the automatic grounding retry. `result` is an
    agent TurnResult."""
    n = len(result.tool_calls)
    why = f": {result.fallback_reason}" if result.fallback_reason else ""
    names = [m.split("/")[-1] + (f" (failover{why})" if m != primary else "") for m in result.models]
    corrected = getattr(result, "retried", False) and not getattr(result, "grounding_issues", None)
    return (f"Done · {n} tool call{'s' if n != 1 else ''} · {result.total_tokens:,} tokens · {seconds:.0f}s"
            + (f" · {' + '.join(names)}" if names else "")
            + (" · auto-verified (1 correction)" if corrected else ""))


def wait_message(seconds_left: int) -> str:
    return f"Rate limit reached on the LLM provider - resuming in ~{seconds_left}s"


class StatusReporter:
    """Connects Agent events and LLM-client waits to a Streamlit status container."""

    def __init__(self, status, clock=time.monotonic, sleep=time.sleep):
        self.status = status
        self.clock, self._sleep = clock, sleep
        self.notice = None
        self.steps: list[str] = []

    def on_event(self, event: str, **kw) -> None:
        if event == "llm_call":
            label = COMPOSING if kw.get("after_tools") else ANALYZING
        elif event == "tool_start":
            label = tool_label(kw["name"], kw.get("args", {}))
        elif event == "retry":
            label = RETRYING
        elif event == "tool_done":
            if kw.get("error"):
                self.status.write(f"Tool {kw['name']} returned an error: {kw['error']}")
            return
        else:
            return
        self.steps.append(label)
        self.status.update(label=label)
        self.status.write(label)

    def sleep(self, seconds: float) -> None:
        """Countdown shown while waiting (pacing or HTTP 429 retry)."""
        if self.notice is None:
            self.notice = self.status.empty()
        end = self.clock() + seconds
        while (left := end - self.clock()) > 0:
            self.notice.info(wait_message(math.ceil(left)))
            self._sleep(min(1.0, left))
        self.notice.empty()
