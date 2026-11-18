"""Agent loop: the LLM picks tools, tools compute, the LLM explains. No framework.

Memory: the last MAX_TEXT_TURNS user/assistant exchanges are resent as text. The most recent completed turn
keeps its full answer and its tool-call/tool-result messages; older answers are truncated to OLD_ANSWER_CHARS
and their tool messages are dropped (token budget).
All tool records are kept in the Agent for the UI and for the number check.
"""

import json
from dataclasses import dataclass, field

from agent.guardrails import untraceable_numbers
from agent.llm import LLMClient, RateLimited
from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOL_SCHEMAS, ToolCallRecord, execute_tool
from data.repository import Frames

MAX_TOOL_ROUNDS = 4
MAX_TEXT_TURNS = 6
OLD_ANSWER_CHARS = 300      # answers older than the latest completed turn are truncated (token budget)


@dataclass
class Turn:
    user: str
    steps: list[dict] = field(default_factory=list)     # assistant tool-call + tool-result messages
    answer: str = ""
    records: list[ToolCallRecord] = field(default_factory=list)


@dataclass
class TurnResult:
    text: str
    tool_calls: list[ToolCallRecord]
    untraceable_numbers: list[dict]
    total_tokens: int = 0
    llm_calls: int = 0
    error: str | None = None


class Agent:
    def __init__(self, client: LLMClient, frames: Frames, max_tool_rounds: int = MAX_TOOL_ROUNDS,
                 tool_overrides: dict | None = None):
        self.client, self.frames = client, frames
        self.max_tool_rounds = max_tool_rounds
        self.tool_overrides = tool_overrides or {}
        self.turns: list[Turn] = []

    # ---------------------------------------------------------------- memory
    def messages(self, current: Turn) -> list[dict]:
        """System prompt + trimmed history + the current turn."""
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        history = self.turns[-MAX_TEXT_TURNS:]
        for i, t in enumerate(history):
            latest = i == len(history) - 1
            msgs.append({"role": "user", "content": t.user})
            if latest:                           # latest completed turn keeps its tool messages
                msgs.extend(t.steps)
            msgs.append({"role": "assistant", "content": t.answer if latest else truncate(t.answer)})
        msgs.append({"role": "user", "content": current.user})
        msgs.extend(current.steps)
        return msgs

    def reset(self) -> None:
        self.turns = []

    # ---------------------------------------------------------------- loop
    def ask(self, text: str) -> TurnResult:
        turn = Turn(user=text)
        tokens, calls = 0, 0
        try:
            answer = None
            for _ in range(self.max_tool_rounds):
                resp = self.client.complete(self.messages(turn), TOOL_SCHEMAS)
                tokens, calls = tokens + resp.total_tokens, calls + 1
                if not resp.tool_calls:
                    answer = resp.content
                    break
                self._run_tools(turn, resp)
            if answer is None:                     # rounds exhausted: force a final answer
                resp = self.client.complete(self.messages(turn), TOOL_SCHEMAS, tool_choice="none")
                tokens, calls, answer = tokens + resp.total_tokens, calls + 1, resp.content
        except RateLimited as exc:
            return TurnResult(str(exc), turn.records, [], tokens, calls, error="rate_limited")
        except Exception as exc:  # provider/network errors: do not crash the UI
            return TurnResult(f"Sorry, the language model call failed ({type(exc).__name__}). Please try again.",
                              turn.records, [], tokens, calls, error=str(exc))

        turn.answer = answer.strip() or "I could not produce an answer. Please rephrase the question."
        self.turns.append(turn)
        everything = [r.envelope for t in self.turns for r in t.records]
        check = untraceable_numbers(turn.answer, everything, user_text=text)
        return TurnResult(turn.answer, turn.records, check, tokens, calls)

    def _run_tools(self, turn: Turn, resp) -> None:
        turn.steps.append({
            "role": "assistant", "content": resp.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments}} for c in resp.tool_calls],
        })
        for call in resp.tool_calls:
            record = execute_tool(call.name, call.arguments, self.frames, self.tool_overrides)
            turn.records.append(record)
            turn.steps.append({"role": "tool", "tool_call_id": call.id, "content": record.llm_view})


def truncate(text: str, limit: int = OLD_ANSWER_CHARS) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def summarize_calls(records: list[ToolCallRecord]) -> list[str]:
    return [f"{r.name}({json.dumps(r.args, ensure_ascii=False)})" + (f" -> ERROR {r.error}" if r.error else "")
            for r in records]
