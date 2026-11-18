"""Agent loop: the LLM picks tools, tools compute, the LLM explains. No framework.

One turn (typed question):
  1. LLM call with the tools (tool_choice "auto"; "required" on a retry). If it answers in text (no tool), done.
  2. The requested tools run (several in parallel are fine) -> event "facts_ready": the UI shows the computed
     Answer line, Key facts and Evidence at once.
  3. One more LLM call (tool_choice "none"), streamed token by token (event "token"), writes the
     "Why / drivers" bullets.
Direct path (example cards, suggested follow-ups): the tool and its arguments are known, so the tool runs first
(start_direct) and only step 3 happens (explain). The tool call is recorded in the conversation exactly like
one the LLM chose, and counts as this turn's tool call for the grounding check.

Answer = deterministic headline + LLM explanation: the "Answer" line is built by code from the tool outputs
(agent/headlines.py); the LLM writes only the bullets. The number and grounding checks run on the LLM part when
the stream ends; if the grounding check fails, the turn is retried once (event "retry": the UI replaces the
bullets) and any issues that remain are returned to the UI.

Memory: the last MAX_TEXT_TURNS user/assistant exchanges are resent as text. Earlier turns' tool outputs are
NOT resent: each earlier answer carries a one-line summary of its tool calls (token budget); answers older than
the latest are truncated to OLD_ANSWER_CHARS. All tool records stay in the Agent for the UI, the follow-up
context and the number check.

Follow-up context: the last successful tool call (name + arguments) is added to every new question, so
"rank them again", "what about X" or "why is X lower" reuse the same region / airport set. A follow-up-style
question with no earlier result gets a note to ask a clarifying question instead.
"""

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Callable

from agent.guardrails import grounding_issues, untraceable_numbers
from agent.headlines import NATIONAL_NOTE, answer_headline, compose, why_bullets
from agent.llm import LLMClient, QuotaExhausted, RateLimited
from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOL_SCHEMAS, TOOLS, ToolCallRecord, execute_tool
from data.repository import Frames

MAX_TEXT_TURNS = 6
OLD_ANSWER_CHARS = 300      # answers older than the latest completed turn are truncated (token budget)
# Words that make a question refer to an earlier result ("rank them again", "which is third now?").
FOLLOWUP_RE = re.compile(r"\b(them|they|those|these|again|now|instead|third|second|fourth|fifth|previous|"
                         r"same|that one|the list|what about)\b", re.I)
NO_CONTEXT_NOTE = ("[Context: there is no previous result in this conversation. If the question refers to an "
                   "earlier result (e.g. 'them', 'now', 'third'), ask one short clarifying question.]")


@dataclass
class Turn:
    user: str
    steps: list[dict] = field(default_factory=list)     # assistant tool-call + tool-result messages
    answer: str = ""
    records: list[ToolCallRecord] = field(default_factory=list)
    note: str = ""          # extra context appended to the user message for the LLM (never shown in history)


@dataclass
class TurnResult:
    text: str
    tool_calls: list[ToolCallRecord]
    untraceable_numbers: list[dict]
    total_tokens: int = 0
    llm_calls: int = 0
    error: str | None = None
    models: list[str] = field(default_factory=list)      # models that answered, in order (primary/fallback)
    grounding_issues: list[str] = field(default_factory=list)   # non-empty = answer failed the grounding check
    retried: bool = False                                 # True = first draft failed the check, turn was retried
    headline: str = ""          # deterministic Answer line (from tool outputs); empty if no tool succeeded
    explanation: str = ""       # the LLM's part (Why / drivers bullets, or its whole answer if no headline)
    fallback_reason: str | None = None   # why this turn switched to the fallback model (None = it did not)


def _noop(event: str, **kw) -> None:
    pass


class Agent:
    def __init__(self, client: LLMClient, frames: Frames, tool_overrides: dict | None = None):
        self.client, self.frames = client, frames
        self.tool_overrides = tool_overrides or {}
        self.turns: list[Turn] = []

    # ---------------------------------------------------------------- memory
    def messages(self, current: Turn) -> list[dict]:
        """System prompt + earlier turns as text (one-line tool summaries, no tool outputs) + the current turn
        with its tool calls and results."""
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        history = self.turns[-MAX_TEXT_TURNS:]
        for i, t in enumerate(history):
            latest = i == len(history) - 1
            msgs.append({"role": "user", "content": t.user})
            summary = tool_summary(t.records)
            answer = t.answer if latest else truncate(t.answer)
            msgs.append({"role": "assistant", "content": f"{summary}\n{answer}" if summary else answer})
        content = current.user + (f"\n\n{current.note}" if current.note else "")
        msgs.append({"role": "user", "content": content})
        msgs.extend(current.steps)
        return msgs

    def last_tool_call(self) -> ToolCallRecord | None:
        """Most recent successful tool call in this conversation (for follow-up questions)."""
        for t in reversed(self.turns):
            for r in reversed(t.records):
                if not r.error:
                    return r
        return None

    def followup_context(self, text: str = "") -> str:
        last = self.last_tool_call()
        if last is None:
            return NO_CONTEXT_NOTE if FOLLOWUP_RE.search(text or "") else ""
        return (f"[Context: previous tool call in this conversation: {last.name}"
                f"({json.dumps(last.args, ensure_ascii=False)}). If this question refers to it (e.g. 'them', "
                f"'again', 'what about', 'why'), call a tool with the same region/airports and change only what "
                f"the question asks. Any new ranking, comparison or weight change requires a tool call.]")

    def reset(self) -> None:
        self.turns = []

    def restore_turn(self, turn: Turn) -> None:
        """Add a previously computed turn (e.g. a cached first answer) so follow-ups have its context."""
        self.turns.append(copy.deepcopy(turn))

    def headline(self, turn: Turn) -> str:
        """The computed Answer line for a turn (with the 'no previous ranking' note when it applies)."""
        notes = []
        earlier_rank = any(r.name == "rank_airports" and not r.error for t in self.turns for r in t.records)
        for r in turn.records:
            if (r.name == "rank_airports" and not r.error and not earlier_rank
                    and not any(r.args.get(k) for k in ("region", "states", "airports"))):
                notes.append(NATIONAL_NOTE)
                break
        return answer_headline(turn.records, turn.user, notes)

    # ---------------------------------------------------------------- typed questions
    def ask(self, text: str, on_event: Callable[..., None] | None = None) -> TurnResult:
        """Answer one user message. `on_event(event, **details)` reports progress to a UI:
        llm_call(round, after_tools), tool_start(name, args), tool_done(name, error),
        facts_ready(records, headline), token(text), retry(issues)."""
        emit = on_event or _noop
        self._start_turn()
        context = self.followup_context(text)
        turn, result = self._attempt(Turn(user=text, note=context), emit, first_choice="auto")
        if result.grounding_issues and not result.error:
            emit("retry", issues=result.grounding_issues)
            note = (context + "\n" if context else "") + (
                "[Your previous draft was rejected by an automatic check: " + "; ".join(result.grounding_issues)
                + ". Call the appropriate tool now and answer only from its output.]")
            first = result
            turn, result = self._attempt(Turn(user=text, note=note), emit, first_choice="required")
            _merge_retry(first, result)
        if not result.error:
            self.turns.append(turn)
        result.fallback_reason = getattr(self.client, "switch_reason", None)
        return result

    def _start_turn(self) -> None:
        """Tell the client a new turn starts (it may switch to its fallback at most once per turn)."""
        if hasattr(self.client, "start_turn"):
            self.client.start_turn()

    # ---------------------------------------------------------------- direct path (known tool + arguments)
    def start_direct(self, text: str, tool: str, args: dict,
                     on_event: Callable[..., None] | None = None) -> Turn:
        """Run a known tool call without the LLM. Returns the pending turn (records ready to render)."""
        emit = on_event or _noop
        emit("tool_start", name=tool, args=args)
        record = execute_tool(tool, args, self.frames, self.tool_overrides)
        emit("tool_done", name=tool, error=record.error)
        call_id = f"direct_{len(self.turns) + 1}"
        turn = Turn(user=text, records=[record])
        turn.steps = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": tool, "arguments": json.dumps(args, ensure_ascii=False)}}]},
            {"role": "tool", "tool_call_id": call_id, "content": record.llm_view},
        ]
        return turn

    def explain(self, turn: Turn, on_event: Callable[..., None] | None = None) -> TurnResult:
        """One streamed LLM call (tool_choice="none") writes the explanation for a turn whose tool already ran."""
        emit = on_event or _noop
        self._start_turn()
        pending = copy.deepcopy(turn)
        done, result = self._attempt(turn, emit, first_choice="none")
        if result.grounding_issues and not result.error:
            emit("retry", issues=result.grounding_issues)
            retry = copy.deepcopy(pending)
            retry.note = ("[Your previous draft was rejected by an automatic check: "
                          + "; ".join(result.grounding_issues) + ". Answer only from the tool output above.]")
            first = result
            done, result = self._attempt(retry, emit, first_choice="none")
            _merge_retry(first, result)
        if result.error:
            # The tool call is real and deterministic: keep it in the conversation even if the LLM failed.
            done = pending
            done.answer = compose(self.headline(done), "")
        self.turns.append(done)
        result.fallback_reason = getattr(self.client, "switch_reason", None)
        return result

    def ask_direct(self, text: str, tool: str, args: dict,
                   on_event: Callable[..., None] | None = None) -> TurnResult:
        return self.explain(self.start_direct(text, tool, args, on_event), on_event)

    # ---------------------------------------------------------------- one attempt
    def _attempt(self, turn: Turn, emit: Callable[..., None], first_choice: str) -> tuple[Turn, TurnResult]:
        """One attempt at a turn. Not added to history here (a rejected draft must not be remembered).
        A turn that already has tool records (direct path) goes straight to the streamed explanation."""
        tokens, calls, models = 0, 0, []

        def track(resp) -> None:
            nonlocal tokens, calls
            tokens, calls = tokens + resp.total_tokens, calls + 1
            if getattr(resp, "model", "") and resp.model not in models:
                models.append(resp.model)
        try:
            answer = None
            if not turn.records:
                emit("llm_call", round=0, after_tools=False)
                resp = self.client.complete(self.messages(turn), TOOL_SCHEMAS, tool_choice=first_choice)
                track(resp)
                if resp.tool_calls:
                    self._run_tools(turn, resp, emit)
                    emit("facts_ready", records=list(turn.records), headline=self.headline(turn))
                else:
                    answer = resp.content
            if answer is None:                     # explain the tool results, streamed
                emit("llm_call", round=1, after_tools=True)
                resp = self.client.complete(self.messages(turn), TOOL_SCHEMAS, tool_choice="none",
                                            on_token=lambda piece: emit("token", text=piece))
                track(resp)
                answer = resp.content
                if getattr(resp, "finish_reason", "stop") != "stop":
                    answer = drop_cut_line(answer)       # never show a bullet that stops mid-sentence
        except Exception as exc:  # quota, rate limit, provider/network errors: do not crash the UI
            if isinstance(exc, QuotaExhausted):
                message, error = str(exc), "quota_exhausted"
            elif isinstance(exc, RateLimited):
                message, error = str(exc), "rate_limited"
            else:
                message, error = f"Sorry, the language model call failed ({type(exc).__name__}). Please try again.", str(exc)
            headline = self.headline(turn)
            return turn, TurnResult(compose(headline, message, why_heading=False), turn.records, [], tokens, calls,
                                    error=error, models=models, headline=headline, explanation=message)

        llm_text = answer.strip() or "I could not produce an answer. Please rephrase the question."
        headline = self.headline(turn)
        explanation = why_bullets(llm_text) if headline else llm_text
        turn.answer = compose(headline, explanation)
        # The headline is computed by code; the checks cover what the LLM wrote. Both what the scoring layer
        # returned (envelope) and what the LLM saw (llm_view: precomputed differences and ranked drivers live
        # only there) count as traceable sources.
        outputs = [x for t in [*self.turns, turn] for r in t.records for x in (r.envelope, r.llm_view)]
        numbers = untraceable_numbers(explanation, outputs, user_text=turn.user)
        grounded_text = json.dumps(outputs, ensure_ascii=False) + " " + " ".join(t.user for t in [*self.turns, turn])
        issues = grounding_issues(explanation, turn.user, [r.name for r in turn.records], list(TOOLS),
                                  grounded_text, set(self.frames.airports["iata"]))
        return turn, TurnResult(turn.answer, turn.records, numbers, tokens, calls, models=models,
                                grounding_issues=issues, headline=headline, explanation=explanation)

    def _run_tools(self, turn: Turn, resp, emit: Callable[..., None]) -> None:
        turn.steps.append({
            "role": "assistant", "content": resp.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                            **({"extra_content": c.extra} if getattr(c, "extra", None) else {})}
                           for c in resp.tool_calls],
        })
        for call in resp.tool_calls:
            emit("tool_start", name=call.name, args=call.arguments)
            record = execute_tool(call.name, call.arguments, self.frames, self.tool_overrides)
            emit("tool_done", name=call.name, error=record.error)
            turn.records.append(record)
            turn.steps.append({"role": "tool", "tool_call_id": call.id, "content": record.llm_view})


def _merge_retry(first: TurnResult, result: TurnResult) -> None:
    """Fold the rejected first attempt's usage into the retry's result."""
    result.retried = True
    result.total_tokens += first.total_tokens
    result.llm_calls += first.llm_calls
    result.models = list(dict.fromkeys(first.models + result.models))


def tool_summary(records: list[ToolCallRecord]) -> str:
    """One line per earlier turn instead of its tool outputs: '[Tools used: rank_airports({"region": ...})]'."""
    if not records:
        return ""
    return "[Tools used: " + "; ".join(summarize_calls(records)) + "]"


def drop_cut_line(text: str) -> str:
    """Remove a last line that was cut off (the answer hit the output limit or the stream ended early): a line is
    complete when it ends with sentence punctuation, a closing bracket or a unit sign."""
    lines = text.rstrip().splitlines()
    if lines and not re.search(r"[.!?)\]%:;\"'”]$", lines[-1].strip()):
        lines = lines[:-1]
    return "\n".join(lines)


def truncate(text: str, limit: int = OLD_ANSWER_CHARS) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def summarize_calls(records: list[ToolCallRecord]) -> list[str]:
    return [f"{r.name}({json.dumps(r.args, ensure_ascii=False)})" + (f" -> ERROR {r.error}" if r.error else "")
            for r in records]
