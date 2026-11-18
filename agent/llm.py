"""LLM client: OpenAI-compatible chat with tool calling, 429 retry with backoff, and TPM pacing.

Provider-agnostic: Groq (default), Google AI Studio or Ollama via LLM_BASE_URL / LLM_MODEL / LLM_API_KEY.
"""

import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Protocol

FRIENDLY_RATE_LIMIT = ("The LLM provider's rate limit was reached (tokens per minute or per day). "
                       "Please wait a minute and ask again.")


class RateLimited(Exception):
    """Raised when the provider still returns HTTP 429 after all retries."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""               # which model produced this response (primary or fallback)


class LLMClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> LLMResponse: ...


# ---------------------------------------------------------------- retry + pacing (pure, tested)

def is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429


def retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    try:
        return float(response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def with_retry(fn: Callable[[], LLMResponse], max_retries: int = 3, max_total_wait: float = 30.0,
               sleep: Callable[[float], None] = time.sleep) -> LLMResponse:
    """Call fn; on HTTP 429 wait retry-after (or 2/4/8 s + jitter) and retry. Raises RateLimited."""
    waited = 0.0
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if not is_rate_limit(exc):
                raise
            if attempt == max_retries:
                break
            delay = retry_after_seconds(exc) or (2 ** (attempt + 1) + random.uniform(0, 1))
            if waited + delay > max_total_wait:
                break
            sleep(delay)
            waited += delay
    raise RateLimited(FRIENDLY_RATE_LIMIT)


class TpmPacer:
    """Sliding 60-second window of tokens used; waits before a call that would exceed the limit."""

    def __init__(self, limit: int = 8000, window: float = 60.0, max_wait: float = 30.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.limit, self.window, self.max_wait = limit, window, max_wait
        self.clock, self.sleep = clock, sleep
        self.events: deque[tuple[float, int]] = deque()
        self.total_waited = 0.0

    def used(self) -> int:
        now = self.clock()
        while self.events and now - self.events[0][0] >= self.window:
            self.events.popleft()
        return sum(t for _, t in self.events)

    def wait_for(self, estimate: int) -> float:
        """Sleep until `estimate` more tokens fit in the window (capped at max_wait). Returns seconds waited."""
        waited = 0.0
        while self.events and self.used() + estimate > self.limit and waited < self.max_wait:
            delay = min(self.window - (self.clock() - self.events[0][0]) + 0.5, self.max_wait - waited)
            if delay <= 0:
                break
            self.sleep(delay)
            waited += delay
        self.total_waited += waited
        return waited

    def wait_needed(self, estimate: int) -> float:
        """Seconds until `estimate` more tokens fit in the window (0 = now). Does not sleep."""
        used = self.used()
        if used + estimate <= self.limit:
            return 0.0
        if estimate > self.limit:
            return self.window
        now, freed = self.clock(), 0
        for stamp, tokens in self.events:          # oldest first
            freed += tokens
            if used - freed + estimate <= self.limit:
                return max(0.0, self.window - (now - stamp) + 0.5)
        return self.window

    def record(self, tokens: int) -> None:
        self.events.append((self.clock(), tokens))


def estimate_tokens(messages: list[dict], tools: list[dict], max_output: int) -> int:
    """Rough estimate (4 characters per token) of a call's total tokens, for pacing only."""
    return len(json.dumps(messages)) // 4 + len(json.dumps(tools)) // 4 + max_output


# ---------------------------------------------------------------- OpenAI-compatible client

class PrimaryRateLimited(Exception):
    """Internal: the primary model returned HTTP 429 and a fallback model is configured."""


class OpenAIChat:
    """OpenAI-compatible chat client with per-model pacing and an optional fallback model.

    Fallback (LLM_FALLBACK_MODEL, e.g. openai/gpt-oss-20b, separate provider rate limit) is used when the primary
    model would have to wait more than `fallback_after_s` for its token budget, or when it returns HTTP 429.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None,
                 reasoning_effort: str | None = None, tpm_limit: int | None = None,
                 max_output_tokens: int | None = None, temperature: float = 0.2,
                 sleep: Callable[[float], None] = time.sleep, fallback_model: str | None = None,
                 fallback_after_s: float = 5.0):
        from openai import OpenAI

        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        fallback = fallback_model if fallback_model is not None else os.getenv("LLM_FALLBACK_MODEL", "")
        self.fallback_model = fallback if fallback and fallback != self.model else None
        self.fallback_after_s = fallback_after_s
        self.client = OpenAI(api_key=api_key or os.getenv("LLM_API_KEY"),
                             base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                             max_retries=0)   # we handle 429 ourselves
        self.effort = reasoning_effort or os.getenv("LLM_REASONING_EFFORT", "low")
        self.reasoning_effort = self._effort_for(self.model)
        self.sleep = sleep
        limit = tpm_limit or int(os.getenv("LLM_TPM_LIMIT", "8000"))
        self.pacers = {m: TpmPacer(limit=limit, sleep=sleep) for m in (self.model, self.fallback_model) if m}
        self.pacer = self.pacers[self.model]           # primary model's pacer
        # Cap per LLM call. For gpt-oss it includes hidden reasoning tokens: answers in the eval were
        # <= ~600 tokens, reasoning at effort "low" ~100-300, so 1,000 leaves headroom.
        self.max_output_tokens = max_output_tokens or int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1000"))
        self.temperature = temperature

    def _effort_for(self, model: str) -> str | None:
        return self.effort if "gpt-oss" in model else None

    def set_sleep(self, sleep: Callable[[float], None]) -> None:
        """Replace how waits happen (pacing and 429 retries), e.g. a UI countdown."""
        self.sleep = sleep
        for pacer in self.pacers.values():
            pacer.sleep = sleep

    def _update_limit(self, headers, pacer: TpmPacer | None = None) -> None:
        """Pace to the provider's real per-model limit (Groq: x-ratelimit-limit-tokens), unless set explicitly."""
        if os.getenv("LLM_TPM_LIMIT"):
            return
        try:
            (pacer or self.pacer).limit = int(headers.get("x-ratelimit-limit-tokens"))
        except (TypeError, ValueError, AttributeError):
            pass

    def choose_model(self, estimate: int) -> str:
        """Primary unless it would wait longer than fallback_after_s and a fallback exists."""
        if self.fallback_model and self.pacer.wait_needed(estimate) > self.fallback_after_s:
            return self.fallback_model
        return self.model

    def complete(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> LLMResponse:
        estimate = estimate_tokens(messages, tools, self.max_output_tokens // 2)
        model = self.choose_model(estimate)
        try:
            return self._call(model, messages, tools, tool_choice, estimate)
        except PrimaryRateLimited:
            return self._call(self.fallback_model, messages, tools, tool_choice, estimate)

    def _call(self, model: str, messages: list[dict], tools: list[dict], tool_choice: str, estimate: int) -> LLMResponse:
        pacer = self.pacers[model]
        pacer.wait_for(estimate)
        kwargs = dict(model=model, messages=messages, tools=tools, tool_choice=tool_choice,
                      temperature=self.temperature, max_completion_tokens=self.max_output_tokens)
        effort = self._effort_for(model)
        if effort:
            kwargs["reasoning_effort"] = effort
        request = lambda: self.client.chat.completions.with_raw_response.create(**kwargs)  # noqa: E731
        if model == self.model and self.fallback_model:
            try:                                   # primary with a fallback: no retry, switch on 429
                raw = request()
            except Exception as exc:
                if is_rate_limit(exc):
                    raise PrimaryRateLimited() from exc
                raise
        else:
            raw = with_retry(request, sleep=self.sleep)
        self._update_limit(raw.headers, pacer)
        resp = raw.parse()
        usage = resp.usage
        pacer.record(usage.total_tokens if usage else 0)
        msg = resp.choices[0].message
        calls = [ToolCall(c.id, c.function.name, c.function.arguments or "{}") for c in (msg.tool_calls or [])]
        return LLMResponse(msg.content or "", calls,
                           usage.total_tokens if usage else 0,
                           usage.prompt_tokens if usage else 0,
                           usage.completion_tokens if usage else 0,
                           model=model)
