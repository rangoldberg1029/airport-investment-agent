"""LLM client: OpenAI-compatible chat with tool calling, one primary and one fallback provider, rate-limit
pacing and HTTP 429 handling.

Providers (both free tiers, both reached through their OpenAI-compatible endpoints):
  primary  = Google Gemini Flash-Lite: GEMINI_API_KEY, model LLM_MODEL, base GEMINI_BASE_URL
  fallback = Groq: LLM_API_KEY, LLM_BASE_URL, model LLM_FALLBACK_MODEL (default openai/gpt-oss-120b)
Without GEMINI_API_KEY, Groq is the only provider.
"""

import json
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
# Gemini 3 validates "thought signatures" on function calls in the conversation. Calls it did not produce itself
# (the app's direct tool calls, or calls made by the Groq fallback) carry Google's documented placeholder.
GEMINI_PLACEHOLDER_SIGNATURE = "skip_thought_signature_validator"

FRIENDLY_RATE_LIMIT = ("The LLM provider's rate limit was reached (tokens per minute or per day). "
                       "Please wait a minute and ask again.")
QUOTA_MESSAGE = ("The language model's daily capacity has been reached. Key facts and Evidence still work via the "
                 "example questions' cached results; try again later or add your own API key.")


class RateLimited(Exception):
    """Raised when the provider still returns HTTP 429 after all retries."""


class DailyLimitReached(Exception):
    """A model hit its daily quota (Groq 'tokens per day (TPD)', Gemini '...PerDay...'): retrying today is pointless."""


class QuotaExhausted(Exception):
    """Every configured model (primary and fallback) has hit its daily quota in this session."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str
    extra: dict | None = None     # provider metadata to send back unchanged (Gemini thought signature)


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


def is_daily_limit(exc: Exception) -> bool:
    """HTTP 429 caused by a per-day quota (Groq: '... tokens per day (TPD) ...'; Gemini quota ids contain
    'PerDay')."""
    text = str(exc).lower()
    return is_rate_limit(exc) and any(k in text for k in ("tokens per day", "(tpd)", "requests per day",
                                                           "perday", "per day"))


def unavailable_reason(exc: Exception) -> str | None:
    """Why a provider could not answer, if switching to the fallback can help: rate limit, server error (5xx),
    timeout or connection failure. None for errors the fallback would hit too (e.g. 400 bad request)."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return "rate limit (HTTP 429)"
    if isinstance(status, int) and status >= 500:
        return f"server error (HTTP {status})"
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or name == "APITimeoutError":
        return "timeout"
    if isinstance(exc, ConnectionError) or name == "APIConnectionError":
        return "connection error"
    return None


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
            if is_daily_limit(exc):                 # waiting minutes/hours inside a chat turn is pointless
                raise DailyLimitReached(str(exc)) from exc
            if attempt == max_retries:
                break
            delay = retry_after_seconds(exc) or (2 ** (attempt + 1) + random.uniform(0, 1))
            if waited + delay > max_total_wait:
                break
            sleep(delay)
            waited += delay
    raise RateLimited(FRIENDLY_RATE_LIMIT)


def parse_duration(text) -> float | None:
    """Rate-limit reset header values -> seconds: '7.66s', '2m59.56s', '120ms', '1h2m3s', '30'."""
    if text is None:
        return None
    s = str(text).strip()
    try:
        return float(s)
    except ValueError:
        pass
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", s)
    if not parts:
        return None
    return sum(float(v) * {"ms": 0.001, "h": 3600.0, "m": 60.0, "s": 1.0}[unit] for v, unit in parts)


def _header(headers, key: str):
    try:
        return headers.get(key)
    except AttributeError:
        return None


def _number(headers, key: str) -> float | None:
    try:
        return float(_header(headers, key))
    except (TypeError, ValueError):
        return None


class TpmPacer:
    """Keeps one provider's calls under its per-minute limits.

    Preferred: the provider's own rate-limit response headers (Groq sends x-ratelimit-remaining-tokens /
    -requests and when they reset). They are exact and already leave out cached prompt tokens.
    Fallback when a provider sends no such headers (Gemini's OpenAI endpoint): a local 60-second window of
    the tokens and requests this client used.
    """

    def __init__(self, limit: int = 8000, window: float = 60.0, max_wait: float = 30.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 rpm_limit: int | None = None):
        self.limit, self.window, self.max_wait, self.rpm_limit = limit, window, max_wait, rpm_limit
        self.clock, self.sleep = clock, sleep
        self.events: deque[tuple[float, int]] = deque()
        self.total_waited = 0.0
        self.headers: dict | None = None        # latest provider-reported state (None = no headers seen)

    # -------------------------------------------------- local window
    def used(self) -> int:
        now = self.clock()
        while self.events and now - self.events[0][0] >= self.window:
            self.events.popleft()
        return sum(t for _, t in self.events)

    def record(self, tokens: int) -> None:
        self.events.append((self.clock(), tokens))

    def _local_wait(self, estimate: int) -> float:
        used, now = self.used(), self.clock()
        token_wait = 0.0
        if used + estimate > self.limit:
            if estimate > self.limit:
                token_wait = self.window
            else:
                token_wait, freed = self.window, 0
                for stamp, tokens in self.events:          # oldest first
                    freed += tokens
                    if used - freed + estimate <= self.limit:
                        token_wait = max(0.0, self.window - (now - stamp) + 0.5)
                        break
        request_wait = 0.0
        if self.rpm_limit and len(self.events) >= self.rpm_limit:
            stamp = self.events[len(self.events) - self.rpm_limit][0]     # this one must leave the window
            request_wait = max(0.0, self.window - (now - stamp) + 0.5)
        return max(token_wait, request_wait)

    # -------------------------------------------------- provider headers
    def observe(self, headers) -> None:
        """Store the provider's remaining tokens/requests and reset times from a response."""
        state = {"at": self.clock(),
                 "remaining_tokens": _number(headers, "x-ratelimit-remaining-tokens"),
                 "remaining_requests": _number(headers, "x-ratelimit-remaining-requests"),
                 "reset_tokens": parse_duration(_header(headers, "x-ratelimit-reset-tokens")),
                 "reset_requests": parse_duration(_header(headers, "x-ratelimit-reset-requests"))}
        if state["remaining_tokens"] is not None or state["remaining_requests"] is not None:
            self.headers = state

    def _header_wait(self, estimate: int) -> float:
        h = self.headers
        elapsed = self.clock() - h["at"]
        waits = [0.0]
        tokens, reset = h["remaining_tokens"], h["reset_tokens"]
        if tokens is not None and estimate > tokens and reset is not None and elapsed < reset:
            # Token budgets refill continuously: wait for the missing tokens, at most until the full reset.
            refill = (estimate - tokens) / self.limit * self.window if self.limit else reset
            waits.append(max(0.0, min(reset, refill) - elapsed))
        requests, reset_r = h["remaining_requests"], h["reset_requests"]
        if requests is not None and requests < 1 and reset_r is not None and elapsed < reset_r:
            waits.append(reset_r - elapsed)
        return max(waits)

    # -------------------------------------------------- public
    def wait_needed(self, estimate: int) -> float:
        """Seconds until a call of `estimate` tokens fits (0 = now). Does not sleep."""
        return self._header_wait(estimate) if self.headers else self._local_wait(estimate)

    def wait_for(self, estimate: int) -> float:
        """Sleep until `estimate` more tokens fit (capped at max_wait). Returns seconds waited."""
        waited = 0.0
        while waited < self.max_wait:
            delay = min(self.wait_needed(estimate), self.max_wait - waited)
            if delay <= 0.01:
                break
            self.sleep(delay)
            waited += delay
        self.total_waited += waited
        return waited


def estimate_tokens(messages: list[dict], tools: list[dict], max_output: int) -> int:
    """Rough estimate (4 characters per token) of a call's total tokens, for pacing only."""
    return len(json.dumps(messages)) // 4 + len(json.dumps(tools)) // 4 + max_output


def prepare_messages(messages: list[dict], gemini: bool) -> list[dict]:
    """Provider-specific message details. Gemini: every tool call carries a thought signature (its own, or the
    placeholder). Others: no Gemini metadata. Assistant tool-call messages without text get content None."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                c = dict(c)
                if gemini:
                    c.setdefault("extra_content", {"google": {"thought_signature": GEMINI_PLACEHOLDER_SIGNATURE}})
                else:
                    c.pop("extra_content", None)
                calls.append(c)
            m = {**m, "tool_calls": calls, "content": m.get("content") or None}
        out.append(m)
    return out


# ---------------------------------------------------------------- providers

@dataclass
class Provider:
    name: str                     # "gemini" | "groq" | ...
    model: str
    base_url: str
    client: Any                   # openai.OpenAI, or a stub in tests
    effort: str | None            # reasoning effort sent with each call (None = not sent)
    pacer: TpmPacer
    fixed_limit: bool = False     # True = TPM limit set explicitly; do not take it from response headers

    @property
    def is_gemini(self) -> bool:
        return "generativelanguage.googleapis.com" in self.base_url or self.model.startswith("gemini")


def _effort_for(model: str, effort: str | None) -> str | None:
    """Reasoning effort is sent only to models known to accept it (gpt-oss on Groq, Gemini)."""
    return effort if effort and ("gpt-oss" in model or model.startswith("gemini")) else None


def make_provider(name: str, model: str, base_url: str, api_key: str | None = None, effort: str | None = None,
                  tpm_limit: int = 8000, rpm_limit: int | None = None, fixed_limit: bool = False,
                  sleep: Callable[[float], None] = time.sleep, client: Any = None,
                  timeout_s: float = 60.0) -> Provider:
    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=api_key or "not-set", base_url=base_url, max_retries=0,   # retries handled here
                        timeout=timeout_s)
    return Provider(name, model, base_url, client, _effort_for(model, effort),
                    TpmPacer(limit=tpm_limit, rpm_limit=rpm_limit, sleep=sleep), fixed_limit)


def providers_from_env(sleep: Callable[[float], None] = time.sleep) -> tuple[Provider, Provider | None]:
    """(primary, fallback) from environment variables; see the module docstring."""
    groq_limit = os.getenv("LLM_FALLBACK_TPM_LIMIT")
    groq = dict(name="groq", model=os.getenv("LLM_FALLBACK_MODEL") or DEFAULT_GROQ_MODEL,
                base_url=os.getenv("LLM_BASE_URL") or GROQ_BASE_URL, api_key=os.getenv("LLM_API_KEY"),
                effort=os.getenv("LLM_FALLBACK_REASONING_EFFORT", "low"),
                tpm_limit=int(groq_limit or 8000), fixed_limit=bool(groq_limit), sleep=sleep)
    if not os.getenv("GEMINI_API_KEY"):
        return make_provider(**groq), None
    model = os.getenv("LLM_MODEL") or DEFAULT_GEMINI_MODEL
    if not model.startswith("gemini"):          # a Groq model name left over from the Groq-only setup
        model = DEFAULT_GEMINI_MODEL
    limit = os.getenv("LLM_TPM_LIMIT")
    primary = make_provider("gemini", model, os.getenv("GEMINI_BASE_URL") or GEMINI_BASE_URL,
                            os.getenv("GEMINI_API_KEY"), os.getenv("LLM_REASONING_EFFORT", "minimal"),
                            tpm_limit=int(limit or 250_000), rpm_limit=int(os.getenv("LLM_RPM_LIMIT", "15")),
                            fixed_limit=bool(limit), sleep=sleep,
                            # a hanging primary call switches to the fallback instead of blocking the turn
                            timeout_s=float(os.getenv("LLM_TIMEOUT_S", "20")))
    return primary, (make_provider(**groq) if os.getenv("LLM_API_KEY") else None)


# ---------------------------------------------------------------- client

class PrimaryUnavailable(Exception):
    """Internal: the primary provider failed in a way the fallback can help with (see unavailable_reason)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class OpenAIChat:
    """Chat client over a primary and an optional fallback provider.

    The fallback is used when the primary would have to wait more than `fallback_after_s` for its rate limit,
    or fails with HTTP 429, a 5xx server error, a timeout or a connection error. At most one switch per turn:
    once switched, the rest of the turn stays on the fallback (`start_turn` resets this; `switch_reason`
    says why it happened, for the UI). Daily quota: a model that returns a per-day 429 is marked exhausted for
    the rest of this client's life (one client per UI session); when every model is exhausted, QuotaExhausted
    is raised.
    """

    def __init__(self, primary: Provider, fallback: Provider | None = None, sleep: Callable[[float], None] = time.sleep,
                 fallback_after_s: float = 5.0, max_output_tokens: int | None = None, temperature: float = 0.2):
        self.primary = primary
        self.fallback = fallback if fallback and fallback.model != primary.model else None
        self.providers = [p for p in (self.primary, self.fallback) if p]
        self.fallback_after_s = fallback_after_s
        self.sleep = sleep
        self.set_sleep(sleep)
        self.exhausted: set[str] = set()               # models that hit their daily quota in this session
        self.switch_reason: str | None = None          # why this turn switched to the fallback (None = it did not)
        # Cap per LLM call, including hidden reasoning tokens. Answers in the evals were <= ~600 tokens.
        self.max_output_tokens = max_output_tokens or int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1000"))
        self.temperature = temperature

    @classmethod
    def from_env(cls, use_fallback: bool = True, **kwargs) -> "OpenAIChat":
        primary, fallback = providers_from_env()
        return cls(primary, fallback if use_fallback else None, **kwargs)

    # -------------------------------------------------- what the UI and scripts read
    @property
    def model(self) -> str:
        return self.primary.model

    @property
    def fallback_model(self) -> str | None:
        return self.fallback.model if self.fallback else None

    @property
    def reasoning_effort(self) -> str | None:
        return self.primary.effort

    @property
    def pacer(self) -> TpmPacer:
        return self.primary.pacer

    def set_sleep(self, sleep: Callable[[float], None]) -> None:
        """Replace how waits happen (pacing and 429 retries), e.g. a UI countdown."""
        self.sleep = sleep
        for p in self.providers:
            p.pacer.sleep = sleep

    # -------------------------------------------------- routing
    def start_turn(self) -> None:
        """A new user turn starts on the primary again."""
        self.switch_reason = None

    def available(self) -> list[Provider]:
        return [p for p in self.providers if p.model not in self.exhausted]

    def choose(self, estimate: int) -> Provider:
        """Primary unless it is out of daily quota, this turn already switched, or it would wait longer than
        fallback_after_s for its rate limit (each only when a usable fallback exists)."""
        available = self.available()
        if not available:
            raise QuotaExhausted(QUOTA_MESSAGE)
        if self.primary not in available:
            return available[0]
        if self.fallback in available:
            if self.switch_reason:
                return self.fallback
            wait = self.primary.pacer.wait_needed(estimate)
            if wait > self.fallback_after_s:
                self.switch_reason = f"primary rate limit (would wait {wait:.0f} s)"
                return self.fallback
        return self.primary

    def complete(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> LLMResponse:
        estimate = estimate_tokens(messages, tools, self.max_output_tokens // 2)
        provider = self.choose(estimate)
        try:
            return self._call(provider, messages, tools, tool_choice, estimate)
        except DailyLimitReached:
            self.exhausted.add(provider.model)
            return self.complete(messages, tools, tool_choice)      # next provider, or QuotaExhausted
        except PrimaryUnavailable as exc:
            self.switch_reason = exc.reason
            try:
                return self._call(self.fallback, messages, tools, tool_choice, estimate)
            except DailyLimitReached:
                self.exhausted.add(self.fallback.model)
                return self.complete(messages, tools, tool_choice)

    # -------------------------------------------------- one HTTP call
    def _kwargs(self, provider: Provider, messages: list[dict], tools: list[dict], tool_choice: str) -> dict:
        kwargs: dict = dict(model=provider.model, messages=prepare_messages(messages, provider.is_gemini))
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice)
        if provider.is_gemini:
            # Gemini 3: Google advises keeping the default temperature (1.0); lower values can cause loops.
            kwargs["max_tokens"] = self.max_output_tokens
        else:
            kwargs.update(temperature=self.temperature, max_completion_tokens=self.max_output_tokens)
        if provider.effort:
            kwargs["reasoning_effort"] = provider.effort
        return kwargs

    def _request(self, provider: Provider, kwargs: dict):
        create = provider.client.chat.completions.with_raw_response.create
        try:
            return create(**kwargs)
        except Exception as exc:
            # A model that rejects the reasoning level: stop sending it and try once more without it.
            if ("reasoning_effort" in kwargs and getattr(exc, "status_code", None) == 400
                    and re.search(r"reasoning|thinking", str(exc), re.I)):
                provider.effort = None
                kwargs.pop("reasoning_effort")
                return create(**kwargs)
            raise

    def _call(self, provider: Provider, messages: list[dict], tools: list[dict], tool_choice: str,
              estimate: int) -> LLMResponse:
        provider.pacer.wait_for(estimate)
        kwargs = self._kwargs(provider, messages, tools, tool_choice)
        request = lambda: self._request(provider, kwargs)  # noqa: E731
        if provider is self.primary and self.fallback and self.fallback.model not in self.exhausted:
            try:                     # primary with a usable fallback: no retry, switch on 429 / 5xx / timeout
                raw = request()
            except Exception as exc:
                if is_daily_limit(exc):
                    self.exhausted.add(provider.model)   # out for today: fallback for the rest of the session
                reason = unavailable_reason(exc)
                if reason:
                    raise PrimaryUnavailable(f"{provider.name} {reason}") from exc
                raise
        else:
            raw = with_retry(request, sleep=self.sleep)
        self._observe(provider, raw.headers)
        resp = raw.parse()
        usage = resp.usage
        total = usage.total_tokens if usage else 0
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        # Groq: cached prompt tokens do not count toward the rate limits.
        provider.pacer.record(total if provider.is_gemini else max(0, total - cached))
        msg = resp.choices[0].message
        calls = [ToolCall(c.id, c.function.name, c.function.arguments or "{}",
                          (getattr(c, "model_extra", None) or {}).get("extra_content"))
                 for c in (msg.tool_calls or [])]
        return LLMResponse(msg.content or "", calls, total,
                           usage.prompt_tokens if usage else 0,
                           usage.completion_tokens if usage else 0,
                           model=provider.model)

    @staticmethod
    def _observe(provider: Provider, headers) -> None:
        """Take the real per-model TPM limit and the remaining budget from the response headers, if sent."""
        if not provider.fixed_limit:
            limit = _number(headers, "x-ratelimit-limit-tokens")
            if limit:
                provider.pacer.limit = int(limit)
        provider.pacer.observe(headers)
