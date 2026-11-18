"""LLM client: OpenAI-compatible chat with tool calling over a primary and an optional failover provider, with
rate-limit pacing, retries and failover.

Configuration (environment or Streamlit secrets; everything else is a constant below):
  GEMINI_API_KEY  required  primary model: Google Gemini (free tier)
  GEMINI_MODEL    optional  default gemini-3.5-flash-lite
  GROQ_API_KEY    optional  enables failover to Groq (and voice input, see agent/speech.py)
  GROQ_MODEL      optional  default openai/gpt-oss-120b
Only GROQ_API_KEY set: Groq is the only provider.

Error policy (per LLM call):
  Gemini 429 (per-minute or daily limit)        -> Groq immediately
  Gemini 5xx, timeout or connection error       -> retry Gemini once after SERVER_RETRY_DELAY_S -> then Groq
  Gemini would wait > FALLBACK_AFTER_S for its rate limit                  -> Groq
Timeout = no first token (non-streamed call: no response) within GEMINI_FIRST_TOKEN_TIMEOUT_S; a stream that
has started is never cut by time.
Once a turn has switched to Groq it stays there for the rest of that turn. Without Groq, 429s are retried
with backoff and a 5xx/timeout is retried once; then the turn shows an error.
"""

import itertools
import json
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_REASONING = "minimal"        # lowest Gemini 3 thinking level
GROQ_REASONING = "low"              # gpt-oss reasoning effort
GEMINI_TPM, GEMINI_RPM = 250_000, 15     # free-tier limits, paced locally (Gemini sends no rate-limit headers)
GROQ_TPM = 8_000                    # replaced by Groq's x-ratelimit-limit-tokens header after the first call
# Gemini: a call must deliver its FIRST token (non-streamed: its response) within 8 s, else it counts as a timeout
# (one retry, then Groq). Once streaming has started there is no total limit; HTTP_TIMEOUT_S only bounds a
# stalled connection (connect / gap between chunks).
GEMINI_FIRST_TOKEN_TIMEOUT_S = 8.0
HTTP_TIMEOUT_S = 60.0
SERVER_RETRY_DELAY_S = 2.0          # wait before the one retry after a 5xx / timeout
FALLBACK_AFTER_S = 5.0              # switch instead of waiting longer than this for the primary's rate limit
MAX_OUTPUT_TOKENS = 1000            # per call, incl. hidden reasoning tokens (eval answers <= ~600)
GEMINI_MAX_OUTPUT_TOKENS = 2000     # Gemini counts its thinking in the output limit: more room, no cut-offs
# Optional request parameters dropped (and the call repeated once) if a provider rejects them with HTTP 400.
OPTIONAL_PARAMS = {"reasoning_effort": r"reasoning|thinking", "stream_options": r"stream_options|include_usage"}
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
    finish_reason: str = "stop"   # "length" = cut off by the output limit; "" = a stream ended without a reason


class LLMClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto",
                 on_token: Callable[[str], None] | None = None) -> LLMResponse: ...


# ---------------------------------------------------------------- retry + pacing (pure, tested)

def is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429


def is_daily_limit(exc: Exception) -> bool:
    """HTTP 429 caused by a per-day quota (Groq: '... tokens per day (TPD) ...'; Gemini quota ids contain
    'PerDay')."""
    text = str(exc).lower()
    return is_rate_limit(exc) and any(k in text for k in ("tokens per day", "(tpd)", "requests per day",
                                                           "perday", "per day"))


def is_server_error(exc: Exception) -> bool:
    """5xx, timeout or connection error: worth one retry of the same provider."""
    reason = unavailable_reason(exc)
    return reason is not None and not reason.startswith("rate limit")


def with_server_retry(fn: Callable[[], Any], sleep: Callable[[float], None] = time.sleep,
                      delay: float = SERVER_RETRY_DELAY_S) -> Any:
    """Call fn; after a 5xx / timeout / connection error wait `delay` seconds and call it once more."""
    try:
        return fn()
    except Exception as exc:
        if not is_server_error(exc):
            raise
        sleep(delay)
        return fn()


class FirstTokenTimeout(TimeoutError):
    """No first token (or, for a non-streamed call, no response) within the provider's first-token limit."""


def run_with_deadline(fn: Callable[[], Any], seconds: float) -> Any:
    """Run fn in a worker thread and return its result, or raise FirstTokenTimeout after `seconds`.
    A timed-out worker is left to finish on its own (daemon thread, bounded by the HTTP read timeout)."""
    outcome: dict = {}
    done = threading.Event()

    def target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:          # handed to the caller below
            outcome["error"] = exc
        finally:
            done.set()
    threading.Thread(target=target, daemon=True).start()
    if not done.wait(seconds):
        raise FirstTokenTimeout(f"no first token within {seconds:g} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def unavailable_reason(exc: Exception) -> str | None:
    """Why a provider could not answer, if switching to the fallback can help: rate limit, server error (5xx),
    timeout or connection failure. None for errors the fallback would hit too (e.g. 400 bad request)."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return "rate limit (HTTP 429)"
    if isinstance(status, int) and status >= 500:
        return f"server error (HTTP {status})"
    name = type(exc).__name__
    if isinstance(exc, FirstTokenTimeout):
        return f"timeout ({exc})"
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
    name: str                     # "gemini" | "groq"
    model: str
    base_url: str
    client: Any                   # openai.OpenAI, or a stub in tests
    effort: str | None            # reasoning effort sent with each call (None = not sent)
    pacer: TpmPacer
    stream_usage: bool = True     # ask for token usage in streamed answers (dropped if the provider rejects it)
    first_token_timeout: float | None = None   # seconds to the first token / response (None = no such limit)

    @property
    def is_gemini(self) -> bool:
        return "generativelanguage.googleapis.com" in self.base_url or self.model.startswith("gemini")


def _effort_for(model: str, effort: str | None) -> str | None:
    """Reasoning effort is sent only to models known to accept it (gpt-oss on Groq, Gemini)."""
    return effort if effort and ("gpt-oss" in model or model.startswith("gemini")) else None


def make_provider(name: str, model: str, base_url: str, api_key: str | None = None, effort: str | None = None,
                  tpm_limit: int = GROQ_TPM, rpm_limit: int | None = None,
                  sleep: Callable[[float], None] = time.sleep, client: Any = None,
                  first_token_timeout: float | None = None) -> Provider:
    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=api_key or "not-set", base_url=base_url, max_retries=0,   # retries handled here
                        timeout=HTTP_TIMEOUT_S)
    return Provider(name, model, base_url, client, _effort_for(model, effort),
                    TpmPacer(limit=tpm_limit, rpm_limit=rpm_limit, sleep=sleep),
                    first_token_timeout=first_token_timeout)


def gemini_provider(api_key: str, model: str | None = None, **kw) -> Provider:
    kw.setdefault("first_token_timeout", GEMINI_FIRST_TOKEN_TIMEOUT_S)
    return make_provider("gemini", model or DEFAULT_GEMINI_MODEL, GEMINI_BASE_URL, api_key, GEMINI_REASONING,
                         tpm_limit=GEMINI_TPM, rpm_limit=GEMINI_RPM, **kw)


def groq_provider(api_key: str, model: str | None = None, **kw) -> Provider:
    return make_provider("groq", model or DEFAULT_GROQ_MODEL, GROQ_BASE_URL, api_key, GROQ_REASONING,
                         tpm_limit=GROQ_TPM, **kw)


def providers_from_env(sleep: Callable[[float], None] = time.sleep) -> tuple[Provider | None, Provider | None]:
    """(primary, failover) from GEMINI_API_KEY / GEMINI_MODEL / GROQ_API_KEY / GROQ_MODEL:
    both keys -> (Gemini, Groq); Gemini only -> (Gemini, None); Groq only -> (Groq, None); none -> (None, None)."""
    gemini_key, groq_key = os.getenv("GEMINI_API_KEY"), os.getenv("GROQ_API_KEY")
    gemini = gemini_provider(gemini_key, os.getenv("GEMINI_MODEL"), sleep=sleep) if gemini_key else None
    groq = groq_provider(groq_key, os.getenv("GROQ_MODEL"), sleep=sleep) if groq_key else None
    return (gemini, groq) if gemini else (groq, None)


# ---------------------------------------------------------------- client

class PrimaryUnavailable(Exception):
    """Internal: the primary provider failed in a way the fallback can help with (see the module docstring)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class OpenAIChat:
    """Chat client over a primary and an optional fallback provider; error policy in the module docstring.

    At most one switch per turn: once switched, the rest of the turn stays on the fallback (`start_turn`
    resets this; `switch_reason` says why it happened, for the "Done" line). Daily quota: a model that returns
    a per-day 429 is marked exhausted for the rest of this client's life (one client per UI session); when
    every model is exhausted, QuotaExhausted is raised.
    """

    def __init__(self, primary: Provider, fallback: Provider | None = None, sleep: Callable[[float], None] = time.sleep,
                 fallback_after_s: float = FALLBACK_AFTER_S, max_output_tokens: int = MAX_OUTPUT_TOKENS,
                 temperature: float = 0.2):
        self.primary = primary
        self.fallback = fallback if fallback and fallback.model != primary.model else None
        self.providers = [p for p in (self.primary, self.fallback) if p]
        self.fallback_after_s = fallback_after_s
        self.sleep = sleep
        self.set_sleep(sleep)
        self.exhausted: set[str] = set()               # models that hit their daily quota in this session
        self.switch_reason: str | None = None          # why this turn switched to the fallback (None = it did not)
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    @classmethod
    def from_env(cls, use_fallback: bool = True, **kwargs) -> "OpenAIChat":
        primary, fallback = providers_from_env()
        if primary is None:
            raise ValueError("No LLM API key: set GEMINI_API_KEY (and optionally GROQ_API_KEY).")
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

    def complete(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto",
                 on_token: Callable[[str], None] | None = None) -> LLMResponse:
        """One chat completion. With on_token, the answer is streamed: each text piece is passed to on_token as
        it arrives (the error policy applies until the stream starts)."""
        estimate = estimate_tokens(messages, tools, self.max_output_tokens // 2)
        provider = self.choose(estimate)
        try:
            return self._call(provider, messages, tools, tool_choice, estimate, on_token)
        except DailyLimitReached:
            self.exhausted.add(provider.model)
            return self.complete(messages, tools, tool_choice, on_token)      # next provider, or QuotaExhausted
        except PrimaryUnavailable as exc:
            self.switch_reason = exc.reason
            try:
                return self._call(self.fallback, messages, tools, tool_choice, estimate, on_token)
            except DailyLimitReached:
                self.exhausted.add(self.fallback.model)
                return self.complete(messages, tools, tool_choice, on_token)

    # -------------------------------------------------- one HTTP call
    def _kwargs(self, provider: Provider, messages: list[dict], tools: list[dict], tool_choice: str) -> dict:
        kwargs: dict = dict(model=provider.model, messages=prepare_messages(messages, provider.is_gemini))
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice)
        if provider.is_gemini:
            # Gemini 3: Google advises keeping the default temperature (1.0); lower values can cause loops.
            kwargs["max_tokens"] = max(self.max_output_tokens, GEMINI_MAX_OUTPUT_TOKENS)
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
            # A provider that rejects an optional parameter (reasoning level, streamed usage): stop sending it
            # and try once more without it.
            if getattr(exc, "status_code", None) == 400:
                for param, pattern in OPTIONAL_PARAMS.items():
                    if param in kwargs and re.search(pattern, str(exc), re.I):
                        kwargs.pop(param)
                        if param == "reasoning_effort":
                            provider.effort = None
                        else:
                            provider.stream_usage = False
                        return create(**kwargs)
            raise

    def _fetch(self, provider: Provider, kwargs: dict, stream: bool) -> tuple[Any, Any]:
        """Send one request and return (response headers, parsed completion) - when streaming, an iterator whose
        first chunk has already arrived. With a first-token limit (Gemini), only the wait for that FIRST chunk
        is timed; the rest of a stream is read without a total limit, so long answers are never cut."""
        def first_chunk():
            raw = self._request(provider, kwargs)
            parsed = raw.parse()
            if stream:
                chunks = iter(parsed)
                first = next(chunks, None)
                parsed = itertools.chain([first], chunks) if first is not None else iter(())
            return raw.headers, parsed
        if not provider.first_token_timeout:
            return first_chunk()
        return run_with_deadline(first_chunk, provider.first_token_timeout)

    def _call(self, provider: Provider, messages: list[dict], tools: list[dict], tool_choice: str,
              estimate: int, on_token: Callable[[str], None] | None = None) -> LLMResponse:
        provider.pacer.wait_for(estimate)
        kwargs = self._kwargs(provider, messages, tools, tool_choice)
        if on_token is not None:
            kwargs["stream"] = True
            if provider.stream_usage:
                kwargs["stream_options"] = {"include_usage": True}
        request = lambda: self._fetch(provider, kwargs, stream=on_token is not None)  # noqa: E731
        if provider is self.primary and self.fallback and self.fallback.model not in self.exhausted:
            # Primary with a usable fallback: 429 -> switch now; 5xx/timeout -> one retry, then switch.
            server_errors = 0
            while True:
                try:
                    raw = request()
                    break
                except Exception as exc:
                    if is_daily_limit(exc):
                        self.exhausted.add(provider.model)   # out for today: fallback for the rest of the session
                    reason = unavailable_reason(exc)
                    if reason is None:
                        raise
                    if is_server_error(exc) and server_errors == 0:
                        server_errors += 1
                        self.sleep(SERVER_RETRY_DELAY_S)
                        continue
                    retried = ", after 1 retry" if server_errors else ""
                    raise PrimaryUnavailable(f"{provider.name} {reason}{retried}") from exc
        else:
            # No fallback (or this is the fallback): 429 -> backoff retries; 5xx/timeout -> one retry.
            raw = with_retry(lambda: with_server_retry(request, self.sleep), sleep=self.sleep)
        headers, parsed = raw
        self._observe(provider, headers)
        if on_token is not None:
            content, calls, usage, finish = self._read_stream(parsed, on_token)
        else:
            resp = parsed
            usage, msg = resp.usage, resp.choices[0].message
            finish = getattr(resp.choices[0], "finish_reason", None) or "stop"
            content = msg.content or ""
            calls = [ToolCall(c.id, c.function.name, c.function.arguments or "{}",
                              (getattr(c, "model_extra", None) or {}).get("extra_content"))
                     for c in (msg.tool_calls or [])]
        # Without usage (a stream whose provider does not report it): estimate, for pacing only.
        total = usage.total_tokens if usage else estimate_tokens(messages, tools, len(content) // 4)
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        # Groq: cached prompt tokens do not count toward the rate limits.
        provider.pacer.record(total if provider.is_gemini else max(0, total - cached))
        return LLMResponse(content, calls, total,
                           usage.prompt_tokens if usage else 0,
                           usage.completion_tokens if usage else 0,
                           model=provider.model, finish_reason=finish)

    @staticmethod
    def _read_stream(stream, on_token: Callable[[str], None]) -> tuple[str, list[ToolCall], Any, str]:
        """Collect a streamed completion, passing each text piece to on_token as it arrives.
        Returns (text, tool calls, usage, finish reason; "" if the stream ended without one)."""
        parts, usage, pending, finish = [], None, {}, ""
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            for choice in getattr(chunk, "choices", None) or []:
                finish = getattr(choice, "finish_reason", None) or finish
                delta = choice.delta
                if getattr(delta, "content", None):
                    parts.append(delta.content)
                    on_token(delta.content)
                for tc in getattr(delta, "tool_calls", None) or []:      # not expected with tool_choice="none"
                    item = pending.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    item["id"] = tc.id or item["id"]
                    if tc.function:
                        item["name"] += tc.function.name or ""
                        item["arguments"] += tc.function.arguments or ""
        calls = [ToolCall(c["id"], c["name"], c["arguments"] or "{}") for c in pending.values() if c["name"]]
        return "".join(parts), calls, usage, finish

    @staticmethod
    def _observe(provider: Provider, headers) -> None:
        """Take the real per-model TPM limit and the remaining budget from the response headers, if sent."""
        limit = _number(headers, "x-ratelimit-limit-tokens")
        if limit:
            provider.pacer.limit = int(limit)
        provider.pacer.observe(headers)
