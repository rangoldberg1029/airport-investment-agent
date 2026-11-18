"""429 retry and TPM pacing with injected sleep/clock (no network, no real waiting)."""

import pytest

from agent.llm import LLMResponse, RateLimited, TpmPacer, with_retry


class Http429(Exception):
    status_code = 429

    def __init__(self, retry_after=None):
        super().__init__("rate limited")
        self.response = type("R", (), {"headers": {"retry-after": retry_after} if retry_after else {}})()


def test_retry_then_success_uses_retry_after():
    attempts, sleeps = [], []

    def fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise Http429(retry_after="1.5")
        return LLMResponse("ok")

    assert with_retry(fn, sleep=sleeps.append).content == "ok"
    assert sleeps == [1.5, 1.5]


def test_retry_exhausted_raises_friendly_rate_limit():
    sleeps = []

    def fn():
        raise Http429()

    with pytest.raises(RateLimited, match="rate-limited"):
        with_retry(fn, max_retries=3, sleep=sleeps.append)
    assert len(sleeps) == 3 and all(s >= 2 for s in sleeps)      # backoff 2/4/8 s (+ jitter)


def test_retry_respects_total_wait_cap():
    sleeps = []
    with pytest.raises(RateLimited):
        with_retry(lambda: (_ for _ in ()).throw(Http429(retry_after="20")), max_total_wait=30, sleep=sleeps.append)
    assert sleeps == [20.0]


def test_non_rate_limit_errors_are_not_retried():
    with pytest.raises(ValueError):
        with_retry(lambda: (_ for _ in ()).throw(ValueError("bad request")), sleep=lambda s: None)


def test_pacer_waits_until_window_has_room():
    now = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    pacer = TpmPacer(limit=8000, window=60, clock=lambda: now[0], sleep=sleep)
    pacer.record(7000)
    now[0] = 10.0
    assert pacer.wait_for(500) == 0                 # 7,500 fits
    waited = pacer.wait_for(2000)                    # 9,000 does not -> wait until the 7,000 expires
    assert waited == pytest.approx(50.5) or waited == pytest.approx(30)   # capped by max_wait (30 s)
    assert sleeps


def test_pacer_window_expiry():
    now = [0.0]
    pacer = TpmPacer(limit=1000, window=60, clock=lambda: now[0], sleep=lambda s: None)
    pacer.record(900)
    now[0] = 61
    assert pacer.used() == 0
