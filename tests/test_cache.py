"""First-question answer cache."""

from agent.agent import Turn
from ui.cache import AnswerCache, cache_key, normalize


def test_normalize_and_key():
    assert normalize("  What is the Unmet demand at SFO?? ") == "what is the unmet demand at sfo"
    assert cache_key("What about SFO?", "m", "v1", "2026-11-18") == ("what about sfo", "m", "v1", "2026-11-18")
    assert cache_key("q", "m", "v1", "d1") != cache_key("q", "m", "v2", "d1")      # prompt change invalidates
    assert cache_key("q", "m", "v1", "d1") != cache_key("q", "m", "v1", "d2")      # data rebuild invalidates


def test_cache_returns_copies_and_evicts_oldest():
    cache = AnswerCache(max_entries=2)
    turn = Turn(user="q1", answer="a1")
    cache.put(("q1",), {"text": "a1", "meta": "Done"}, turn)
    hit = cache.get(("q1",))
    hit.message["text"] = "changed"
    assert cache.get(("q1",)).message["text"] == "a1"                       # stored copy is not mutated
    cache.put(("q2",), {"text": "a2"}, Turn(user="q2"))
    cache.put(("q3",), {"text": "a3"}, Turn(user="q3"))
    assert cache.get(("q1",)) is None and len(cache) == 2
