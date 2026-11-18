"""Answer cache for the FIRST question of a conversation (instant demo answers, zero tokens).

Key = (normalized question, primary model, prompt version, data build date): any change to the model, the
prompt/answer format or the data invalidates it. Follow-up questions are never cached because their answer
depends on the conversation so far.
"""

import copy
import re
from collections import OrderedDict
from dataclasses import dataclass

from agent.agent import Turn


def normalize(question: str) -> str:
    """Lower-case, collapse whitespace, drop trailing punctuation: 'What is X?' == 'what is  x'."""
    return re.sub(r"\s+", " ", question.strip().lower()).rstrip(" ?.!")


def cache_key(question: str, model: str, prompt_version: str, data_built: str | None) -> tuple:
    return normalize(question), model, prompt_version, data_built or "unknown"


@dataclass
class CachedAnswer:
    message: dict        # what the UI renders: text, records, untraceable, meta
    turn: Turn           # restored into the Agent so follow-ups have context


class AnswerCache:
    """Small in-memory LRU cache (one per server process)."""

    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self._items: OrderedDict[tuple, CachedAnswer] = OrderedDict()

    def get(self, key: tuple) -> CachedAnswer | None:
        item = self._items.get(key)
        if item is not None:
            self._items.move_to_end(key)
            return copy.deepcopy(item)
        return None

    def put(self, key: tuple, message: dict, turn: Turn) -> None:
        self._items[key] = CachedAnswer(copy.deepcopy(message), copy.deepcopy(turn))
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)
