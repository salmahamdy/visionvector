"""In-memory LRU cache with TTL for embedding and search results."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, Generic, TypeVar

V = TypeVar("V")


class TTLCache(Generic[V]):
    """Thread-unsafe LRU cache with per-entry TTL."""

    def __init__(self, maxsize: int = 256, ttl_seconds: float = 300.0) -> None:
        self._cache: OrderedDict[str, tuple[V, float]] = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl_seconds

    def _is_expired(self, ts: float) -> bool:
        return (time.monotonic() - ts) > self.ttl

    def get(self, key: str) -> V | None:
        if key not in self._cache:
            return None
        value, ts = self._cache[key]
        if self._is_expired(ts):
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: V) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.monotonic())
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


def make_cache_key(*args: Any) -> str:
    payload = "|".join(str(a) for a in args)
    return hashlib.sha256(payload.encode()).hexdigest()
