from abc import ABC, abstractmethod
from typing import Any


class KPCache(ABC):
    """Interface for Cache objects"""

    @abstractmethod
    def read(self, key: str) -> Any: ...
    @abstractmethod
    def write(self, key: str, value: Any) -> bool: ...
    @abstractmethod
    def has(self, key: str) -> bool: ...


class KPCacheInMem(KPCache):
    _cache: dict[str, Any]

    def __init__(self, cache: dict[str, Any] | None = None) -> None:
        self._cache = cache or {}

    def read(self, key: str) -> Any:
        return self._cache[key]

    def write(self, key: str, value: Any) -> bool:
        self._cache[key] = value
        return True

    def has(self, key: str) -> bool:
        return key in self._cache


class KPCacheProxy(KPCache):
    """A pass-through cache used to ensure pointers are kept"""

    _cache: KPCache

    def __init__(self, cache: KPCache) -> None:
        self._cache = cache

    def read(self, key: str) -> Any:
        return self._cache.read(key)

    def write(self, key: str, value: Any) -> bool:
        return self._cache.write(key, value)

    def has(self, key: str) -> bool:
        return self._cache.has(key)
