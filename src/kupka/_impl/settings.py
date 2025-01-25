from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from kupka._impl.cache import KPCache, KPCacheInMem
from kupka._impl.execution.core import KPExecutor


class kp_settings:
    _executor: KPExecutor = KPExecutor()
    _cache_type: type[KPCache] = KPCacheInMem
    _cache_settings: dict[str, Any] = {}

    @classmethod
    def set_global_executor(cls, executor: KPExecutor) -> None:
        cls._executor = executor

    @classmethod
    def set_global_cache(cls, cache_type: type[KPCache], cache_settings: dict[str, Any]) -> None:
        cls._cache_type = cache_type
        cls._cache_settings = cache_settings

    @classmethod
    @contextmanager
    def use(
        cls,
        *,
        executor: KPExecutor | None = None,
        cache_type: type[KPCache] | None = None,
        cache_settings: dict[str, Any] | None = None,
    ) -> Iterator:
        executor = executor or cls._executor
        cache_type = cache_type or cls._cache_type
        cache_settings = cache_settings or cls._cache_settings
        try:
            old = cls._executor, cls._cache_type, cls._cache_settings
            cls._executor = executor
            cls._cache_type = cache_type
            cls._cache_settings = cache_settings
            yield
        finally:
            cls._executor, cls._cache_type, cls._cache_settings = old

    @classmethod
    def executor(cls) -> KPExecutor:
        return cls._executor

    @classmethod
    def build_cache(cls) -> KPCache:
        return cls._cache_type(**cls._cache_settings)
