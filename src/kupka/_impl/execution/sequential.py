import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue as PQueue
from typing import Any, Callable

from kupka._impl.execution.core import KPExecutor, Contextual


_LOGGER = logging.getLogger(__name__)


class SequentialProcessKPExecutor(KPExecutor):
    def __init__(self) -> None:
        queue: PQueue[tuple[str, Any]] = PQueue()
        super().__init__(finalized_tasks_queue=queue)  # type: ignore  # multiprocessing.Queue should be subtype of queue.Queue

    def submit(self, node: str, func: Callable, kwargs: dict[str, Any], context: Contextual) -> None:

        _LOGGER.debug(f"[EXECUTION] Submitting {func.__name__}(**{kwargs}) to pool")
        future = context.submit(func, **kwargs)
        _LOGGER.debug(f"[EXECUTION] Submit done")
        result = future.result()
        self._finalized_tasks_queue.put((node, result))

    def context(self) -> Contextual:
        return ProcessPoolExecutor(max_workers=1)  # type: ignore  # unable to map initargs types to initializer signature
