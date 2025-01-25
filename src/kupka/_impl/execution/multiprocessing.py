import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue as PQueue
from queue import Queue
from typing import Any, Callable, Generic, TypeVar

from kupka._impl.execution.core import KPExecutor, Contextual


T = TypeVar("T")
_LOGGER = logging.getLogger(__name__)


def init_pool_processes(q: Queue[tuple[str, Any]]) -> None:
    """This function adds the queue in the global context of the Processes spawn by the process pool"""
    global queue

    queue = q  # type: ignore


class FunctionWrapper(Generic[T]):
    """Meant to be used within a process pool. The queue is added to the global context"""

    def __init__(self, node: str, func: Callable[..., T]):
        self.func = func
        self.node = node

    def __call__(self, kwargs: Any) -> T:
        result = self.func(**kwargs)
        # The queue is added to the global context by the process pool
        queue.put((self.node, result))  # type: ignore  # The queue is in the global context
        return result


class MultiprocessingKPExecutor(KPExecutor):
    def __init__(self, max_workers: int | None = None) -> None:
        self._queue: PQueue[tuple[str, Any]] = PQueue()
        super().__init__(finalized_tasks_queue=self._queue)  # type: ignore  # multiprocessing.Queue should be subtype of queue.Queue
        self._max_workers = max_workers

    def submit(self, node: str, func: Callable, kwargs: dict[str, Any], context: Contextual) -> None:

        _func = FunctionWrapper(node, func)

        _LOGGER.debug(f"[EXECUTION] Submitting {func.__name__}(**{kwargs}) to pool")
        future = context.submit(_func, kwargs)
        _LOGGER.debug(f"[EXECUTION] Submit done")

    def context(self) -> Contextual:
        return ProcessPoolExecutor(max_workers=self._max_workers, initializer=init_pool_processes, initargs=(self._queue,))  # type: ignore  # unable to map initargs types to initializer signature
