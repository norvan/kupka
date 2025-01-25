import logging
from contextlib import contextmanager
from typing import Any, Callable, ContextManager, Iterator, TYPE_CHECKING
from queue import Queue

if TYPE_CHECKING:
    from kupka._impl.kupka import Kupka


Contextual = Any  # TODO: Find a better type hint
_LOGGER = logging.getLogger(__name__)


@contextmanager
def passthrough() -> Iterator:
    try:
        yield
    finally:
        pass


class KPExecutor:
    """An in-process sequential executor which can be sub-classed for easy extensions"""

    def __init__(self, finalized_tasks_queue: Queue[tuple[str, Any]] | None = None):
        self._finalized_tasks_queue = finalized_tasks_queue or Queue()

    def __call__(self, name: str, kupka: "Kupka") -> Any:
        execution_graph = kupka.build_exec_graph(name)
        execution_graph.prepare()

        with self.context() as context:
            while execution_graph.is_active():
                for node in execution_graph.get_ready():
                    _LOGGER.debug(f"[EXECUTION] Getting next computation from execution_graph for {node}")
                    _func = kupka.func_map[node]
                    kwargs = {key: kupka.cache.read(predecessor) for key, predecessor in kupka.input_map[node].items()}
                    _LOGGER.debug(f"[EXECUTION] Submitting {node}={_func}(**{kwargs})")
                    self.submit(node, _func, kwargs, context)

                (node, result) = self._finalized_tasks_queue.get()
                _LOGGER.debug(f"[EXECUTION] Writing in cache {node}={result}")
                kupka.cache.write(node, result)
                execution_graph.done(node)

        return kupka.cache.read(name)

    def submit(self, node: str, func: Callable, kwargs: dict[str, Any], context: Contextual) -> None:
        result = func(**kwargs)
        self._finalized_tasks_queue.put((node, result))

    def context(self) -> ContextManager[Contextual]:
        return passthrough()
