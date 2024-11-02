from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from graphlib import TopologicalSorter
from multiprocessing import Queue as PQueue
from queue import Queue
from typing import Any, Callable, Generic, TypeVar, cast

import pydot  # type: ignore

T = TypeVar("T")


class CGCache(ABC):
    """Interface for Cache objects"""

    @abstractmethod
    def read(self, key: str) -> Any: ...
    @abstractmethod
    def write(self, key: str, value: Any) -> bool: ...
    @abstractmethod
    def has(self, key: str) -> bool: ...


class CGCacheInMem(CGCache):
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


class CGCacheProxy(CGCache):
    """A pass-through cache used to ensure pointers are kept"""

    _cache: CGCache

    def __init__(self, cache: CGCache) -> None:
        self._cache = cache

    def read(self, key: str) -> Any:
        return self._cache.read(key)

    def write(self, key: str, value: Any) -> bool:
        return self._cache.write(key, value)

    def has(self, key: str) -> bool:
        return self._cache.has(key)


class CG(ABC):
    """The abstract class for all graphs"""

    __cg_graph__: dict[str, set[str]]
    __cg_cache__: CGCacheProxy
    __cg_inputs__: set[str]
    __cg_func_map__: dict[str, Callable]
    __cg_input_map__: dict[str, dict[str, str]]

    def __init__(self, **inputs: Any):
        for name, value in inputs.items():
            if name not in self.__cg_inputs__:
                raise AttributeError()
            print(f"Setting {name} on CG instance")
            setattr(self, name, value)


class CGMember(ABC, Generic[T]):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def graph(self) -> dict[str, set[str]]: ...
    @property
    @abstractmethod
    def inputs(self) -> dict[str, "CGMember[Any]"]: ...
    @property
    @abstractmethod
    def func_map(self) -> dict[str, Callable]: ...
    @property
    @abstractmethod
    def input_map(self) -> dict[str, dict[str, str]]: ...
    @abstractmethod
    def func(self, *args: Any, **kwargs: Any) -> T: ...
    def _repr_png_(self) -> Any:
        from IPython.display import SVG, display  # type: ignore

        graph = build_viz_graph(self)
        return display(SVG(graph.create_svg()))


class CGField(CGMember[T], Generic[T]):
    _func: Callable[..., T]
    _inputs: dict[str, CGMember[Any]]

    _name: str | None
    _cache: CGCache | None
    _graph: dict[str, set[str]] | None
    _func_map: dict[str, Callable] | None
    _input_map: dict[str, dict[str, str]] | None

    def __init__(self, func: Callable[..., T], **inputs: CGMember[T]) -> None:
        print("Initializing CGField")
        self._func = func
        self._inputs = inputs

        self._name = None
        self._cache = None
        self._graph = None
        self._func_map = None

    def __set_name__(self, owner: CG, name: str) -> None:
        print(f"setting name on CGField {name} - owner: {owner}")
        self._name = name
        if not hasattr(owner, "__cg_graph__"):
            setattr(owner, "__cg_graph__", dict())
        if not hasattr(owner, "__cg_func_map__"):
            setattr(owner, "__cg_func_map__", dict())
        if not hasattr(owner, "__cg_input_map__"):
            setattr(owner, "__cg_input_map__", dict())
        if not hasattr(owner, "__cg_cache__"):
            # TODO: should this be initialized upon CG.__init__ ?
            setattr(owner, "__cg_cache__", CGCacheProxy(CGCacheInMem()))
        self._cache = owner.__cg_cache__
        self._graph = owner.__cg_graph__
        self._func_map = owner.__cg_func_map__
        self._func_map[self.name] = self.func
        self._input_map = owner.__cg_input_map__
        self._input_map[self.name] = {k: inp.name for k, inp in self.inputs.items()}
        self._graph[name] = set([inp.name for inp in self.inputs.values()])
        print(f"{self._name} : {self._graph} {self._cache}")

    def __get__(self, owner: CG | None, owner_type: type[CG] | None = None) -> "CGField":
        print(f"Getting on CGField {self.name} - owner: {owner}")
        if owner is None:
            return self
        return self

    def __call__(self) -> T:
        if self.cache.has(self.name):
            return cast(T, self.cache.read(self.name))
        # exec = _call
        # exec = call
        exec = Executor()
        # exec = MultiprocessingExecutor()
        # exec = SequentialProcessExecutor()
        return exec(self, self.graph, self.cache)

    @property
    def name(self) -> str:
        if self._name is None:
            raise ValueError("CGField must be used as a descriptor")
        return self._name

    @property
    def cache(self) -> CGCache:
        # TODO: Should this be private - and _cache becomes __cache?
        if self._cache is None:
            raise ValueError("CGField must be used as a descriptor")
        return self._cache

    @property
    def graph(self) -> dict[str, set[str]]:
        # TODO: Should this be private - and _graph becomes __graph?
        if self._graph is None:
            raise ValueError("CGField must be used as a descriptor")
        return self._graph

    @property
    def value(self) -> T:
        if self.cache.has(self.name):
            return cast(T, self.cache.read(self.name))
        raise ValueError("Field {self.name} has not been computed yet")

    @property
    def func_map(self) -> dict[str, Callable]:
        if self._func_map is None:
            raise ValueError("CGField must be used as a descriptor")
        return self._func_map

    @property
    def input_map(self) -> dict[str, dict[str, str]]:
        if self._input_map is None:
            raise ValueError("CGField must be used as a descriptor")
        return self._input_map

    @property
    def inputs(self) -> dict[str, CGMember[Any]]:
        return self._inputs

    def func(self, *args: Any, **kwargs: Any) -> T:
        return self._func(*args, **kwargs)


class CGInput(CGMember[T], Generic[T]):
    _name: str | None
    _value: T | None
    _graph: dict[str, set[str]] | None
    _cache: CGCache | None
    _input_map: dict[str, dict[str, str]] | None

    _func_map: dict[str, Callable] | None

    def __init__(self) -> None:
        self._name = None
        self._value = None
        self._graph = None
        self._cache = None
        self._input_map = None

    def __set_name__(self, owner: CG, name: str) -> None:
        print(f"Setting name on CGInput {name} - owner: {owner}")
        self._name = name
        if not hasattr(owner, "__cg_graph__"):
            setattr(owner, "__cg_graph__", dict())
        if not hasattr(owner, "__cg_input_map__"):
            setattr(owner, "__cg_input_map__", dict())
        if not hasattr(owner, "__cg_inputs__"):
            setattr(owner, "__cg_inputs__", set())
        if not hasattr(owner, "__cg_func_map__"):
            setattr(owner, "__cg_func_map__", dict())
        if not hasattr(owner, "__cg_cache__"):
            # TODO: should this be initialized upon CG.__init__ ?
            setattr(owner, "__cg_cache__", CGCacheProxy(CGCacheInMem()))
        owner.__cg_inputs__.add(name)
        owner.__cg_func_map__[name] = self.func
        self._graph = owner.__cg_graph__
        self._cache = owner.__cg_cache__
        self._input_map = owner.__cg_input_map__
        self._input_map[self.name] = {}

    def __set__(self, owner: CG, value: T) -> None:
        print(f"Setting value on CGInput {self.name} = {value} # owner: {owner}")
        self._value = value
        self.cache.write(self.name, value)

    def __get__(self, owner: CG | None, owner_type: type[CG] | None = None) -> T:
        print(f"Getting on CGInput {self.name} - owner: {owner}")
        return self.value

    def func(self) -> T:
        return self.value

    @property
    def name(self) -> str:
        if self._name is None:
            raise ValueError("CGInput should be used as a descriptor")
        return self._name

    @property
    def value(self) -> T:
        if self._value is None:
            raise ValueError("CGInput should be used as a descriptor")
        return self._value

    @property
    def inputs(self) -> dict[str, CGMember[T]]:
        return {}

    @property
    def graph(self) -> dict[str, set[str]]:
        if self._graph is None:
            raise ValueError("CGField must be used as a descriptor")
        return self._graph

    @property
    def func_map(self) -> dict[str, Callable]:
        if self._func_map is None:
            raise ValueError("CGField must be used as a descriptor")
        return self._func_map

    @property
    def input_map(self) -> dict[str, dict[str, str]]:
        if self._input_map is None:
            raise ValueError("CGField must be used as a descriptor")
        return self._input_map

    @property
    def cache(self) -> CGCache:
        # TODO: Should this be private - and _cache becomes __cache?
        if self._cache is None:
            raise ValueError("CGField must be used as a descriptor")
        return self._cache

    def __call__(self) -> T:
        return self.value


def call(field: CGMember[T], graph: dict[str, set[str]], cache: CGCache) -> T:
    """Simplest executor, as part of same process, leverages recursion"""
    value: T = field.func(**{key: call(input, graph, cache) for key, input in field.inputs.items()})
    cache.write(field.name, value)
    return value


class Executor:
    """An in-process sequential executor which can be sub-classed for easy extensions"""

    def __init__(self, finalized_tasks_queue: Queue[tuple[str, Any]] | None = None):
        self._finalized_tasks_queue = finalized_tasks_queue or Queue()

    def __call__(self, field: CGMember[T], graph: dict[str, set[str]], cache: CGCache) -> T:
        execution_graph = build_exec_graph(field, graph, cache)
        execution_graph.prepare()

        while execution_graph.is_active():
            for node in execution_graph.get_ready():
                print(f"Getting next computation from execution_graph for {node}")
                _func = field.func_map[node]
                kwargs = {key: cache.read(predecessor) for key, predecessor in field.input_map[node].items()}
                print(f"Submitting {node}={_func}(**{kwargs})")
                self.submit(node, _func, kwargs)

            (node, result) = self._finalized_tasks_queue.get()
            print(f"Writing in cache {node}={result}")
            cache.write(node, result)
            execution_graph.done(node)

        return cast(T, cache.read(field.name))

    def submit(self, node: str, func: Callable, kwargs: dict[str, Any]) -> None:
        result = func(**kwargs)
        self._finalized_tasks_queue.put((node, result))


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


class MultiprocessingExecutor(Executor):
    def __init__(self, max_workers: int | None = None) -> None:
        queue: PQueue[tuple[str, Any]] = PQueue()
        super().__init__(finalized_tasks_queue=queue)  # type: ignore  # multiprocessing.Queue should be subtype of queue.Queue
        self._executor = ProcessPoolExecutor(max_workers=max_workers, initializer=init_pool_processes, initargs=(queue,))  # type: ignore  # unable to map initargs types to initializer signature

    def submit(self, node: str, func: Callable, kwargs: dict[str, Any]) -> None:

        _func = FunctionWrapper(node, func)

        with self._executor as executor:
            print(f"Submitting {func.__name__}(**{kwargs}) to pool")
            future = executor.submit(_func, kwargs)
            print(f"Submit done")


class SequentialProcessExecutor(Executor):
    def __init__(self) -> None:
        queue: PQueue[tuple[str, Any]] = PQueue()
        super().__init__(finalized_tasks_queue=queue)  # type: ignore  # multiprocessing.Queue should be subtype of queue.Queue
        self._executor = ProcessPoolExecutor(max_workers=1)  # type: ignore  # unable to map initargs types to initializer signature

    def submit(self, node: str, func: Callable, kwargs: dict[str, Any]) -> None:

        with self._executor as executor:
            print(f"Submitting {func.__name__}(**{kwargs}) to pool")
            future = executor.submit(func, **kwargs)
            print(f"Submit done")
            result = future.result()
            self._finalized_tasks_queue.put((node, result))


def _call(field: CGMember[T], graph: dict[str, set[str]], cache: CGCache) -> T:

    execution_graph = build_exec_graph(field, graph, cache)
    execution_graph.prepare()

    task_queue: Queue[tuple[str, Callable, dict[str, Any]]] = Queue()
    finalized_tasks_queue: Queue[tuple[str, Any]] = Queue()

    while execution_graph.is_active():
        for node in execution_graph.get_ready():
            print(f"Getting next computation from execution_graph for {node}")
            # Worker threads or processes take nodes to work on off the
            # 'task_queue' queue.
            _func = field.func_map[node]
            # Can we read this without the cache?
            kwargs = {key: cache.read(predecessor) for key, predecessor in field.input_map[node].items()}
            task_queue.put((node, _func, kwargs))

            ###
            node, _func, predecessors = task_queue.get()
            result = _func(**predecessors)
            finalized_tasks_queue.put((node, result))
            ###
            print(f"Computed {node}={result}")

        (node, result) = finalized_tasks_queue.get()
        print(f"Writing in cache {node}={result}")
        cache.write(node, result)
        execution_graph.done(node)

    return cast(T, cache.read(field.name))


def build_exec_graph(field: CGMember[T], graph: dict[str, set[str]], cache: CGCache) -> TopologicalSorter:
    ts: TopologicalSorter[str] = TopologicalSorter()
    print(f"{graph}")
    predecessors = [(field.name, p) for p in graph[field.name]]
    print(f"{predecessors}")
    pruned_graph: dict[str, set[str]] = {field.name: set()}
    while predecessors:
        node, predecessor = predecessors.pop()
        if cache.has(predecessor):
            continue
        print(f"Adding nodes: {node} -> {predecessor}")
        if node not in pruned_graph:
            pruned_graph[node] = set()
        pruned_graph[node].add(predecessor)
        predecessors += [(predecessor, p) for p in graph[predecessor]]
    print("Pruned", pruned_graph)
    return TopologicalSorter(pruned_graph)


def build_viz_graph(field: CGMember[T]) -> pydot.Dot:
    graph = pydot.Dot("my_graph", graph_type="digraph", rankdir="LR")
    for node in field.graph.keys():
        graph.add_node(pydot.Node(node, label=node, color="black", style="rounded"))
    for node, dependencies in field.graph.items():
        for dependency in dependencies:
            graph.add_edge(pydot.Edge(node, dependency, color="black"))
    return graph


# Test only
def add(x: float, y: float) -> float:
    return x + y


class Example(CG):
    a: CGInput[float] = CGInput()
    b: CGInput[float] = CGInput()
    c: CGField[float] = CGField(add, x=a, y=b)
    d: CGField[float] = CGField(add, x=c, y=b)
    e: CGField[float] = CGField(add, x=d, y=a)
    f: CGField[float] = CGField(add, x=e, y=e)


if __name__ == "__main__":

    e = Example(a=1, b=2)
    assert e.c() == 3, f"Found {e.c()}"
    assert e.d() == 5
    assert e.e() == 6
    assert e.f() == 12
