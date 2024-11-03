from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from graphlib import TopologicalSorter
from multiprocessing import Queue as PQueue
from queue import Queue
from typing import Any, Callable, Generic, TypeVar, cast

import pydot  # type: ignore

T = TypeVar("T")


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


class WrappedValue(Generic[T]):
    _value: T | None
    _initialized: bool

    @classmethod
    def from_value(cls, value: T) -> "WrappedValue":
        return cls(value, True)

    @classmethod
    def init(cls) -> "WrappedValue":
        return cls(None, False)

    def __init__(self, value: T | None, initialized: bool):
        self._value = value
        self._initialized = initialized

    def set_value(self, value: T) -> None:
        if self._initialized:
            raise AttributeError(f"Value has already been set!")
        self._value = value
        self._initialized = True

    def __call__(self) -> T:
        if not self._initialized:
            raise ValueError(f"Value has not been set!")
        return cast(T, self._value)


class Kupka:
    _graph: dict[str, set[str]]
    _cache: KPCacheProxy
    _inputs: set[str]
    _func_map: dict[str, Callable]
    _input_map: dict[str, dict[str, str]]

    @classmethod
    def get_kupka(cls, owner: "KP") -> "Kupka":
        if not hasattr(owner, "__kupka__"):
            setattr(owner, "__kupka__", Kupka())
        return owner.__kupka__

    def __init__(self) -> None:
        self._cache = KPCacheProxy(KPCacheInMem())
        self._graph = {}
        self._inputs = set()
        self._func_map = {}
        self._input_map = {}

    def add_field(self, node: str, func: Callable, inputs: dict[str, str]) -> None:
        self._graph[node] = set(inputs.values())
        self._func_map[node] = func
        self._input_map[node] = inputs

    def add_input(self, node: str, value: WrappedValue[T]) -> None:
        self._graph[node] = set()
        self._func_map[node] = value
        self._input_map[node] = {}
        self._inputs.add(node)

    def register_input_value(self, node: str, value: T) -> None:
        self._cache.write(node, value)

    @property
    def cache(self) -> KPCache:
        return self._cache

    @property
    def graph(self) -> dict[str, set[str]]:
        return self._graph

    @property
    def inputs(self) -> set[str]:
        return self._inputs

    @property
    def func_map(self) -> dict[str, Callable]:
        return self._func_map

    @property
    def input_map(self) -> dict[str, dict[str, str]]:
        return self._input_map

    def build_exec_graph(self, name: str) -> TopologicalSorter:
        ts: TopologicalSorter[str] = TopologicalSorter()
        print(f"{self.graph}")
        predecessors = [(name, p) for p in self.graph[name]]
        print(f"{predecessors}")
        pruned_graph: dict[str, set[str]] = {name: set()}
        while predecessors:
            node, predecessor = predecessors.pop()
            if self.cache.has(predecessor):
                continue
            print(f"Adding nodes: {node} -> {predecessor}")
            if node not in pruned_graph:
                pruned_graph[node] = set()
            pruned_graph[node].add(predecessor)
            predecessors += [(predecessor, p) for p in self.graph[predecessor]]
        print("Pruned", pruned_graph)
        return TopologicalSorter(pruned_graph)

    def build_viz_graph(self, node: str) -> pydot.Dot:
        graph = pydot.Dot("my_graph", graph_type="digraph", rankdir="LR")
        for node in self.graph.keys():
            graph.add_node(pydot.Node(node, label=node, color="black", style="rounded"))
        for node, dependencies in self.graph.items():
            for dependency in dependencies:
                graph.add_edge(pydot.Edge(node, dependency, color="black"))
        return graph


class KP(ABC):
    """The abstract class for all graphs"""

    __kupka__: Kupka

    def __init__(self, **inputs: Any):
        for name, value in inputs.items():
            if name not in self.__kupka__.inputs:
                raise AttributeError()
            print(f"Setting {name} on KP instance")
            setattr(self, name, value)


class KPMember(ABC, Generic[T]):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def __kupka__(self) -> Kupka: ...
    def _repr_png_(self) -> Any:
        from IPython.display import SVG, display  # type: ignore

        graph = self.__kupka__.build_viz_graph(self.name)
        return display(SVG(graph.create_svg()))


class KPField(KPMember[T], Generic[T]):
    _func: Callable[..., T]
    _inputs: dict[str, KPMember[Any]]
    _name: str | None
    _kupka: Kupka | None

    def __init__(self, func: Callable[..., T], **inputs: KPMember[T]) -> None:
        print("Initializing KPField")
        self._func = func
        self._inputs = inputs
        self._kupka = None
        self._name = None

    @property
    def __kupka__(self) -> Kupka:
        if self._kupka is None:
            raise ValueError("KPField must be used as a descriptor")
        return self._kupka

    def __set_name__(self, owner: KP, name: str) -> None:
        print(f"setting name on KPField {name} - owner: {owner}")
        self._kupka = Kupka.get_kupka(owner)
        self._kupka.add_field(
            node=name,
            func=self._func,
            inputs={k: inp.name for k, inp in self._inputs.items()},
        )
        self._name = name

    def __get__(self, owner: KP | None, owner_type: type[KP] | None = None) -> "KPField":
        print(f"Getting on KPField {self.name} - owner: {owner}")
        if owner is None:
            return self
        return self

    def __call__(self) -> T:
        if self.__kupka__.cache.has(self.name):
            return cast(T, self.__kupka__.cache.read(self.name))
        # exec = _call
        exec = call
        # exec = KPExecutor()
        # exec = MultiprocessingKPExecutor()
        # exec = SequentialProcessKPExecutor()

        return cast(T, exec(self.name, self.__kupka__))

    @property
    def name(self) -> str:
        if self._name is None:
            raise ValueError("KPField must be used as a descriptor")
        return self._name


class KPInput(KPMember[T], Generic[T]):
    _name: str | None
    _value: T | None
    _wrapper: WrappedValue
    _kupka: Kupka | None

    def __init__(self) -> None:
        self._name = None
        self._value = None
        self._wrapper = WrappedValue.init()

    @property
    def __kupka__(self) -> Kupka:
        if self._kupka is None:
            raise ValueError("KPField must be used as a descriptor")
        return self._kupka

    def __set_name__(self, owner: KP, name: str) -> None:
        print(f"Setting name on KPInput {name} - owner: {owner}")
        self._kupka = Kupka.get_kupka(owner)
        self._name = name
        self._kupka.add_input(node=self.name, value=self._wrapper)

    def __set__(self, owner: KP, value: T) -> None:
        print(f"Setting value on KPInput {self.name} = {value} # owner: {owner}")
        self._value = value
        self._wrapper.set_value(value)
        self.__kupka__.register_input_value(self.name, value)

    def __get__(self, owner: KP | None, owner_type: type[KP] | None = None) -> T:
        print(f"Getting on KPInput {self.name} - owner: {owner}")
        return self.value

    @property
    def name(self) -> str:
        if self._name is None:
            raise ValueError("KPInput should be used as a descriptor")
        return self._name

    @property
    def value(self) -> T:
        if self._value is None:
            raise ValueError("KPInput should be used as a descriptor")
        return self._value

    def __call__(self) -> T:
        return self.value


def call(name: str, kupka: Kupka) -> Any:
    """Simplest executor, as part of same process, leverages recursion"""
    value = kupka.func_map[name](**{key: call(input, kupka) for key, input in kupka.input_map[name].items()})
    kupka.cache.write(name, value)
    return value


class KPExecutor:
    """An in-process sequential executor which can be sub-classed for easy extensions"""

    def __init__(self, finalized_tasks_queue: Queue[tuple[str, Any]] | None = None):
        self._finalized_tasks_queue = finalized_tasks_queue or Queue()

    def __call__(self, name: str, kupka: Kupka) -> Any:
        execution_graph = kupka.build_exec_graph(name)
        execution_graph.prepare()

        while execution_graph.is_active():
            for node in execution_graph.get_ready():
                print(f"Getting next computation from execution_graph for {node}")
                _func = kupka.func_map[node]
                kwargs = {key: kupka.cache.read(predecessor) for key, predecessor in kupka.input_map[node].items()}
                print(f"Submitting {node}={_func}(**{kwargs})")
                self.submit(node, _func, kwargs)

            (node, result) = self._finalized_tasks_queue.get()
            print(f"Writing in cache {node}={result}")
            kupka.cache.write(node, result)
            execution_graph.done(node)

        return kupka.cache.read(name)

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


class MultiprocessingKPExecutor(KPExecutor):
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


class SequentialProcessKPExecutor(KPExecutor):
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


def _call(name: str, kupka: Kupka) -> Any:

    execution_graph = kupka.build_exec_graph(name)
    execution_graph.prepare()

    task_queue: Queue[tuple[str, Callable, dict[str, Any]]] = Queue()
    finalized_tasks_queue: Queue[tuple[str, Any]] = Queue()

    while execution_graph.is_active():
        for node in execution_graph.get_ready():
            print(f"Getting next computation from execution_graph for {node}")
            _func = kupka.func_map[node]
            kwargs = {key: kupka.cache.read(predecessor) for key, predecessor in kupka.input_map[node].items()}
            task_queue.put((node, _func, kwargs))

            ###
            node, _func, predecessors = task_queue.get()
            result = _func(**predecessors)
            finalized_tasks_queue.put((node, result))
            ###
            print(f"Computed {node}={result}")

        (node, result) = finalized_tasks_queue.get()
        print(f"Writing in cache {node}={result}")
        kupka.cache.write(node, result)
        execution_graph.done(node)

    return kupka.cache.read(name)


def build_exec_graph(field: KPMember[T], graph: dict[str, set[str]], cache: KPCache) -> TopologicalSorter:
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


# Test only
def add(x: float, y: float) -> float:
    return x + y


class Example(KP):
    a: KPInput[float] = KPInput()
    b: KPInput[float] = KPInput()
    c: KPField[float] = KPField(add, x=a, y=b)
    d: KPField[float] = KPField(add, x=c, y=b)
    e: KPField[float] = KPField(add, x=d, y=a)
    f: KPField[float] = KPField(add, x=e, y=e)


if __name__ == "__main__":

    e = Example(a=1, b=2)
    assert e.c() == 3, f"Found {e.c()}"
    assert e.d() == 5
    assert e.e() == 6
    assert e.f() == 12
