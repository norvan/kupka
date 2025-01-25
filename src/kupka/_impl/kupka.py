import logging
from typing import Callable, TypeVar
from graphlib import TopologicalSorter

import pydot  # type: ignore

from kupka._impl.cache import KPCache
from kupka._impl.kupklass import Kupklass
from kupka._impl.settings import kp_settings
from kupka._impl.wrappers import WrappedValue

T = TypeVar("T")
_LOGGER = logging.getLogger(__name__)


class Kupka:
    _graph: dict[str, set[str]]
    _cache: KPCache
    _inputs: set[str]
    _func_map: dict[str, Callable]
    _input_map: dict[str, dict[str, str]]

    @classmethod
    def from_klass(cls, kupklass: Kupklass) -> "Kupka":
        return cls(
            cache=kp_settings.build_cache(),
            graph=kupklass.graph,
            inputs=kupklass.inputs,
            func_map={node: func for node, func in kupklass.func_map.items()},
            input_map=kupklass.input_map,
        )

    def __init__(
        self,
        cache: KPCache,
        graph: dict[str, set[str]],
        inputs: set[str],
        func_map: dict[str, Callable],
        input_map: dict[str, dict[str, str]],
    ) -> None:
        self._cache = cache
        self._graph = graph
        self._inputs = inputs
        self._func_map = func_map
        self._input_map = input_map

    def set_input(self, node: str, value: WrappedValue[T]) -> None:
        self._func_map[node] = WrappedValue.from_value(value)
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
        _LOGGER.debug(f"[EXECUTION GRAPH] {self.graph}")
        predecessors = [(name, p) for p in self.graph[name]]
        _LOGGER.debug(f"[EXECUTION GRAPH] {predecessors}")
        pruned_graph: dict[str, set[str]] = {name: set()}
        while predecessors:
            node, predecessor = predecessors.pop()
            if self.cache.has(predecessor):
                continue
            _LOGGER.debug(f"[EXECUTION GRAPH] Adding nodes: {node} -> {predecessor}")
            if node not in pruned_graph:
                pruned_graph[node] = set()
            pruned_graph[node].add(predecessor)
            predecessors += [(predecessor, p) for p in self.graph[predecessor]]
        _LOGGER.debug("[EXECUTION GRAPH] Pruned", pruned_graph)
        return TopologicalSorter(pruned_graph)

    def build_viz_graph(self, node: str) -> pydot.Dot:
        graph = pydot.Dot("my_graph", graph_type="digraph", rankdir="LR")
        for node in self.graph.keys():
            graph.add_node(pydot.Node(node, label=f"{node}", color="black", style="rounded"))
        for node, dependencies in self.graph.items():
            for dependency in dependencies:
                graph.add_edge(pydot.Edge(dependency, node, color="black"))
        return graph
