from collections.abc import Callable


class Kupklass:
    _graph: dict[str, set[str]]
    _inputs: set[str]
    _func_map: dict[str, Callable]
    _input_map: dict[str, dict[str, str]]

    def __init__(
        self,
        graph: dict[str, set[str]],
        inputs: set[str],
        func_map: dict[str, Callable],
        input_map: dict[str, dict[str, str]],
    ) -> None:
        self._graph = graph
        self._inputs = inputs
        self._func_map = func_map
        self._input_map = input_map

    def add_field(self, node: str, func: Callable, inputs: dict[str, str]) -> None:
        self._graph[node] = set(inputs.values())
        self._func_map[node] = func
        self._input_map[node] = inputs

    def add_input(self, node: str) -> None:
        self._graph[node] = set()
        self._input_map[node] = {}
        self._inputs.add(node)

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
