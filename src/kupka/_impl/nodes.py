import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

from kupka._impl.kupka import Kupka
from kupka._impl.kupklass import Kupklass
from kupka._impl import graph_tools as gt
from kupka._impl.visualisation import (
    build_vis_graph,
    display_graph,
)
from kupka._impl.settings import kp_settings

T = TypeVar("T")
_LOGGER = logging.getLogger(__name__)


class KP(ABC):
    """The abstract class for all graphs."""

    __kupka__: Kupka

    def __init__(self, **inputs: Any):
        self.__kupka__ = Kupka.from_klass(cast(Kupklass, getattr(self.__class__, "__kupklass__")))
        for name, value in inputs.items():
            if name not in self.__kupka__.inputs:
                raise AttributeError()
            _LOGGER.debug(f"[INITIALIZATION] Setting {name} on KP instance")
            self.__kupka__.set_input(node=name, value=value)

    @staticmethod
    def get_kupklass(cls: "type[KP]") -> Kupklass:
        if not hasattr(cls, "__kupklass__"):
            setattr(
                cls,
                "__kupklass__",
                Kupklass(
                    graph={},
                    inputs=set(),
                    func_map={},
                    input_map={},
                ),
            )
        return cast(Kupklass, getattr(cls, "__kupklass__"))

    def _repr_png_(self) -> Any:
        graph = build_vis_graph(self.__kupka__.graph)
        return display_graph(graph)


class KPMember(ABC, Generic[T]):
    @property
    @abstractmethod
    def name(self) -> str: ...


# TODO: Use composition instead of inheritance
class KPNode(Kupka, Generic[T]):
    def __init__(self, node: str, kupka: Kupka) -> None:
        self._node = node
        super().__init__(
            cache=kupka._cache,
            graph=kupka._graph,
            inputs=kupka._inputs,
            func_map=kupka._func_map,
            input_map=kupka._input_map,
        )

    def __call__(self) -> T:
        if self.cache.has(self._node):
            _LOGGER.debug(f"[EXECUTION - CACHE] Found {self._node} in cache, not computing")
            return cast(T, self.cache.read(self._node))

        _LOGGER.debug(f"[EXECUTION - COMPUTATION] Did not find {self._node} in cache, computing...")
        exec = kp_settings.executor()

        return cast(T, exec(self._node, self))

    def _repr_png_(self) -> Any:
        subgraph = gt.get_subgraph(self.graph, self._node)
        vis_graph = build_vis_graph(subgraph)
        return display_graph(vis_graph)


class KPField(KPMember[T], Generic[T]):
    _func: Callable[..., T]
    _inputs: dict[str, KPMember[T]]
    _name: str | None

    def __init__(self, func: Callable[..., T], **inputs: KPMember[T]) -> None:
        _LOGGER.debug("[GRAPH INIT] Initializing KPField")
        self._func = func
        self._inputs = inputs
        self._name = None

    def __set_name__(self, owner: type[KP], name: str) -> None:
        _LOGGER.debug(f"[GRAPH INIT] setting name on KPField {name} - owner: {owner}")
        self._name = name
        _kupklass = KP.get_kupklass(owner)
        _kupklass.add_field(
            node=name,
            func=self._func,
            inputs={k: inp.name for k, inp in self._inputs.items()},
        )

    def __get__(self, owner: KP | None, owner_type: type[KP] | None = None) -> KPNode[T]:
        if owner is None:
            _LOGGER.debug(f"[GRAPH INIT] call on KPField class for {self.name}")
            return self  # type: ignore  # FIXME: use prototype to override behaviour between instance and class
        _LOGGER.debug(f"[EXECUTION] Getting on KPField {self.name} - owner: {owner}")
        return KPNode(self.name, owner.__kupka__)

    @property
    def name(self) -> str:
        if self._name is None:
            raise ValueError("KPField must be used as a descriptor")
        return self._name


class KPInput(KPMember[T], Generic[T]):
    _name: str | None

    def __init__(self) -> None:
        self._name = None

    def __set_name__(self, owner: type[KP], name: str) -> None:
        _LOGGER.debug(f"[GRAPH INIT] Setting name on KPInput {name} - owner: {owner}")
        self._name = name

        _kupklass = KP.get_kupklass(owner)
        _kupklass.add_input(node=name)

    def __get__(self, owner: KP | None, owner_type: type[KP] | None = None) -> KPNode[T]:
        if owner is None:
            _LOGGER.debug(f"[GRAPH INIT] call on KPInput class for {self.name}")
            return self  # type: ignore  # FIXME: use prototype to override behaviour between instance and class
        _LOGGER.debug(f"[EXECUTION] Getting on KPInput {self.name} - owner: {owner}")
        return KPNode(self.name, owner.__kupka__)

    @property
    def name(self) -> str:
        if self._name is None:
            raise ValueError("KPInput should be used as a descriptor")
        return self._name
