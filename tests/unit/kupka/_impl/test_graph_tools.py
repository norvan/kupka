from kupka._impl.graph_tools import (
    reverse,
    get_upstream,
    get_downstream,
    get_subgraph,
)
import pytest


@pytest.mark.parametrize(
    "graph, reversed",
    [
        ({}, {}),
        (
            {"a": {"b", "c"}},
            {"b": {"a"}, "c": {"a"}, "a": set()},
        ),
        (
            {"b": {"a"}, "c": {"a"}, "a": set()},
            {"a": {"b", "c"}, "b": set(), "c": set()},
        ),
    ]
)
def test_reverse(graph, reversed):
    assert reverse(graph) == reversed


@pytest.mark.parametrize(
    "graph, node, upstream",
    [
        ({}, "missing", {}),
        (
            {"a": {"b", "c"}},
            "a",
            {"a": {"b", "c"}, "b": set(), "c": set()},
        ),
        (
            {"a": {"b", "c"}}, "b", {},
        ),
        (
            {"b": {"a"}, "c": {"a"}, "a": set()},
            "b",
            {"b": {"a"}, "a": set()},
        ),
        (
            {
                "a": set(),
                "b": set(),
                "c": {"a", "b"},
                "d": {"b", "c"},
                "e": {"a", "d"},
                "f": {"e"}
            },
            "d",
            {
                "a": set(),
                "b": set(),
                "d": {"b", "c"},
                "c": {"a", "b"},
            },
        ),
    ]
)
def test_get_upstream(graph, node, upstream):
    assert get_upstream(graph, node) == upstream


@pytest.mark.parametrize(
    "graph, node, downstream",
    [
        (
            {
                "a": set(),
                "b": set(),
                "c": {"a", "b"},
                "d": {"b", "c"},
                "e": {"a", "d"},
                "f": {"e"}
            },
            "d",
            {
                "d": set(),
                "e": {"d"},
                "f": {"e"}
            },
        ),
    ]
)
def test_get_downstream(graph, node, downstream):
    assert get_downstream(graph, node) == downstream


@pytest.mark.parametrize(
    "graph, node, subgraph",
    [
        (
            {
                "a": set(),
                "b": set(),
                "c": {"a", "b"},
                "d": {"b", "c"},
                "e": {"a", "d"},
                "f": {"e"}
            },
            "d",
            {
                "a": set(),
                "b": set(),
                "c": {"a", "b"},
                "d": {"b", "c"},
                "e": {"d"},  # should this be: "e": {"a", "d"}, ?
                "f": {"e"}
            },
        ),
    ]
)
def test_get_subgraph(graph, node, subgraph):
    assert get_subgraph(graph, node) == subgraph
