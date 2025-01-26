from typing import Iterator


Graph = dict[str, set[str]]


def _traverse(graph: Graph, node: str) -> Iterator[tuple[str, set[str]]]:
    to_visit = list(graph[node])
    visited = set()

    while to_visit:
        visiting = to_visit.pop()
        if visiting in visited:
            continue
        visited.add(visiting)
        if visiting in graph:  # might not be required
            to_visit += list(graph[visiting])
        yield visiting, graph.get(visiting, set())


def reverse(graph: Graph) -> Graph:
    new_graph: Graph = {}
    for parent, children in graph.items():
        if parent not in new_graph:
            new_graph[parent] = set()
        for child in children:
            if child not in new_graph:
                new_graph[child] = set()
            new_graph[child].add(parent)

    return new_graph


def get_upstream(graph: Graph, node: str) -> Graph:
    if node not in graph:
        return {}
    new_graph: Graph = {node: graph[node]}

    for parent, children in _traverse(graph, node):
        new_graph[parent] = children

    return new_graph


def get_downstream(graph: Graph, node: str) -> Graph:
    reversed = reverse(graph)
    downstream_reversed = get_upstream(reversed, node)
    downstream = reverse(downstream_reversed)
    return downstream


def get_subgraph(graph: Graph, node: str) -> Graph:
    upstream = get_upstream(graph, node)
    downstream = get_downstream(graph, node)
    return {**downstream, **upstream}

