from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pydot  # type: ignore


def get_node_style(node: str) -> dict[str, str]:
    return {
        "label": node,
        "color": "black",
        #"style": "rounded",
        "shape": "rect",
    }


def build_vis_graph(graph: dict[str, set[str]]) -> "pydot.Dot":
    import pydot  # type: ignore

    vis_graph = pydot.Dot("my_graph", graph_type="digraph", rankdir="LR")
    for node in graph.keys():
        vis_graph.add_node(pydot.Node(node, **get_node_style(node)))
    for node, dependencies in graph.items():
        for dependency in dependencies:
            vis_graph.add_edge(pydot.Edge(dependency, node, color="black"))
    return vis_graph


def display_graph(graph: "pydot.Dot") -> Any:
    from IPython.display import SVG, display  # type: ignore
    return display(SVG(graph.create_svg()))
