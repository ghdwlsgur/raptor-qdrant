import pytest

from src.rag.builder.models.structure import Node, Tree


def make_node(index: int) -> Node:
    return Node(text=f"node-{index}", index=index, children=set(), embeddings={})


@pytest.fixture
def tree() -> Tree:
    leaves = {i: make_node(i) for i in range(3)}
    summaries = {3: make_node(3)}
    layer_to_nodes = {
        0: list(leaves.values()),
        1: list(summaries.values()),
    }
    return Tree(
        all_nodes={**leaves, **summaries},
        root_nodes=summaries,
        leaf_nodes=leaves,
        num_layers=1,
        layer_to_nodes=layer_to_nodes,
    )


def test_maps_every_node_to_its_layer(tree):
    assert [tree.get_node_layer(i) for i in range(4)] == [0, 0, 0, 1]


def test_unknown_node_has_no_layer(tree):
    assert tree.get_node_layer(99) is None


def test_lists_nodes_at_a_level(tree):
    assert len(tree.get_all_nodes_at_level(0)) == 3
    assert len(tree.get_all_nodes_at_level(1)) == 1
    assert tree.get_all_nodes_at_level(2) == []


def test_repr_reports_size(tree):
    assert repr(tree) == "Tree(num_layers=1, num_nodes=4)"
