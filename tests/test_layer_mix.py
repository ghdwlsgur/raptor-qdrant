import pytest

from raptor_qdrant.rag.retriever.layer_mix import balance_layers, is_summary


class FakeNode:
    def __init__(self, name: str, layer: int | None) -> None:
        self.name = name
        self.metadata = {"layer": layer}

    def __repr__(self) -> str:
        return self.name


def leaf(name: str) -> FakeNode:
    return FakeNode(name, 0)


def summary(name: str, layer: int = 1) -> FakeNode:
    return FakeNode(name, layer)


def names(nodes) -> list[str]:
    return [node.name for node in nodes]


def test_layer_zero_is_not_a_summary():
    assert not is_summary(leaf("a"))
    assert is_summary(summary("s"))
    assert not is_summary(FakeNode("unknown", None))


def test_summaries_get_their_slots_even_when_outranked():
    nodes = [
        leaf("l1"),
        leaf("l2"),
        leaf("l3"),
        leaf("l4"),
        leaf("l5"),
        summary("s1"),
        summary("s2"),
    ]

    assert names(balance_layers(nodes, top_k=5, summary_quota=2)) == [
        "l1",
        "l2",
        "l3",
        "s1",
        "s2",
    ]


def test_relevance_order_is_preserved():
    nodes = [summary("s1"), leaf("l1"), summary("s2"), leaf("l2"), leaf("l3")]

    assert names(balance_layers(nodes, top_k=4, summary_quota=2)) == [
        "s1",
        "l1",
        "s2",
        "l2",
    ]


def test_unused_summary_slots_go_back_to_leaves():
    nodes = [leaf("l1"), leaf("l2"), leaf("l3"), leaf("l4")]

    assert names(balance_layers(nodes, top_k=3, summary_quota=2)) == [
        "l1",
        "l2",
        "l3",
    ]


def test_a_single_summary_does_not_reserve_the_whole_quota():
    nodes = [leaf("l1"), leaf("l2"), leaf("l3"), leaf("l4"), summary("s1")]

    assert names(balance_layers(nodes, top_k=4, summary_quota=2)) == [
        "l1",
        "l2",
        "l3",
        "s1",
    ]


def test_quota_larger_than_top_k_is_clamped():
    nodes = [leaf("l1"), summary("s1"), summary("s2"), summary("s3")]

    assert names(balance_layers(nodes, top_k=2, summary_quota=5)) == [
        "s1",
        "s2",
    ]


def test_fewer_candidates_than_top_k():
    nodes = [leaf("l1"), summary("s1")]

    assert names(balance_layers(nodes, top_k=5, summary_quota=2)) == [
        "l1",
        "s1",
    ]


@pytest.mark.parametrize("top_k", [0, -1])
def test_no_room_means_nothing(top_k):
    assert balance_layers([leaf("l1")], top_k=top_k, summary_quota=2) == []


def test_zero_quota_keeps_the_old_behaviour():
    nodes = [leaf("l1"), leaf("l2"), summary("s1"), leaf("l3")]

    assert names(balance_layers(nodes, top_k=3, summary_quota=0)) == [
        "l1",
        "l2",
        "l3",
    ]
