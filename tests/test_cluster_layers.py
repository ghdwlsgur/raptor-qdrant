import pytest

from fakes import CountingEmbedding, RecordingSummary, WholeNoteChunker
from raptor_qdrant.rag.builder.cluster.cluster_builder import (
    ClusterTreeBuilder,
    ClusterTreeConfig,
)
from raptor_qdrant.rag.builder.cluster.raptor_clustering import (
    ClusteringAlgorithm,
)
from raptor_qdrant.rag.builder.models.structure import Node
from raptor_qdrant.rag.constants import SOURCE_SET_KEY
from raptor_qdrant.rag.summarizer import BaseSummarizationModel


class PairClustering(ClusteringAlgorithm):
    """노드를 인덱스 순서대로 둘씩 묶는다. 결과가 결정적이라 세기 쉽다."""

    def __init__(self, **_: object) -> None:
        pass

    def perform_clustering(
        self, nodes: list[Node], embedding_model_name: str
    ) -> list[list[Node]]:
        ordered = sorted(nodes, key=lambda node: node.index)
        return [ordered[i : i + 2] for i in range(0, len(ordered), 2)]


NOTES = {f"note-{i}.md": f"note-{i} 본문" for i in range(8)}


def build(
    summarizer: BaseSummarizationModel,
    parallel: bool = True,
    seen: list[tuple[int, int]] | None = None,
) -> tuple[object, CountingEmbedding]:
    embedding = CountingEmbedding()
    config = ClusterTreeConfig(
        reduction_dimension=1,  # 노드 2개 이하가 되면 멈춘다
        clustering_algorithm=PairClustering,
        summarization_model=summarizer,
        embedding_models={"fake": embedding},
        cluster_embedding_model="fake",
        chunker=WholeNoteChunker(),
        summarization_max_workers=2,
    )
    tree = ClusterTreeBuilder(config).build_from_documents(
        NOTES,
        use_multithreading=parallel,
        on_layer_built=(
            (lambda layer, nodes: seen.append((layer, len(nodes))))
            if seen is not None
            else None
        ),
    )
    return tree, embedding


@pytest.mark.parametrize("parallel", [True, False])
def test_every_cluster_is_summarized_exactly_once(parallel):
    """8잎 → 4요약 → 2요약 → 멈춤. 요약 호출은 정확히 6회다."""
    summarizer = RecordingSummary()

    tree, _ = build(summarizer, parallel=parallel)

    assert len(summarizer.calls) == 6
    assert tree.num_layers == 2
    assert [len(tree.layer_to_nodes[i]) for i in range(3)] == [8, 4, 2]


def test_a_failed_cluster_is_retried_alone():
    """실패한 클러스터만 다시 돈다. 성공한 요약은 버리지 않는다."""
    summarizer = RecordingSummary(fail_once_on="note-3")

    tree, _ = build(summarizer)

    assert len(summarizer.calls) == 7
    assert len(tree.layer_to_nodes[1]) == 4


def test_unusable_summaries_drop_only_their_cluster():
    summarizer = RecordingSummary(unusable_on="note-5")

    tree, _ = build(summarizer)

    layer_one = tree.layer_to_nodes[1]
    assert len(layer_one) == 3
    assert {4, 5} not in [node.children for node in layer_one]


def test_children_and_sources_follow_the_cluster():
    tree, _ = build(RecordingSummary())

    first = tree.layer_to_nodes[1][0]
    assert first.children == {0, 1}
    assert first.metadata[SOURCE_SET_KEY] == ["note-0.md", "note-1.md"]

    top = tree.layer_to_nodes[2][0]
    assert sorted(top.metadata[SOURCE_SET_KEY]) == [
        f"note-{i}.md" for i in range(4)
    ]


def test_summary_indices_continue_after_the_leaves():
    tree, _ = build(RecordingSummary())

    assert sorted(node.index for node in tree.layer_to_nodes[1]) == [
        8,
        9,
        10,
        11,
    ]
    assert set(tree.all_nodes) == set(range(14))


def test_each_layer_is_embedded_as_one_batch():
    _, embedding = build(RecordingSummary())

    assert embedding.batches == [8, 4, 2]
    assert embedding.singles == 0


def test_layers_are_reported_in_order():
    seen: list[tuple[int, int]] = []

    build(RecordingSummary(), seen=seen)

    assert seen == [(0, 8), (1, 4), (2, 2)]


def test_stops_when_no_summary_is_usable():
    tree, _ = build(RecordingSummary(unusable_on="note-"))

    assert tree.num_layers == 0
    assert list(tree.layer_to_nodes) == [0]
