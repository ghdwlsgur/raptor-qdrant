from llama_index.core.schema import TextNode

from fakes import (
    CountingEmbedding,
    FixedSummary,
    SinglesOnly,
    WholeNoteChunker,
)
from raptor_qdrant.rag.builder import tree_builder as tb
from raptor_qdrant.rag.builder.models.structure import Node
from raptor_qdrant.rag.builder.tree_builder import (
    LayerCallback,
    TreeBuilder,
    TreeBuilderConfig,
)
from raptor_qdrant.rag.constants import SOURCE_KEY


class LeavesAsRoots(TreeBuilder):
    """상위 레이어를 만들지 않는 빌더. 잎 생성 경로만 본다."""

    def construct_tree(
        self,
        all_tree_nodes: dict[int, Node],
        layer_to_nodes: dict[int, list[Node]],
        use_multithreading: bool = True,
        on_layer_built: LayerCallback | None = None,
    ) -> dict[int, Node]:
        return dict(all_tree_nodes)


def make_builder() -> tuple[LeavesAsRoots, CountingEmbedding]:
    embedding = CountingEmbedding()
    config = TreeBuilderConfig(
        summarization_model=FixedSummary(),
        embedding_models={"fake": embedding},
        cluster_embedding_model="fake",
        chunker=WholeNoteChunker(),
    )
    return LeavesAsRoots(config), embedding


def test_nodes_are_embedded_in_one_batch():
    builder, embedding = make_builder()

    nodes = builder.create_nodes(
        0, [TextNode(text=f"청크 {i}") for i in range(5)]
    )

    assert len(nodes) == 5
    assert embedding.batches == [5]
    assert embedding.singles == 0


def test_batches_are_capped_at_the_configured_size(monkeypatch):
    monkeypatch.setattr(tb, "EMBEDDING_BATCH_SIZE", 2)
    builder, embedding = make_builder()

    builder.create_nodes(0, [TextNode(text=f"청크 {i}") for i in range(5)])

    assert embedding.batches == [2, 2, 1]


def test_existing_embeddings_are_not_recomputed():
    builder, embedding = make_builder()
    ready = TextNode(text="이미 임베딩됨", embedding=[9.0, 9.0])
    fresh = TextNode(text="새 청크")

    nodes = builder.create_nodes(0, [ready, fresh])

    assert nodes[0].embeddings["fake"] == [9.0, 9.0]
    assert embedding.batches == [1]


def test_children_and_indices_follow_the_start_index():
    builder, _ = make_builder()

    nodes = builder.create_nodes(
        10,
        [TextNode(text="a"), TextNode(text="b")],
        children=[{1, 2}, {3}],
    )

    assert sorted(nodes) == [10, 11]
    assert nodes[10].children == {1, 2}
    assert nodes[11].index == 11


def test_children_must_line_up_with_nodes():
    builder, _ = make_builder()

    try:
        builder.create_nodes(0, [TextNode(text="a")], children=[])
    except ValueError as e:
        assert "line up" in str(e)
    else:
        raise AssertionError("mismatched children were accepted")


def test_single_node_helper_uses_the_batch_path():
    builder, embedding = make_builder()

    index, node = builder.create_node(3, TextNode(text="하나"), {1})

    assert (index, node.index, node.children) == (3, 3, {1})
    assert embedding.batches == [1]


def test_tree_shares_leaf_objects_instead_of_deep_copies():
    builder, _ = make_builder()

    tree = builder.build_from_documents({"a.md": "가", "b.md": "나"})

    assert tree.all_nodes[0] is tree.leaf_nodes[0]
    assert tree.all_nodes[0].metadata[SOURCE_KEY] == "a.md"
    assert tree.get_node_layer(1) == 0


def test_leaf_layer_is_reported_to_the_callback():
    builder, _ = make_builder()
    seen: list[tuple[int, int]] = []

    builder.build_from_documents(
        {"a.md": "가", "b.md": "나", "c.md": "다"},
        on_layer_built=lambda layer, nodes: seen.append((layer, len(nodes))),
    )

    assert seen == [(0, 3)]


def test_base_model_batches_by_looping_over_singles():
    assert SinglesOnly().create_embeddings(["a", "bb"]) == [[1.0], [2.0]]
