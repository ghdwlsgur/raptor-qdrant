from llama_index.core.schema import TextNode

from raptor_qdrant.rag.chunker.hybrid_chunker import tag_chunk
from raptor_qdrant.rag.chunker.models.chunk_metadata import ChunkingMethod
from raptor_qdrant.rag.utils import count_tokens

SHORT = "짧은 청크."
LONG = "신데렐라는 재 속에서 렌틸콩을 골라냈다. " * 40


def test_counts_each_chunk_separately():
    """부모 섹션의 토큰 수를 물려받으면 검색 예산 계산이 무너진다."""
    section_tokens = count_tokens(LONG)

    tagged = [
        tag_chunk(TextNode(text=SHORT), {}, ChunkingMethod.HYBRID),
        tag_chunk(TextNode(text=LONG), {}, ChunkingMethod.HYBRID),
    ]
    counts = [node.metadata["token_count"] for node in tagged]

    assert counts == [count_tokens(SHORT), section_tokens]
    assert counts[0] != counts[1]


def test_explicit_token_count_wins():
    node = tag_chunk(
        TextNode(text=SHORT), {}, ChunkingMethod.MARKDOWN, token_count=999
    )

    assert node.metadata["token_count"] == 999


def test_keeps_inherited_metadata():
    node = tag_chunk(
        TextNode(text=SHORT), {"header_path": "/1장"}, ChunkingMethod.MARKDOWN
    )

    assert node.metadata["header_path"] == "/1장"
    assert node.metadata["chunked_by"] == "markdown"


def test_chunk_metadata_overrides_inherited_keys():
    node = tag_chunk(
        TextNode(text=SHORT),
        {"chunked_by": "stale", "token_count": 1},
        ChunkingMethod.SUMMARY,
    )

    assert node.metadata["chunked_by"] == "summary"
    assert node.metadata["token_count"] == count_tokens(SHORT)


def test_does_not_mutate_the_inherited_dict():
    inherited = {"header_path": "/1장"}

    tag_chunk(TextNode(text=SHORT), inherited, ChunkingMethod.MARKDOWN)

    assert inherited == {"header_path": "/1장"}
