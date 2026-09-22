from dataclasses import dataclass, field
from typing import Any

import pytest

from raptor_qdrant.rag.retriever.context_window import assemble_context


@dataclass
class FakeNode:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


def node(text: str, tokens: int, index: int = 0, **extra) -> FakeNode:
    return FakeNode(
        text=text,
        metadata={"token_count": tokens, "node_index": index, **extra},
        score=extra.get("score", 0.5),
    )


def test_collects_nodes_until_the_budget_is_reached():
    window = assemble_context(
        [node("a", 100, 0), node("b", 100, 1), node("c", 100, 2)],
        max_tokens=250,
    )

    assert window.chunks == ["a", "b"]
    # 근거 머리표도 예산을 쓴다. 본문 200 토큰에 라벨 두 개가 더 붙는다
    assert 200 < window.total_tokens <= 250
    assert window.skipped == [(2, 100)]


def test_a_single_oversized_node_does_not_drop_the_rest():
    """이 케이스가 검색 결과를 통째로 비게 만들던 회귀다."""
    window = assemble_context(
        [node("huge", 5000, 0), node("small", 100, 1)],
        max_tokens=4096,
    )

    assert window.chunks == ["small"]
    assert window.skipped == [(0, 5000)]
    assert not window.is_empty


def test_reports_empty_when_nothing_fits():
    window = assemble_context([node("huge", 5000, 0)], max_tokens=4096)

    assert window.is_empty
    assert window.text == ""
    assert window.chunk_info == []


def test_each_chunk_carries_a_numbered_label():
    window = assemble_context(
        [
            node("a", 1, 0, document_name="메모/one.md"),
            node("b", 1, 1, layer=2, source_notes=["x.md", "y.md"]),
        ],
        max_tokens=100,
    )

    blocks = window.text.split("\n\n")
    assert blocks[0] == "[근거 1 | 원문 | 메모/one.md]\na"
    assert blocks[1] == "[근거 2 | 요약 | 2개 노트 (x.md, y.md)]\nb"


def test_the_same_text_is_not_sent_twice():
    window = assemble_context(
        [node("같은 본문", 10, 0), node("같은 본문", 10, 1)], max_tokens=500
    )

    assert window.chunks == ["같은 본문"]


def test_counts_tokens_from_text_when_metadata_lacks_them():
    orphan = FakeNode(text="신데렐라", metadata={"node_index": 7})

    window = assemble_context([orphan], max_tokens=4096)

    assert window.chunks == ["신데렐라"]
    assert window.total_tokens > 0


def test_carries_layer_and_method_into_chunk_info():
    window = assemble_context(
        [node("a", 10, 3, layer=1, chunked_by="summary")], max_tokens=100
    )

    assert window.chunk_info == [
        {
            "node_index": 3,
            "layer_number": 1,
            "chunked_by": "summary",
            "token_count": 10,
            "score": 0.5,
            "sources": [],
        }
    ]


@pytest.mark.parametrize("score", [None, 0.0])
def test_missing_score_becomes_zero(score):
    n = node("a", 10, 0)
    n.score = score

    assert assemble_context([n], max_tokens=100).chunk_info[0]["score"] == 0.0


def test_no_nodes_yields_an_empty_window():
    window = assemble_context([], max_tokens=4096)

    assert (
        window.is_empty and window.total_tokens == 0 and window.skipped == []
    )


def test_leaf_node_reports_its_source_note():
    leaf = node("a", 10, 0, document_name="메모/정리.md")

    assert assemble_context([leaf], max_tokens=100).sources == ["메모/정리.md"]


def test_summary_node_reports_every_note_it_covers():
    summary = node("s", 10, 1, source_notes=["b.md", "a.md"])

    assert assemble_context([summary], max_tokens=100).sources == [
        "b.md",
        "a.md",
    ]


def test_sources_are_deduplicated_in_relevance_order():
    window = assemble_context(
        [
            node("a", 10, 0, source_notes=["a.md", "b.md"]),
            node("b", 10, 1, document_name="b.md"),
            node("c", 10, 2, document_name="c.md"),
        ],
        max_tokens=100,
    )

    assert window.sources == ["a.md", "b.md", "c.md"]


def test_no_sources_when_metadata_has_none():
    assert assemble_context([node("a", 10, 0)], max_tokens=100).sources == []
