from llama_index.core.schema import TextNode

from raptor_qdrant.rag.chunker.hybrid_chunker import merge_small_chunks
from raptor_qdrant.rag.utils import TOKEN_COUNT_KEY, count_tokens


def node(text: str) -> TextNode:
    return TextNode(text=text, metadata={TOKEN_COUNT_KEY: count_tokens(text)})


BODY = "본문이 충분히 길어서 그대로 남아야 하는 조각이다. " * 6
OTHER = "다른 절의 본문도 충분히 길게 적어 둔다. " * 6


def test_a_heading_only_chunk_joins_the_next_one():
    merged = merge_small_chunks([node("## 제목만"), node(BODY)])

    assert len(merged) == 1
    assert merged[0].get_content().startswith("## 제목만")
    assert BODY.strip() in merged[0].get_content()


def test_the_merged_chunk_gets_a_fresh_token_count():
    merged = merge_small_chunks([node("## 제목만"), node(BODY)])

    assert merged[0].metadata[TOKEN_COUNT_KEY] == count_tokens(
        merged[0].get_content()
    )


def test_a_trailing_small_chunk_joins_the_previous_one():
    merged = merge_small_chunks([node(BODY), node("## 꼬리 제목")])

    assert len(merged) == 1
    assert merged[0].get_content().endswith("## 꼬리 제목")


def test_big_chunks_are_left_alone():
    merged = merge_small_chunks([node(BODY), node(OTHER)])

    assert [n.get_content() for n in merged] == [BODY, OTHER]


def test_a_note_made_only_of_small_chunks_survives_as_one():
    merged = merge_small_chunks([node("# 제목"), node("한 줄 메모")])

    assert len(merged) == 1
    assert "제목" in merged[0].get_content()
    assert "한 줄 메모" in merged[0].get_content()


def test_several_small_chunks_pile_onto_the_next_body():
    merged = merge_small_chunks(
        [node("# 하나"), node("## 둘"), node("### 셋"), node(BODY)]
    )

    assert len(merged) == 1
    content = merged[0].get_content()
    assert (
        content.index("# 하나")
        < content.index("## 둘")
        < content.index("### 셋")
    )


def test_no_chunks_stays_empty():
    assert merge_small_chunks([]) == []


def test_merging_stops_at_the_token_cap():
    big = "상한에 거의 닿는 본문이다. " * 40
    merged = merge_small_chunks(
        [node("## 제목만"), node(big)], min_tokens=32, max_tokens=60
    )

    assert len(merged) == 2
    assert merged[0].get_content() == "## 제목만"
    assert merged[1].get_content() == big


def test_a_trailing_small_chunk_stays_out_when_it_would_overflow():
    big = "상한에 거의 닿는 본문이다. " * 40
    merged = merge_small_chunks(
        [node(big), node("## 꼬리")], min_tokens=32, max_tokens=60
    )

    assert [n.get_content() for n in merged] == [big, "## 꼬리"]


def test_an_oversized_chunk_is_split_under_the_cap():
    from raptor_qdrant.rag.chunker.hybrid_chunker import split_oversized

    long_note = "\n".join(
        f"{i}번째 줄에 적어 둔 내용이다." for i in range(200)
    )
    pieces = split_oversized([node(long_note)], max_tokens=200)

    assert len(pieces) > 1
    assert all(count_tokens(p.get_content()) <= 200 for p in pieces)
    assert all(
        p.metadata[TOKEN_COUNT_KEY] == count_tokens(p.get_content())
        for p in pieces
    )


def test_splitting_keeps_every_line():
    from raptor_qdrant.rag.chunker.hybrid_chunker import split_oversized

    long_note = "\n".join(f"{i}번째 줄" for i in range(120))
    pieces = split_oversized([node(long_note)], max_tokens=100)

    joined = "\n".join(p.get_content() for p in pieces)
    assert "0번째 줄" in joined and "119번째 줄" in joined


def test_a_chunk_under_the_cap_is_untouched():
    from raptor_qdrant.rag.chunker.hybrid_chunker import split_oversized

    pieces = split_oversized([node(BODY)], max_tokens=4096)

    assert [p.get_content() for p in pieces] == [BODY]


def test_semantic_splitting_is_off_by_default():
    from fakes import CountingEmbedding
    from raptor_qdrant.rag.chunker.hybrid_chunker import HybridChunker

    assert not HybridChunker(CountingEmbedding()).semantic


def test_the_setting_turns_semantic_splitting_on(monkeypatch):
    from fakes import CountingEmbedding
    from raptor_qdrant.core.config import settings
    from raptor_qdrant.rag.chunker.hybrid_chunker import HybridChunker

    monkeypatch.setattr(settings, "SEMANTIC_CHUNKING", True)

    assert HybridChunker(CountingEmbedding()).semantic
    assert not HybridChunker(CountingEmbedding(), semantic=False).semantic


def test_a_long_section_is_cut_by_length_without_the_embedder():
    from fakes import CountingEmbedding
    from raptor_qdrant.rag.chunker.hybrid_chunker import HybridChunker

    model = CountingEmbedding()
    long_note = (
        "# 제목\n\n" + "이 문장은 길이 기준 분할을 확인하려고 적는다. " * 120
    )

    nodes = HybridChunker(model, max_tokens=200).chunk(long_note)

    assert len(nodes) > 1
    assert all(count_tokens(n.get_content()) <= 200 for n in nodes)
    # 의미 분할을 껐으므로 임베딩을 한 번도 부르지 않는다
    assert model.batches == [] and model.singles == 0
