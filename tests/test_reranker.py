from raptor_qdrant.rag.retriever.reranker import (
    BaseReranker,
    default_reranker,
    rerank,
)


class FakeNode:
    def __init__(self, text: str, score: float = 0.0) -> None:
        self.text = text
        self.score = score
        self.metadata: dict = {"layer": 0}

    def __repr__(self) -> str:
        return self.text


class ByLength(BaseReranker):
    """긴 글일수록 높은 점수. 순서가 실제로 바뀌는지만 본다."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []

    def scores(self, query: str, documents) -> list[float]:
        self.seen.append((query, len(documents)))
        return [float(len(document)) for document in documents]


def test_candidates_come_back_in_the_new_order():
    nodes = [
        FakeNode("짧다"),
        FakeNode("아주 많이 긴 문장이다"),
        FakeNode("중간이다"),
    ]

    ranked = rerank("질문", nodes, ByLength())

    assert [n.text for n in ranked] == [
        "아주 많이 긴 문장이다",
        "중간이다",
        "짧다",
    ]


def test_the_reported_score_becomes_the_rerank_score():
    nodes = [
        FakeNode("짧다", score=0.9),
        FakeNode("더 긴 문장이다", score=0.1),
    ]

    ranked = rerank("질문", nodes, ByLength())

    assert ranked[0].score == len("더 긴 문장이다")
    assert ranked[1].score == len("짧다")


def test_every_candidate_is_scored_in_one_call():
    reranker = ByLength()

    rerank("질문", [FakeNode("가"), FakeNode("나"), FakeNode("다")], reranker)

    assert reranker.seen == [("질문", 3)]


def test_no_candidates_means_no_model_call():
    reranker = ByLength()

    assert rerank("질문", [], reranker) == []
    assert reranker.seen == []


def test_an_empty_setting_turns_reranking_off(monkeypatch):
    from raptor_qdrant.core.config import settings

    monkeypatch.setattr(settings, "RERANKER_MODEL", "")
    assert default_reranker() is None

    monkeypatch.setattr(settings, "RERANKER_MODEL", "some/model")
    assert default_reranker() is not None


def test_only_the_head_is_rescored_when_a_limit_is_given():
    reranker = ByLength()
    nodes = [
        FakeNode("가"),
        FakeNode("아주 많이 긴 문장이다"),
        FakeNode("나"),
        FakeNode("다"),
    ]

    ranked = rerank("질문", nodes, reranker, limit=2)

    assert reranker.seen == [("질문", 2)]
    assert [n.text for n in ranked] == [
        "아주 많이 긴 문장이다",
        "가",
        "나",
        "다",
    ]


def test_a_limit_larger_than_the_candidates_scores_them_all():
    reranker = ByLength()

    ranked = rerank(
        "질문", [FakeNode("가"), FakeNode("나다")], reranker, limit=10
    )

    assert reranker.seen == [("질문", 2)]
    assert [n.text for n in ranked] == ["나다", "가"]
