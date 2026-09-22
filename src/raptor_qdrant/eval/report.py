"""평가 결과 집계.

숫자 하나로 줄이지 않는다. 정답을 물어왔는지(recall), 몇 번째로 물어왔는지
(MRR), 묶음 질문이면 몇 개나 덮었는지(coverage), 그 과정에서 컨텍스트를
어떻게 썼는지를 같이 본다. 서로 다른 것이 망가졌을 때 따로 움직인다.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueryOutcome:
    question: str
    sources: tuple[str, ...]
    kind: str = "note"
    # 정답을 문 첫 청크의 순위 (1 부터). 못 찾았으면 None
    hit_rank: int | None = None
    # 요약 말고 원문 청크로 찾은 순위
    leaf_hit_rank: int | None = None
    # 정답 노트 중 컨텍스트에 들어온 개수
    covered: int = 0
    summaries: int = 0
    tokens: int = 0
    chunks: int = 0

    @property
    def coverage(self) -> float:
        if not self.sources:
            return 0.0
        return self.covered / len(self.sources)


def outcome_of(
    question: str,
    sources: Sequence[str] | str,
    chunk_info: Sequence[Mapping[str, Any]],
    kind: str = "note",
) -> QueryOutcome:
    # 문자열 하나를 넘기면 글자 집합이 되어 조용히 다 맞는다
    wanted = {sources} if isinstance(sources, str) else set(sources)
    hit = leaf_hit = None
    seen: set[str] = set()

    for rank, chunk in enumerate(chunk_info, start=1):
        found = wanted.intersection(chunk.get("sources") or [])
        if not found:
            continue

        seen.update(found)
        if hit is None:
            hit = rank
        if leaf_hit is None and not chunk.get("layer_number"):
            leaf_hit = rank

    return QueryOutcome(
        question=question,
        sources=tuple(wanted) if isinstance(sources, str) else tuple(sources),
        kind=kind,
        hit_rank=hit,
        leaf_hit_rank=leaf_hit,
        covered=len(seen),
        summaries=sum(1 for c in chunk_info if c.get("layer_number")),
        tokens=sum(c.get("token_count") or 0 for c in chunk_info),
        chunks=len(chunk_info),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class EvalReport:
    outcomes: list[QueryOutcome] = field(default_factory=list)
    max_tokens: int = 4096

    @property
    def total(self) -> int:
        return len(self.outcomes)

    def recall_at(self, k: int) -> float:
        """상위 k 안에 정답이 들어온 비율.

        k 를 낮춰 보면 자가 포화됐는지 알 수 있다. recall@12 가 100% 여도
        recall@1 이 낮으면 아직 잴 여지가 있다는 뜻이다.
        """
        if not self.total:
            return 0.0
        hits = sum(1 for o in self.outcomes if o.hit_rank and o.hit_rank <= k)
        return hits / self.total

    @property
    def recall(self) -> float:
        return self.recall_at(10**6)

    @property
    def leaf_recall(self) -> float:
        """원문 청크로 들어온 비율.

        요약은 덮는 노트를 전부 달고 있어서 정답을 쉽게 스친다. 구체적인
        값을 묻는 질문에 답하려면 원문이 있어야 한다.
        """
        if not self.total:
            return 0.0
        return sum(1 for o in self.outcomes if o.leaf_hit_rank) / self.total

    @property
    def mrr(self) -> float:
        return _mean(
            [1 / o.hit_rank if o.hit_rank else 0.0 for o in self.outcomes]
        )

    @property
    def coverage(self) -> float:
        return _mean([o.coverage for o in self.outcomes])

    @property
    def budget_use(self) -> float:
        return _mean([o.tokens for o in self.outcomes]) / self.max_tokens

    @property
    def avg_summaries(self) -> float:
        return _mean([o.summaries for o in self.outcomes])

    @property
    def misses(self) -> list[QueryOutcome]:
        return [o for o in self.outcomes if not o.hit_rank]

    def of_kind(self, kind: str) -> "EvalReport":
        return EvalReport(
            outcomes=[o for o in self.outcomes if o.kind == kind],
            max_tokens=self.max_tokens,
        )

    def summary(self) -> str:
        return (
            f"질문 {self.total}개 | recall@1 {self.recall_at(1):.0%} "
            f"@3 {self.recall_at(3):.0%} @5 {self.recall_at(5):.0%} "
            f"전체 {self.recall:.0%} | MRR {self.mrr:.3f} | "
            f"coverage {self.coverage:.0%} | 원문 {self.leaf_recall:.0%} | "
            f"예산 {self.budget_use:.0%} | 요약 {self.avg_summaries:.1f}개"
        )


def score(outcomes: list[QueryOutcome], max_tokens: int) -> EvalReport:
    return EvalReport(outcomes=outcomes, max_tokens=max_tokens)
