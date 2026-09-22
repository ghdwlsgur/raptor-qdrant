from collections.abc import Sequence
from typing import Any

from raptor_qdrant.rag.constants import LAYER_KEY, LEAF_LAYER


def is_summary(node: Any) -> bool:
    layer = node.metadata.get(LAYER_KEY)
    return layer is not None and layer > LEAF_LAYER


def balance_layers(
    nodes: Sequence[Any], top_k: int, summary_quota: int
) -> list[Any]:
    """관련도 순서를 지키되 상위 top_k 안에 요약 노드 자리를 남긴다.

    포인트의 대부분이 잎이라(볼트 하나에서 1,982 대 351) 후보를 점수순으로
    그냥 자르면 개괄 질문에도 잎만 남는다. 실측으로는 최고 잎 0.587 대 최고
    요약 0.539 로 차이가 근소한데 그 근소한 차이로 요약이 전부 밀려났다.
    자리를 남겨 둬야 "이 주제로 내가 남긴 것 전반" 같은 질문이 요약을 본다.

    쿼터를 채울 요약이 후보에 없으면 그 자리는 잎으로 메운다. 없는 요약을
    억지로 끌어오지는 않는다.
    """
    if top_k <= 0:
        return []

    summary_slots = min(summary_quota, top_k)
    leaf_slots = top_k - summary_slots

    chosen: set[int] = set()
    leaves = summaries = 0
    for i, node in enumerate(nodes):
        if is_summary(node):
            if summaries < summary_slots:
                chosen.add(i)
                summaries += 1
        elif leaves < leaf_slots:
            chosen.add(i)
            leaves += 1

    for i in range(len(nodes)):
        if len(chosen) >= top_k:
            break
        chosen.add(i)

    return [node for i, node in enumerate(nodes) if i in chosen]
