from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from raptor_qdrant.rag.constants import SOURCE_KEY, SOURCE_SET_KEY
from raptor_qdrant.rag.utils import count_tokens, resolve_token_count

CHUNK_SEPARATOR = "\n\n"
# 라벨에 이름을 다 적으면 요약 하나가 노트 스무 개를 늘어놓는다
MAX_NAMED_SOURCES = 3


def _sources_of(metadata: dict[str, Any]) -> list[str]:
    covered = metadata.get(SOURCE_SET_KEY)
    if covered:
        return list(covered)
    source = metadata.get(SOURCE_KEY)
    return [source] if source else []


def ordered_sources(chunk_info: Iterable[Mapping[str, Any]]) -> list[str]:
    """청크가 문 출처를 관련도 순서대로 중복 없이 편다."""
    ordered: dict[str, None] = {}
    for info in chunk_info:
        for source in info.get("sources") or []:
            ordered.setdefault(source, None)
    return list(ordered)


def describe_origin(sources: list[str]) -> str:
    if not sources:
        return "출처 미상"
    if len(sources) == 1:
        return sources[0]

    named = ", ".join(sources[:MAX_NAMED_SOURCES])
    if len(sources) > MAX_NAMED_SOURCES:
        named += f" 외 {len(sources) - MAX_NAMED_SOURCES}개"
    return f"{len(sources)}개 노트 ({named})"


def label_for(number: int, info: Mapping[str, Any]) -> str:
    """LLM 이 어느 근거를 인용했는지 말할 수 있도록 머리표를 붙인다.

    본문만 이어 붙이면 모델은 무엇이 원문이고 무엇이 요약인지, 어느 노트에서
    왔는지 모른 채 답한다. 요약은 자기가 덮는 노트를 전부 달고 있어서 특정
    주장의 직접 근거로 쓰면 안 되는데, 그 구분도 본문만으로는 전할 수 없다.
    """
    kind = "요약" if (info.get("layer_number") or 0) else "원문"
    return f"[근거 {number} | {kind} | {describe_origin(info.get('sources') or [])}]"


@dataclass
class ContextWindow:
    chunks: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    chunk_info: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    skipped: list[tuple[Any, int]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return CHUNK_SEPARATOR.join(self.blocks)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def sources(self) -> list[str]:
        """컨텍스트에 쓰인 원본 문서 이름을 관련도 순서대로 중복 없이 반환한다."""
        return ordered_sources(self.chunk_info)


def _info_of(node: Any) -> dict[str, Any]:
    return {
        "node_index": node.metadata.get("node_index"),
        "layer_number": node.metadata.get("layer"),
        "chunked_by": node.metadata.get("chunked_by"),
        "token_count": resolve_token_count(node.metadata, node.text),
        "score": getattr(node, "score", 0.0) or 0.0,
        "sources": _sources_of(node.metadata),
    }


def assemble_context(nodes: Iterable[Any], max_tokens: int) -> ContextWindow:
    """관련도 순서를 유지한 채 예산이 허용하는 노드만 모은다.

    예산을 넘는 노드는 건너뛴다. 하나가 크다는 이유로 뒤따르는 작은 노드까지
    버리면 컨텍스트가 통째로 비어버린다. 같은 본문이 두 번 들어오면 두 번째는
    버린다. 검색이 같은 글을 잎과 요약 양쪽에서 물어오는 일이 있는데, 그때
    예산만 두 번 쓰고 모델에게 주는 정보는 늘지 않는다.
    """
    window = ContextWindow()

    for node in nodes:
        info = _info_of(node)
        body = node.text
        if body in window.chunks:
            continue

        label = label_for(len(window.chunks) + 1, info)
        cost = info["token_count"] + count_tokens(label)

        if window.total_tokens + cost > max_tokens:
            window.skipped.append((info["node_index"], info["token_count"]))
            continue

        window.chunks.append(body)
        window.blocks.append(f"{label}\n{body}")
        window.total_tokens += cost
        window.chunk_info.append(info)

    return window
