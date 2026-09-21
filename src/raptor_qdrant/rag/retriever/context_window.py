from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from raptor_qdrant.rag.constants import SOURCE_KEY, SOURCE_SET_KEY
from raptor_qdrant.rag.utils import resolve_token_count

CHUNK_SEPARATOR = "\n\n"


def _sources_of(metadata: dict[str, Any]) -> list[str]:
    covered = metadata.get(SOURCE_SET_KEY)
    if covered:
        return list(covered)
    source = metadata.get(SOURCE_KEY)
    return [source] if source else []


@dataclass
class ContextWindow:
    chunks: list[str] = field(default_factory=list)
    chunk_info: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    skipped: list[tuple[Any, int]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return CHUNK_SEPARATOR.join(self.chunks)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def sources(self) -> list[str]:
        """컨텍스트에 쓰인 원본 문서 이름을 관련도 순서대로 중복 없이 반환한다."""
        ordered: dict[str, None] = {}
        for info in self.chunk_info:
            for source in info.get("sources") or []:
                ordered.setdefault(source, None)
        return list(ordered)


def assemble_context(nodes: Iterable[Any], max_tokens: int) -> ContextWindow:
    """관련도 순서를 유지한 채 예산이 허용하는 노드만 모은다.

    예산을 넘는 노드는 건너뛴다. 하나가 크다는 이유로 뒤따르는 작은 노드까지
    버리면 컨텍스트가 통째로 비어버린다.
    """
    window = ContextWindow()

    for node in nodes:
        tokens = resolve_token_count(node.metadata, node.text)

        if window.total_tokens + tokens > max_tokens:
            window.skipped.append((node.metadata.get("node_index"), tokens))
            continue

        window.chunks.append(node.text)
        window.total_tokens += tokens
        window.chunk_info.append(
            {
                "node_index": node.metadata.get("node_index"),
                "layer_number": node.metadata.get("layer"),
                "chunked_by": node.metadata.get("chunked_by"),
                "token_count": tokens,
                "score": getattr(node, "score", 0.0) or 0.0,
                "sources": _sources_of(node.metadata),
            }
        )

    return window
