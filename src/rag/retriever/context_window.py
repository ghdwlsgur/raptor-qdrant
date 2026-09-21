from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from src.rag.utils import resolve_token_count

CHUNK_SEPARATOR = "\n\n"


@dataclass
class ContextWindow:
    chunks: List[str] = field(default_factory=list)
    chunk_info: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    skipped: List[Tuple[Any, int]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return CHUNK_SEPARATOR.join(self.chunks)

    @property
    def is_empty(self) -> bool:
        return not self.chunks


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
            }
        )

    return window
