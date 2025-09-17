from dataclasses import dataclass, asdict
from enum import Enum
from dataclasses import asdict


class ChunkingMethod(Enum):
    MARKDOWN = "markdown"
    HYBRID = "markdown+semantic"
    SUMMARY = "summary"


@dataclass
class ChunkMetadata:
    """청킹 과정에서 추가되는 메타데이터를 위한 데이터클래스"""

    chunked_by: ChunkingMethod
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)
