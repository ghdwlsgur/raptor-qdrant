from dataclasses import asdict, dataclass
from enum import StrEnum


class ChunkingMethod(StrEnum):
    MARKDOWN = "markdown"
    HYBRID = "markdown+semantic"
    SUMMARY = "summary"


@dataclass
class ChunkMetadata:
    chunked_by: ChunkingMethod
    token_count: int

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "chunked_by": ChunkingMethod(self.chunked_by).value,
        }
