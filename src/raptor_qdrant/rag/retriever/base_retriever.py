from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        collapse_tree: bool = True,
        start_layer: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """컨텍스트 문자열과 그것을 구성한 청크 정보를 반환한다."""
