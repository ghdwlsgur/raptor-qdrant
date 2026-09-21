from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """컨텍스트 문자열과 그것을 구성한 청크 정보를 반환한다."""
