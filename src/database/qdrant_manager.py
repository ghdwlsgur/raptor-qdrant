import threading
import logging
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient, models
from src.core.config import settings
from .constants import create_vector_config, get_sparse_vector_config
from src.rag.constants import MAX_SCROLL_LIMIT


class QdrantManager:
    _instance: Optional['QdrantManager'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.client: Optional[QdrantClient] = None
        self.logger = logging.getLogger(__name__)
        self._initialized: bool = True
        self.logger.info("created qdrant manager instance")

    def connect(
        self, host: str = settings.QDRANT_HOST, port: int = settings.QDRANT_PORT
    ) -> QdrantClient:
        if self.client:
            self.logger.info("qdrant client already connected")
            return self.client

        try:
            self.logger.info(f"trying to connect qdrant at {host}:{port}")
            self.client = QdrantClient(host=host, port=port)
            self.logger.info("qdrant client connected successfully")
            return self.client
        except Exception as e:
            self.logger.error(f"qdrant connection failed: {e}")
            self.client = None
            raise ConnectionError(f"qdrant connection failed: {host}:{port}")

    def get_client(self) -> QdrantClient:
        if not self.client:
            raise ConnectionError(
                "qdrant client is not connected. Please call .connect() method first"
            )
        return self.client

    def create_collection_if_not_exists(self, name: str, vector_size: int):
        client = self.get_client()
        try:
            responses = client.get_collections()
            existing_collections = [c.name for c in responses.collections]

            if name not in existing_collections:
                self.logger.info(
                    f"collection '{name}' not found, creating a new one"
                )
                client.create_collection(
                    collection_name=name,
                    vectors_config=create_vector_config(vector_size),
                    sparse_vectors_config=get_sparse_vector_config(),
                )
                self.logger.info(f"created collection '{name}' successfully")
            else:
                self.logger.info(f"collection '{name}' already exists")
        except Exception as e:
            self.logger.error(f"failed to create collection '{name}': {e}")
            raise

    def recreate_collection(self, name: str, vector_size: int):
        client = self.get_client()
        try:
            self.logger.info(f"recreating collection '{name}'")
            client.recreate_collection(
                collection_name=name,
                vectors_config=create_vector_config(vector_size),
                sparse_vectors_config=get_sparse_vector_config(),
            )
            self.logger.info(f"recreated collection '{name}' successfully")
        except Exception as e:
            self.logger.error(f"failed to recreate collection '{name}': {e}")
            raise

    def get_points_by_document_name(
        self, collection_name: str, document_name: str
    ) -> List[Dict[str, Any]]:
        """Qdrant 컬렉션에서 특정 document_name 메타데이터를 가진 모든 데이터 포인트를 조회,
        유사도 검색이 아닌 조건 필터링을 통해 전체 데이터를 스크롤

        Args:
            collection_name (str): 조회할 컬렉션 이름
            document_name (str): 조회하고 싶은 문서명

        Returns:
            List[Dict[str, Any]]: 각 포인트의 ID와 payload(메타데이터)를 담은 딕셔너리 리스트
        """
        try:
            client = self.get_client()
            # Qdrant에 전달할 검색 필터 생성
            search_filter = models.Filter(
                # 'must'는 모든 조건이 AND로 연결
                must=[
                    # payload의 특정 필드('key')에 대한 조건
                    models.FieldCondition(
                        key="document_name",
                        match=models.MatchValue(value=document_name),
                    )
                ]
            )

            # 'scroll' 메서드를 사용하여 대량의 데이터를 페이지 단위로 조회
            scroll_result = client.scroll(
                collection_name=collection_name,
                scroll_filter=search_filter,
                with_payload=True,  # payload(메타데이터) 포함 여부
                with_vectors=False,  # 벡터(임베딩) 포함 여부
                limit=MAX_SCROLL_LIMIT,  # 한 번에 조회할 최대 포인트 수
            )

            # 조회 결과를 포인트 ID와 payload만 추출하여 리스트로 변환
            points = [
                {
                    "id": point.id,
                    "payload": point.payload,
                }
                for point in scroll_result[0]
            ]

            self.logger.debug(
                f"found {len(points)} points for document: {document_name}"
            )
            return points

        except Exception as e:
            self.logger.error(
                f"failed to get points for document '{document_name}': {e}"
            )
            raise

    def delete_points_by_document_name(
        self, collection_name: str, document_name: str
    ) -> int:
        """Qdrant 컬렉션에서 특정 document_name 메타데이터를 가진 모든 데이터 포인트를 삭제

        Args:
            collection_name (str): 삭제할 컬렉션 이름
            document_name (str): 삭제하고 싶은 문서명

        Returns:
            int: 삭제된 포인트 수
        """
        try:
            points = self.get_points_by_document_name(
                collection_name, document_name
            )

            # 삭제할 포인트가 없으면 0 반환
            if not points:
                self.logger.info(
                    f"no points found for document: {document_name}"
                )
                return 0

            point_ids = [point["id"] for point in points]
            client = self.get_client()

            # Qdrant의 'delete' 메서드를 사용하여 포인트 일괄 삭제
            client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=point_ids),
                wait=True,  # 삭제 작업이 완료될 때까지 대기
            )

            self.logger.info(
                f"deleted {len(point_ids)} points for document: {document_name}"
            )
            return len(point_ids)

        except Exception as e:
            self.logger.error(
                f"failed to delete points for document '{document_name}': {e}"
            )
            raise
