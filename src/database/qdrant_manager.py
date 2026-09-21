import logging
import threading
from typing import Any, Dict, Iterator, List, Optional

from qdrant_client import QdrantClient, models

from src.core.config import settings

SCROLL_PAGE_SIZE = 10000
DOCUMENT_NAME_KEY = "document_name"


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
        if getattr(self, "_initialized", False):
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

    def collection_exists(self, name: str) -> bool:
        return self.get_client().collection_exists(collection_name=name)

    def drop_collection(self, name: str) -> bool:
        """컬렉션을 삭제한다. 없었으면 False."""
        client = self.get_client()
        try:
            if not self.collection_exists(name):
                self.logger.info(
                    f"collection '{name}' does not exist, nothing to drop"
                )
                return False

            client.delete_collection(collection_name=name)
            self.logger.info(f"dropped collection '{name}'")
            return True
        except Exception as e:
            self.logger.error(f"failed to drop collection '{name}': {e}")
            raise

    def list_collections(self) -> List[str]:
        responses = self.get_client().get_collections()
        return sorted(collection.name for collection in responses.collections)

    def count_points_by_document_name(
        self, collection_name: str, document_name: str
    ) -> int:
        if not self.collection_exists(collection_name):
            return 0

        return self.get_client().count(
            collection_name=collection_name,
            count_filter=self._document_name_filter(document_name),
            exact=True,
        ).count

    def get_points_by_document_name(
        self, collection_name: str, document_name: str
    ) -> List[Dict[str, Any]]:
        try:
            return [
                {"id": point.id, "payload": point.payload}
                for point in self._scroll_all(
                    collection_name,
                    scroll_filter=self._document_name_filter(document_name),
                )
            ]
        except Exception as e:
            self.logger.error(
                f"failed to get points for document '{document_name}': {e}"
            )
            raise

    def delete_points_by_document_name(
        self, collection_name: str, document_name: str
    ) -> int:
        """문서에 속한 포인트를 모두 지우고 지운 개수를 반환한다."""
        try:
            if not self.collection_exists(collection_name):
                self.logger.info(
                    f"collection '{collection_name}' does not exist, nothing to delete"
                )
                return 0

            deleted = self.count_points_by_document_name(
                collection_name, document_name
            )
            if not deleted:
                self.logger.info(
                    f"no points found for document: {document_name}"
                )
                return 0

            self.get_client().delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=self._document_name_filter(document_name)
                ),
                wait=True,
            )

            self.logger.info(
                f"deleted {deleted} points for document: {document_name}"
            )
            return deleted

        except Exception as e:
            self.logger.error(
                f"failed to delete points for document '{document_name}': {e}"
            )
            raise

    def list_document_names(self, collection_name: str) -> List[str]:
        if not self.collection_exists(collection_name):
            return []

        return sorted(
            {
                point.payload[DOCUMENT_NAME_KEY]
                for point in self._scroll_all(collection_name)
                if point.payload and DOCUMENT_NAME_KEY in point.payload
            }
        )

    @staticmethod
    def _document_name_filter(document_name: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=DOCUMENT_NAME_KEY,
                    match=models.MatchValue(value=document_name),
                )
            ]
        )

    def _scroll_all(
        self,
        collection_name: str,
        scroll_filter: Optional[models.Filter] = None,
    ) -> Iterator[models.Record]:
        """컬렉션을 페이지 단위로 끝까지 순회한다."""
        client = self.get_client()
        offset = None

        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False,
                limit=SCROLL_PAGE_SIZE,
                offset=offset,
            )

            yield from points

            if offset is None:
                break
