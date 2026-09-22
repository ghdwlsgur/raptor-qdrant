import json
import logging
import threading
from collections.abc import Iterator, Sequence
from typing import Any, Optional

from qdrant_client import QdrantClient, models

from raptor_qdrant.core.config import settings

SCROLL_PAGE_SIZE = 10000
DOCUMENT_NAME_KEY = "document_name"
# llama-index 는 본문을 평평한 text 필드가 아니라 이 JSON 안에 넣는다
NODE_CONTENT_KEY = "_node_content"


def stored_text(payload: dict[str, Any]) -> str:
    """포인트 payload 에서 본문을 꺼낸다."""
    raw = payload.get(NODE_CONTENT_KEY)
    if not isinstance(raw, str):
        return ""
    try:
        return str(json.loads(raw).get("text", ""))
    except (ValueError, TypeError):
        return ""


class QdrantManager:
    _instance: Optional["QdrantManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args: object, **kwargs: object) -> "QdrantManager":
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.client: QdrantClient | None = None
        self.logger = logging.getLogger(__name__)
        self._initialized: bool = True
        self.logger.info("created qdrant manager instance")

    def connect(
        self,
        host: str = settings.QDRANT_HOST,
        port: int = settings.QDRANT_PORT,
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
            raise ConnectionError(
                f"qdrant connection failed: {host}:{port}"
            ) from e

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

    def list_collections(self) -> list[str]:
        responses = self.get_client().get_collections()
        return sorted(collection.name for collection in responses.collections)

    def count_points_by_document_name(
        self, collection_name: str, document_name: str
    ) -> int:
        if not self.collection_exists(collection_name):
            return 0

        return (
            self.get_client()
            .count(
                collection_name=collection_name,
                count_filter=self._document_name_filter(document_name),
                exact=True,
            )
            .count
        )

    def get_points_by_document_name(
        self, collection_name: str, document_name: str
    ) -> list[dict[str, Any]]:
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

    def iter_payloads(
        self,
        collection_name: str,
        fields: Sequence[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """컬렉션의 모든 payload 를 순회한다.

        fields 를 주면 그 키만 받는다. 기본값은 본문(text)까지 포함한 전부라,
        포인트가 수만 개면 훑을 때마다 볼트 전체가 네트워크를 건너온다.
        """
        if not self.collection_exists(collection_name):
            return
        for point in self._scroll_all(collection_name, payload_fields=fields):
            if point.payload:
                yield point.payload

    def list_document_names(self, collection_name: str) -> list[str]:
        if not self.collection_exists(collection_name):
            return []

        return sorted(
            {
                point.payload[DOCUMENT_NAME_KEY]
                for point in self._scroll_all(
                    collection_name, payload_fields=[DOCUMENT_NAME_KEY]
                )
                if point.payload and DOCUMENT_NAME_KEY in point.payload
            }
        )

    def count_points_where(
        self,
        collection_name: str,
        key: str,
        value: Any,
        negate: bool = False,
    ) -> int:
        """payload 의 key 가 value 인(negate 면 아닌) 포인트 수."""
        if not self.collection_exists(collection_name):
            return 0

        return (
            self.get_client()
            .count(
                collection_name=collection_name,
                count_filter=self._match_filter(key, value, negate),
                exact=True,
            )
            .count
        )

    def delete_points_where(
        self,
        collection_name: str,
        key: str,
        value: Any,
        negate: bool = False,
    ) -> int:
        """payload 의 key 가 value 인(negate 면 아닌) 포인트를 지우고 개수를 돌려준다.

        negate=True 는 "이 세대가 아닌 것 전부" 처럼 쓴다. key 가 없거나 null
        인 포인트도 value 와 다르므로 함께 지워진다.
        """
        deleted = self.count_points_where(collection_name, key, value, negate)
        if not deleted:
            return 0

        self.get_client().delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=self._match_filter(key, value, negate)
            ),
            wait=True,
        )
        self.logger.info(
            f"deleted {deleted} points where {key} "
            f"{'!=' if negate else '=='} {value!r}"
        )
        return deleted

    def set_payload_where(
        self,
        collection_name: str,
        key: str,
        value: Any,
        payload: dict[str, Any],
    ) -> int:
        """payload 의 key 가 value 인 포인트에 payload 를 덧쓰고 개수를 돌려준다.

        배열 필드도 원소 하나가 맞으면 걸린다. 요약 노드가 물고 있는
        source_notes 에서 노트 하나를 찾을 때 쓴다.
        """
        matched = self.count_points_where(collection_name, key, value)
        if not matched:
            return 0

        self.get_client().set_payload(
            collection_name=collection_name,
            payload=payload,
            points=self._match_filter(key, value, False),
            wait=True,
        )
        self.logger.info(
            f"set {sorted(payload)} on {matched} points where {key} == {value!r}"
        )
        return matched

    @staticmethod
    def _match_filter(key: str, value: Any, negate: bool) -> models.Filter:
        condition = models.FieldCondition(
            key=key, match=models.MatchValue(value=value)
        )
        if negate:
            return models.Filter(must_not=[condition])
        return models.Filter(must=[condition])

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
        scroll_filter: models.Filter | None = None,
        payload_fields: Sequence[str] | None = None,
    ) -> Iterator[models.Record]:
        """컬렉션을 페이지 단위로 끝까지 순회한다."""
        client = self.get_client()
        offset = None
        with_payload: bool | list[str] = (
            list(payload_fields) if payload_fields else True
        )

        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                with_payload=with_payload,
                with_vectors=False,
                limit=SCROLL_PAGE_SIZE,
                offset=offset,
            )

            yield from points

            if offset is None:
                break
