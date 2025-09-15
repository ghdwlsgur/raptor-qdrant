import threading
import logging
from typing import Optional
from qdrant_client import QdrantClient, models
from src.core.config import settings


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
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                    sparse_vectors_config={
                        "text-sparse-new": models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False)
                        )
                    },
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
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
                sparse_vectors_config={
                    "text-sparse-new": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            self.logger.info(f"recreated collection '{name}' successfully")
        except Exception as e:
            self.logger.error(f"failed to recreate collection '{name}': {e}")
            raise
