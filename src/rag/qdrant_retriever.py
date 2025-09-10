import uuid
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, List, Tuple

import numpy as np
import tiktoken
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from tiktoken.core import Encoding

from .embedding_models import BaseEmbeddingModel, OpenAIEmbeddingModel
from .retrievers import BaseRetriever
from .tree_structures import Tree
from .utils import split_text
from src.database.qdrant_manager import QdrantManager

logger = logging.getLogger(__name__)


class QdrantRetrieverConfig:
    """Qdrant Retriever 설정을 관리하는 클래스"""

    def __init__(
        self,
        max_tokens: int = 512,
        max_context_tokens: int = 3500,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        question_embedding_model: Optional[BaseEmbeddingModel] = None,
        top_k: int = 5,
        tokenizer: Optional[Encoding] = None,
        embedding_model_string: Optional[str] = None,
        collection_name: str = "default_collection",
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        if question_embedding_model is not None and not isinstance(
            question_embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "question_embedding_model must be an instance of BaseEmbeddingModel"
            )

        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.embedding_model = embedding_model or OpenAIEmbeddingModel()
        self.question_embedding_model = (
            question_embedding_model or self.embedding_model
        )
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
        self.embedding_model_string = embedding_model_string or "OpenAI"
        self.collection_name = collection_name

        self.vector_size = len(
            self.embedding_model.create_embedding("test vector size")
        )
        logger.info(
            f"Vector size for collection '{self.collection_name}' is set to {self.vector_size}"
        )


class QdrantRetriever(BaseRetriever):
    """
    Qdrant를 사용하여 RAPTOR 트리에 대한 계층적 검색을 수행하는 클래스.
    """

    def __init__(self, config: QdrantRetrieverConfig):
        self.config = config
        self.tokenizer = config.tokenizer
        self.question_embedding_model = config.question_embedding_model
        self.embedding_model = config.embedding_model

        self.client: QdrantClient = QdrantManager().get_client()
        self.collection_name = config.collection_name
        self.vector_size = config.vector_size

        logger.info(
            f"Initialized QdrantRetriever for collection: '{self.collection_name}'"
        )
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Qdrant에 컬렉션이 없으면 생성합니다."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            if self.collection_name not in collection_names:
                logger.info(
                    f"Collection '{self.collection_name}' not found. Creating a new one."
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size, distance=models.Distance.COSINE
                    ),
                )
        except Exception as e:
            logger.error(
                f"Failed to ensure collection '{self.collection_name}': {e}"
            )
            raise

    def build_from_tree(self, tree: Tree):
        """
        RAPTOR Tree 객체로부터 Qdrant 컬렉션을 구축합니다.
        기존 컬렉션은 삭제하고 새로 만듭니다.
        """
        all_nodes = list(tree.all_nodes.values())
        logger.info(
            f"Building index from a tree with {len(all_nodes)} total nodes."
        )

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size, distance=models.Distance.COSINE
            ),
        )

        logger.info(f"Uploading {len(all_nodes)} nodes to Qdrant...")
        points_to_upload = []
        for node in all_nodes:
            layer = tree.get_node_layer(node.index)
            points_to_upload.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=node.embeddings[self.config.embedding_model_string],
                    payload={
                        "text": node.text,
                        "layer": layer,
                        "node_index": node.index,
                    },
                )
            )

        # 'batch_size' 인자를 제거했습니다.
        self.client.upsert(
            collection_name=self.collection_name,
            points=points_to_upload,
            wait=True,
        )
        logger.info("Tree indexing complete.")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> Tuple[str, List[dict]]:
        """
        쿼리를 기반으로 Qdrant에서 관련 컨텍스트를 검색합니다.
        """
        # 파라미터가 제공되지 않으면 config의 기본값을 사용합니다.
        top_k = top_k if top_k is not None else self.config.top_k
        max_tokens = (
            max_tokens
            if max_tokens is not None
            else self.config.max_context_tokens
        )

        query_embedding = self.question_embedding_model.create_embedding(query)

        search_filter = None
        if not collapse_tree:
            if start_layer is None:
                raise ValueError(
                    "start_layer must be specified when collapse_tree is False."
                )
            # 지정된 레이어의 노드만 검색하도록 필터 생성
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="layer", match=models.MatchValue(value=start_layer)
                    )
                ]
            )

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
        )

        context = ""
        total_tokens = 0
        layer_information = []
        for hit in search_results:
            chunk = hit.payload['text']
            tokens = len(self.tokenizer.encode(chunk))
            if total_tokens + tokens <= max_tokens:
                context += chunk + "\n\n"
                total_tokens += tokens
                layer_information.append(
                    {
                        "node_index": hit.payload.get('node_index'),
                        "layer_number": hit.payload.get('layer'),
                        "score": hit.score,
                    }
                )
            else:
                break

        logger.info(
            f"Retrieved context with {total_tokens} tokens from {len(layer_information)} chunks."
        )
        return context.strip(), layer_information
