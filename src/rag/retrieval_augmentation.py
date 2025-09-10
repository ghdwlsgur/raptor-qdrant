# 파일 경로: src/rag/retrieval_augmentation.py

import logging
from typing import Optional, Tuple, List

# import 이름을 OpenAISummarizationModel 로 통일합니다.
from .summarization_models import (
    BaseSummarizationModel,
    OpenAISummarizationModel,
)
from .embedding_models import BaseEmbeddingModel, OpenAIEmbeddingModel
from .qa_models import BaseQAModel, GPT3TurboQAModel
from .cluster_tree_builder import (
    ClusterTreeBuilder,
    ClusterTreeConfig,
)
from .qdrant_retriever import (
    QdrantRetriever,
    QdrantRetrieverConfig,
)
from .tree_structures import Tree

logger = logging.getLogger(__name__)

SUPPORTED_TREE_BUILDERS = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}


class RAGConfig:
    def __init__(
        self,
        retriever_type: str = "qdrant",
        tree_builder_type: str = "cluster",
        embedding_model: Optional[BaseEmbeddingModel] = None,
        qa_model: Optional[BaseQAModel] = None,
        summarization_model: Optional[BaseSummarizationModel] = None,
        collection_name: str = "default_collection",
        max_tokens_per_chunk: int = 512,
        top_k: int = 5,
        max_context_tokens: int = 3500,
    ):
        self.embedding_model = embedding_model or OpenAIEmbeddingModel()
        self.qa_model = qa_model or GPT3TurboQAModel()
        self.summarization_model = (
            summarization_model or OpenAISummarizationModel()
        )
        self.retriever_type = retriever_type
        self.tree_builder_type = tree_builder_type

        # Qdrant Retriever 설정
        self.retriever_config = QdrantRetrieverConfig(
            max_tokens=max_tokens_per_chunk,
            max_context_tokens=max_context_tokens,
            embedding_model=self.embedding_model,
            top_k=top_k,
            collection_name=collection_name,
        )

        # Tree Builder 설정
        if tree_builder_type in SUPPORTED_TREE_BUILDERS:
            _, tree_builder_config_class = SUPPORTED_TREE_BUILDERS[
                tree_builder_type
            ]
            # TreeBuilderConfig에 필요한 파라미터를 retriever_config와 model에서 가져옵니다.
            self.tree_builder_config = tree_builder_config_class(
                tokenizer=self.retriever_config.tokenizer,
                max_tokens=self.retriever_config.max_tokens,
                embedding_models={
                    "OpenAI": self.embedding_model
                },  # 키를 'OpenAI'로 통일
                cluster_embedding_model="OpenAI",  # 키를 'OpenAI'로 통일
                summarization_model=self.summarization_model,
            )
        else:
            raise ValueError(
                f"Unsupported tree_builder_type: {tree_builder_type}"
            )


class RetrievalAugmentation:
    def __init__(self, config: Optional[RAGConfig] = None):
        if config is None:
            config = RAGConfig()

        self.config = config
        self.qa_model = self.config.qa_model
        self.retriever = QdrantRetriever(self.config.retriever_config)
        tree_builder_class, _ = SUPPORTED_TREE_BUILDERS[
            self.config.tree_builder_type
        ]
        self.tree_builder = tree_builder_class(self.config.tree_builder_config)

        logger.info(
            f"RetrievalAugmentation initialized for collection '{self.retriever.collection_name}'"
        )

    def add_documents(self, text: str):
        logger.info("Building RAPTOR tree from document...")
        tree = self.tree_builder.build_from_text(text)

        logger.info(
            f"Indexing all {len(tree.all_nodes)} nodes from the tree into Qdrant..."
        )
        self.retriever.build_from_tree(tree)
        logger.info("Document processing and indexing complete.")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> Tuple[str, List[dict]]:
        if self.retriever is None:
            raise ValueError("Retriever is not initialized.")

        logger.info(f"Retrieving context for query: '{query}'")
        return self.retriever.retrieve(
            query,
            top_k=top_k,
            max_tokens=max_tokens,
            collapse_tree=collapse_tree,
            start_layer=start_layer,
        )

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        collapse_tree: bool = True,
        start_layer: Optional[int] = None,
    ) -> str:
        context, _ = self.retrieve(
            question,
            top_k=top_k,
            max_tokens=max_tokens,
            collapse_tree=collapse_tree,
            start_layer=start_layer,
        )

        if not context:
            logger.warning("No context retrieved for the question.")
            return "죄송합니다. 관련 정보를 찾을 수 없어 답변할 수 없습니다."

        logger.info("Context retrieved. Generating answer...")
        answer = self.qa_model.answer_question(context, question)
        return answer
