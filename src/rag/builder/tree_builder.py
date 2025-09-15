import copy
import logging
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple
from llama_index.core.schema import TextNode

from tqdm import tqdm
from src.rag.embedding import BaseEmbeddingModel, KoreanEmbeddingModel
from src.rag.chunking import BaseChunker, SemanticChunker
from src.core.config import settings
from src.rag.summarizer import (
    BaseSummarizationModel,
    BedrockSummarizer,
)
from .models.structure import Node, Tree

logger = logging.getLogger(__name__)


class TreeBuilderConfig:
    def __init__(
        self,
        num_layers: int = 5,
        summarization_length: int = 100,
        summarization_model: Optional[BaseSummarizationModel] = None,
        embedding_models: Optional[Dict[str, BaseEmbeddingModel]] = None,
        cluster_embedding_model: Optional[str] = None,
        chunker: Optional[BaseChunker] = None,
    ):
        self.num_layers = num_layers
        self.summarization_length = summarization_length
        self.summarization_model = summarization_model or BedrockSummarizer()
        if not isinstance(self.summarization_model, BaseSummarizationModel):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )

        if embedding_models is None:
            embedding_models = {settings.EMBEDDING_MODEL_STRING: KoreanEmbeddingModel()}
        self.embedding_models = embedding_models

        self.cluster_embedding_model = cluster_embedding_model or settings.EMBEDDING_MODEL_STRING
        if self.cluster_embedding_model not in self.embedding_models:
            raise ValueError(
                "cluster_embedding_model must be a key in the embedding_models dictionary"
            )

        # 기본 청킹 전략 설정
        if chunker is None:
            main_model_name = list(self.embedding_models.keys())[0]
            embedding_model = self.embedding_models[main_model_name]
            chunker = SemanticChunker(embedding_model)
        self.chunker = chunker

    def log_config(self) -> str:
        config_log = """
        TreeBuilderConfig:
            Num Layers: {num_layers}
            Summarization Length: {summarization_length}
            Summarization Model: {summarization_model}
            Embedding Models: {embedding_models}
            Cluster Embedding Model: {cluster_embedding_model}
            Chunker: {chunker}
        """.format(
            num_layers=self.num_layers,
            summarization_length=self.summarization_length,
            summarization_model=self.summarization_model.__class__.__name__,
            embedding_models={
                k: v.__class__.__name__
                for k, v in self.embedding_models.items()
            },
            cluster_embedding_model=self.cluster_embedding_model,
            chunker=self.chunker.__class__.__name__,
        )
        return config_log


class TreeBuilder:
    def __init__(self, config: TreeBuilderConfig) -> None:
        self.num_layers = config.num_layers
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.embedding_models = config.embedding_models
        self.cluster_embedding_model = config.cluster_embedding_model
        self.chunker = config.chunker

    def create_node(
        self,
        index: int,
        llama_node: TextNode,
        children_indices: Optional[Set[int]] = None,
    ) -> Tuple[int, Node]:
        if children_indices is None:
            children_indices = set()

        text = llama_node.get_content()
        main_model_name = list(self.embedding_models.keys())[0]

        if llama_node.embedding is None:
            embedding = self.embedding_models[main_model_name].create_embedding(
                text
            )
        else:
            embedding = llama_node.embedding

        embeddings_dict = {main_model_name: embedding}
        return (index, Node(text, index, children_indices, embeddings_dict))

    def summarize(self, context, max_tokens=150) -> str:
        return self.summarization_model.summarize(context, max_tokens)

    def multithreaded_create_leaf_nodes(
        self, nodes: List[TextNode]
    ) -> Dict[int, Node]:
        leaf_nodes = {}
        with ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(self.create_node, i, node): i
                for i, node in enumerate(nodes)
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(nodes),
                desc="Creating Leaf Nodes",
            ):
                index, node = future.result()
                leaf_nodes[index] = node
        return leaf_nodes

    def build_from_text(
        self, text: str, use_multithreading: bool = True
    ) -> Tree:
        # 청킹 전략을 사용하여 텍스트 분할
        nodes = self.chunker.chunk(text)

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(nodes)
        else:
            leaf_nodes = {
                i: self.create_node(i, node)[1]
                for i, node in enumerate(
                    tqdm(nodes, desc="Creating Leaf Nodes")
                )
            }

        all_nodes = copy.deepcopy(leaf_nodes)
        layer_to_nodes = {0: list(leaf_nodes.values())}

        logger.info(f"created {len(leaf_nodes)} leaf nodes")

        root_nodes = self.construct_tree(
            all_nodes, layer_to_nodes, use_multithreading
        )
        final_depth = len(layer_to_nodes) - 1

        tree = Tree(
            all_nodes, root_nodes, leaf_nodes, final_depth, layer_to_nodes
        )
        return tree

    @abstractmethod
    def construct_tree(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        pass
