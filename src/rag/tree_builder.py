import copy
import logging
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

import tiktoken
from tiktoken.core import Encoding
from tqdm import tqdm  # <--- 이 부분이 수정되었습니다.

from .summarization_models import (
    BaseSummarizationModel,
    OpenAISummarizationModel,
)
from .embedding_models import BaseEmbeddingModel, OpenAIEmbeddingModel
from .tree_structures import Node, Tree
from .utils import split_text

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TreeBuilderConfig:
    def __init__(
        self,
        tokenizer: Optional[Encoding] = None,
        max_tokens: int = 100,
        num_layers: int = 5,
        summarization_length: int = 100,
        summarization_model: Optional[BaseSummarizationModel] = None,
        embedding_models: Optional[Dict[str, BaseEmbeddingModel]] = None,
        cluster_embedding_model: Optional[str] = None,
    ):
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens
        self.num_layers = num_layers
        self.summarization_length = summarization_length

        self.summarization_model = (
            summarization_model or OpenAISummarizationModel()
        )
        if not isinstance(self.summarization_model, BaseSummarizationModel):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )

        if embedding_models is None:
            embedding_models = {"OpenAI": OpenAIEmbeddingModel()}
        self.embedding_models = embedding_models

        self.cluster_embedding_model = cluster_embedding_model or "OpenAI"
        if self.cluster_embedding_model not in self.embedding_models:
            raise ValueError(
                "cluster_embedding_model must be a key in the embedding_models dictionary"
            )

    def log_config(self) -> str:
        config_log = """
        TreeBuilderConfig:
            Tokenizer: {tokenizer}
            Max Tokens: {max_tokens}
            Num Layers: {num_layers}
            Summarization Length: {summarization_length}
            Summarization Model: {summarization_model}
            Embedding Models: {embedding_models}
            Cluster Embedding Model: {cluster_embedding_model}
        """.format(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            num_layers=self.num_layers,
            summarization_length=self.summarization_length,
            summarization_model=self.summarization_model.__class__.__name__,
            embedding_models={
                k: v.__class__.__name__
                for k, v in self.embedding_models.items()
            },
            cluster_embedding_model=self.cluster_embedding_model,
        )
        return config_log


class TreeBuilder:
    def __init__(self, config: TreeBuilderConfig) -> None:
        self.tokenizer = config.tokenizer
        self.max_tokens = config.max_tokens
        self.num_layers = config.num_layers
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.embedding_models = config.embedding_models
        self.cluster_embedding_model = config.cluster_embedding_model
        logging.info("Successfully initialized TreeBuilder")

    def create_node(
        self, index: int, text: str, children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, Node]:
        if children_indices is None:
            children_indices = set()
        embeddings = {
            model_name: model.create_embedding(text)
            for model_name, model in self.embedding_models.items()
        }
        return (index, Node(text, index, children_indices, embeddings))

    def summarize(self, context, max_tokens=150) -> str:
        return self.summarization_model.summarize(context, max_tokens)

    def multithreaded_create_leaf_nodes(
        self, chunks: List[str]
    ) -> Dict[int, Node]:
        leaf_nodes = {}
        with ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(self.create_node, i, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(chunks),
                desc="Creating Leaf Nodes",
            ):
                index, node = future.result()
                leaf_nodes[index] = node
        return leaf_nodes

    def build_from_text(
        self, text: str, use_multithreading: bool = True
    ) -> Tree:
        chunks = split_text(
            text=text, tokenizer=self.tokenizer, max_tokens=self.max_tokens
        )
        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
        else:
            leaf_nodes = {
                i: self.create_node(i, chunk)[1]
                for i, chunk in enumerate(
                    tqdm(chunks, desc="Creating Leaf Nodes")
                )
            }

        all_nodes = copy.deepcopy(leaf_nodes)
        layer_to_nodes = {0: list(leaf_nodes.values())}

        logging.info(f"Created {len(leaf_nodes)} Leaf nodes.")
        logging.info("Building summary layers...")

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
