import copy
import logging
import os
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm
from llama_index.core.schema import TextNode
from src.rag.embedding import BaseEmbeddingModel, KoreanEmbeddingModel
from src.rag.chunker.hybrid_chunker import BaseChunker, HybridChunker
from src.core.config import settings
from src.rag.constants import (
    SUMMARIZATION_MAX_WORKERS,
    DEFAULT_SUMMARIZATION_MAX_WORKERS,
)
from src.rag.summarizer import (
    BaseSummarizationModel,
    LLMSummarizer,
)
from .models.structure import Node, Tree

logger = logging.getLogger(__name__)


class TreeBuilderConfig:
    def __init__(
        self,
        num_layers: int = 5,  # 생성할 트리의 최대 계층(깊이) 수
        summarization_length: int = 100,  # 상위 노드를 생성할 때 요약 텍스트의 길이
        summarization_model: Optional[
            BaseSummarizationModel
        ] = None,  # 요약에 사용할 LLM 모델
        embedding_models: Optional[
            Dict[str, BaseEmbeddingModel]
        ] = None,  # 텍스트 임베딩에 사용할 모델
        cluster_embedding_model: Optional[
            str
        ] = None,  # 클러스터링 시 사용할 임베딩 모델 키
        chunker: Optional[
            BaseChunker
        ] = None,  # 텍스트 청킹에 사용할 청킹 오브젝트
        summarization_max_workers: Optional[
            int
        ] = None,  # 클러스터 요약 동시 실행 수
    ):
        self.num_layers = num_layers
        self.summarization_max_workers = (
            summarization_max_workers
            or SUMMARIZATION_MAX_WORKERS.get(
                settings.LLM_PROVIDER, DEFAULT_SUMMARIZATION_MAX_WORKERS
            )
        )
        self.summarization_length = summarization_length
        self.summarization_model = summarization_model or LLMSummarizer()
        if not isinstance(self.summarization_model, BaseSummarizationModel):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )

        if embedding_models is None:
            embedding_models = {
                settings.EMBEDDING_MODEL_STRING: KoreanEmbeddingModel()
            }

        self.embedding_models = embedding_models
        self.cluster_embedding_model = (
            cluster_embedding_model or settings.EMBEDDING_MODEL_STRING
        )
        if self.cluster_embedding_model not in self.embedding_models:
            raise ValueError(
                "cluster_embedding_model must be a key in the embedding_models dictionary"
            )

        # 기본 청킹 전략으로 HybridChunker 사용
        if chunker is None:
            main_model_name = list(self.embedding_models.keys())[0]
            embedding_model = self.embedding_models[main_model_name]
            chunker = HybridChunker(embedding_model)
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
            Summarization Max Workers: {summarization_max_workers}
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
            summarization_max_workers=self.summarization_max_workers,
        )
        return config_log


class TreeBuilder:
    """텍스트로부터 RAPTOR 트리를 구축하는 클래스"""

    def __init__(self, config: TreeBuilderConfig) -> None:
        self.num_layers = config.num_layers
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.embedding_models = config.embedding_models
        self.cluster_embedding_model = config.cluster_embedding_model
        self.chunker = config.chunker
        self.summarization_max_workers = config.summarization_max_workers

    def create_node(
        self,
        index: int,
        llama_node: TextNode,
        children_indices: Optional[Set[int]] = None,
    ) -> Tuple[int, Node]:
        """주어진 LlamaIndex TextNode로부터 커스텀 Node 객체 생성"""
        if children_indices is None:
            children_indices = set()

        text = llama_node.get_content()
        main_model_name = list(self.embedding_models.keys())[0]

        if llama_node.embedding is None:
            embedding_model = self.embedding_models[main_model_name]
            embedding = embedding_model.create_embedding(text)
        else:
            embedding = llama_node.embedding

        embeddings_dict = {main_model_name: embedding}

        # LlamaIndex TextNode의 메타데이터를 복사
        metadata = dict(llama_node.metadata) if llama_node.metadata else {}

        # (인덱스, 생성된 Node 객체) 튜플 반환
        return (
            index,
            Node(
                text=text,
                index=index,
                children=children_indices,
                embeddings=embeddings_dict,
                metadata=metadata,
            ),
        )

    def summarize(self, text) -> str:
        """주어진 컨텍스트(텍스트)를 요약"""
        return self.summarization_model.summarize(text)

    def multithreaded_create_leaf_nodes(
        self, nodes: List[TextNode]
    ) -> Dict[int, Node]:
        """ThreadPoolExecutor를 사용하여 안전한 멀티스레딩으로 Leaf Node 생성"""
        leaf_nodes = {}

        if not nodes:
            logger.warning("No nodes provided for leaf node creation")
            return leaf_nodes

        # 적절한 스레드 수 계산 (CPU 코어 수의 2배, 최대 16개로 제한)
        max_workers = min(len(nodes), max(1, os.cpu_count() * 2), 16)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 모든 작업을 한 번에 제출
                futures = [
                    executor.submit(self.create_node, i, node)
                    for i, node in enumerate(nodes)
                ]

                # 완료된 작업들을 안전하게 처리
                completed_count = 0
                with tqdm(
                    total=len(futures), desc="Creating Leaf Nodes"
                ) as pbar:
                    for future in concurrent.futures.as_completed(
                        futures, timeout=300
                    ):
                        try:
                            index, node = future.result(timeout=60)
                            leaf_nodes[index] = node
                            completed_count += 1
                        except concurrent.futures.TimeoutError:
                            logger.error("task timed out")
                        except Exception as e:
                            logger.error(f"task failed: {e}")
                        finally:
                            pbar.update(1)

                logger.info(
                    f"successfully created {completed_count}/{len(nodes)} leaf nodes"
                )

        except Exception as e:
            logger.error(f"Critical error in multithreaded node creation: {e}")
            # 폴백: 싱글스레드로 처리
            logger.info("Falling back to single-threaded node creation")
            for i, node in enumerate(
                tqdm(nodes, desc="Creating Leaf Nodes (Fallback)")
            ):
                try:
                    index, created_node = self.create_node(i, node)
                    leaf_nodes[index] = created_node
                except Exception as node_error:
                    logger.error(
                        f"Failed to create node {i} in fallback mode: {node_error}"
                    )

        return leaf_nodes

    def _chunk_into_nodes(self, text: str) -> List[TextNode]:
        nodes = self.chunker.chunk(text)

        kept = [node for node in nodes if node.get_content().strip()]
        dropped = len(nodes) - len(kept)
        if dropped:
            logger.info(f"dropped {dropped} empty chunk(s) before embedding")

        if not kept:
            raise ValueError("chunking produced no non-empty nodes")

        return kept

    def build_from_text(
        self, text: str, use_multithreading: bool = True
    ) -> Tree:
        """전체 텍스트에서 최종 Tree 객체를 생성"""
        if not text or not text.strip():
            raise ValueError("cannot build a tree from empty text")

        # chunker를 사용하여 텍스트를 여러 개의 TextNode로 분할
        nodes = self._chunk_into_nodes(text)

        # 분할된 TextNode들로 트리의 리프 노드 구성
        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(nodes)
        else:
            leaf_nodes = {
                i: self.create_node(i, node)[1]
                for i, node in enumerate(
                    tqdm(nodes, desc="Creating Leaf Nodes")
                )
            }

        # 모든 노드를 깊은 복사하여 all_nodes에 저장
        all_nodes = copy.deepcopy(leaf_nodes)
        # 각 계층별 노드 리스트를 저장할 딕셔너리
        layer_to_nodes = {0: list(leaf_nodes.values())}
        logger.info(f"created {len(leaf_nodes)} leaf nodes")

        # 상위 노드 생성 및 트리 구축
        # 하위 클래스(ClusterTreeBuilder)에서 구현된 construct_tree 메서드 호출
        root_nodes = self.construct_tree(
            all_nodes, layer_to_nodes, use_multithreading
        )
        final_depth = len(layer_to_nodes) - 1

        # 최종 Tree 객체 생성
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
