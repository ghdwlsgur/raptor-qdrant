import concurrent.futures
import copy
import logging
import os
from abc import abstractmethod
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.schema import TextNode
from tqdm import tqdm

from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.chunker.hybrid_chunker import BaseChunker, HybridChunker
from raptor_qdrant.rag.constants import (
    DEFAULT_SUMMARIZATION_MAX_WORKERS,
    SOURCE_KEY,
    SUMMARIZATION_MAX_WORKERS,
)
from raptor_qdrant.rag.embedding import (
    BaseEmbeddingModel,
    KoreanEmbeddingModel,
)
from raptor_qdrant.rag.summarizer import (
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
        summarization_model: BaseSummarizationModel
        | None = None,  # 요약에 사용할 LLM 모델
        embedding_models: dict[str, BaseEmbeddingModel]
        | None = None,  # 텍스트 임베딩에 사용할 모델
        cluster_embedding_model: str
        | None = None,  # 클러스터링 시 사용할 임베딩 모델 키
        chunker: BaseChunker
        | None = None,  # 텍스트 청킹에 사용할 청킹 오브젝트
        summarization_max_workers: int
        | None = None,  # 클러스터 요약 동시 실행 수
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
            main_model_name = next(iter(self.embedding_models))
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
        children_indices: set[int] | None = None,
    ) -> tuple[int, Node]:
        """주어진 LlamaIndex TextNode로부터 커스텀 Node 객체 생성"""
        if children_indices is None:
            children_indices = set()

        text = llama_node.get_content()
        main_model_name = next(iter(self.embedding_models))

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
        self, nodes: list[TextNode]
    ) -> dict[int, Node]:
        """ThreadPoolExecutor를 사용하여 안전한 멀티스레딩으로 Leaf Node 생성"""
        leaf_nodes: dict[int, Node] = {}

        if not nodes:
            logger.warning("No nodes provided for leaf node creation")
            return leaf_nodes

        # 적절한 스레드 수 계산 (CPU 코어 수의 2배, 최대 16개로 제한)
        max_workers = min(len(nodes), max(1, (os.cpu_count() or 1) * 2), 16)

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
                            index, created = future.result(timeout=60)
                            leaf_nodes[index] = created
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

    def _chunk_into_nodes(self, text: str) -> list[TextNode]:
        nodes = self.chunker.chunk(text)

        kept = [node for node in nodes if node.get_content().strip()]
        dropped = len(nodes) - len(kept)
        if dropped:
            logger.info(f"dropped {dropped} empty chunk(s) before embedding")

        if not kept:
            raise ValueError("chunking produced no non-empty nodes")

        return kept

    def build_from_documents(
        self,
        documents: Mapping[str, str],
        use_multithreading: bool = True,
    ) -> Tree:
        """여러 문서의 청크 위에 트리 하나를 올린다.

        문서를 이어붙여 한 번에 청킹하면 잎마다 어느 문서에서 왔는지가 사라진다.
        문서별로 청킹해 출처를 박은 뒤, 그 잎 전체를 대상으로 트리를 쌓는다.
        """
        if not documents:
            raise ValueError("documents must not be empty")

        return self._assemble(
            self._chunk_documents(documents), use_multithreading
        )

    def _chunk_documents(self, documents: Mapping[str, str]) -> list[TextNode]:
        if not documents:
            raise ValueError("documents must not be empty")

        nodes: list[TextNode] = []
        for name, text in documents.items():
            if not text or not text.strip():
                continue
            for node in self._chunk_into_nodes(text):
                node.metadata = {**node.metadata, SOURCE_KEY: name}
                nodes.append(node)

        if not nodes:
            raise ValueError("chunking produced no non-empty nodes")

        logger.info(
            f"chunked {len(documents)} documents into {len(nodes)} leaf chunks"
        )
        return nodes

    def build_leaves_only(
        self,
        documents: Mapping[str, str],
        use_multithreading: bool = True,
    ) -> Tree:
        """클러스터링과 요약 없이 잎 노드만 만든다.

        노트 하나를 고칠 때마다 트리를 다시 세울 수는 없다. 잎은 청킹과 로컬
        임베딩뿐이라 즉시 갱신할 수 있고, 트리 간선은 저장되지 않으므로 잎만
        갈아끼워도 적재된 요약 노드가 깨지지 않는다.
        """
        nodes = self._chunk_documents(documents)
        leaf_nodes = self._create_leaf_nodes(nodes, use_multithreading)

        return Tree(
            all_nodes=copy.deepcopy(leaf_nodes),
            root_nodes=leaf_nodes,
            leaf_nodes=leaf_nodes,
            num_layers=0,
            layer_to_nodes={0: list(leaf_nodes.values())},
        )

    def build_from_text(
        self, text: str, use_multithreading: bool = True
    ) -> Tree:
        """전체 텍스트에서 최종 Tree 객체를 생성"""
        if not text or not text.strip():
            raise ValueError("cannot build a tree from empty text")

        # chunker를 사용하여 텍스트를 여러 개의 TextNode로 분할
        return self._assemble(self._chunk_into_nodes(text), use_multithreading)

    def _create_leaf_nodes(
        self, nodes: list[TextNode], use_multithreading: bool
    ) -> dict[int, Node]:
        if use_multithreading:
            return self.multithreaded_create_leaf_nodes(nodes)
        return {
            i: self.create_node(i, node)[1]
            for i, node in enumerate(tqdm(nodes, desc="Creating Leaf Nodes"))
        }

    def _assemble(
        self, nodes: list[TextNode], use_multithreading: bool
    ) -> Tree:
        leaf_nodes = self._create_leaf_nodes(nodes, use_multithreading)

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
        all_tree_nodes: dict[int, Node],
        layer_to_nodes: dict[int, list[Node]],
        use_multithreading: bool = True,
    ) -> dict[int, Node]:
        pass
