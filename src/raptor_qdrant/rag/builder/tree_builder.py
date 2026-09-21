import logging
from abc import abstractmethod
from collections.abc import Callable, Mapping

from llama_index.core.schema import TextNode

from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.chunker.hybrid_chunker import BaseChunker, HybridChunker
from raptor_qdrant.rag.constants import (
    DEFAULT_SUMMARIZATION_MAX_WORKERS,
    EMBEDDING_BATCH_SIZE,
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

# 레이어 하나가 완성될 때마다 (레이어 번호, 그 레이어의 노드들) 로 불린다.
# 잎은 레이어 0 이다. 적재를 레이어 단위로 미루지 않고 바로 하기 위한 훅이다.
LayerCallback = Callable[[int, list[Node]], None]


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

    def _main_embedding(self) -> tuple[str, BaseEmbeddingModel]:
        name = next(iter(self.embedding_models))
        return name, self.embedding_models[name]

    def create_nodes(
        self,
        start_index: int,
        llama_nodes: list[TextNode],
        children: list[set[int]] | None = None,
        label: str = "nodes",
    ) -> dict[int, Node]:
        """TextNode 묶음을 임베딩해 Node 로 만든다. 임베딩은 배치로 던진다.

        노드 하나씩 encode 를 부르면 호출마다 GPU 왕복이 생기고, 그걸 스레드
        열여섯 개가 동시에 하면 서로 경합만 한다. 리스트로 한 번에 넘기는
        쪽이 훨씬 빠르다. 이미 임베딩이 붙어 있는 노드는 그대로 쓴다.
        """
        if children is not None and len(children) != len(llama_nodes):
            raise ValueError("children must line up with llama_nodes")

        model_name, model = self._main_embedding()
        texts = [node.get_content() for node in llama_nodes]

        vectors: dict[int, list[float]] = {
            i: list(node.embedding)
            for i, node in enumerate(llama_nodes)
            if node.embedding is not None
        }
        pending = [i for i in range(len(llama_nodes)) if i not in vectors]

        for start in range(0, len(pending), EMBEDDING_BATCH_SIZE):
            batch = pending[start : start + EMBEDDING_BATCH_SIZE]
            embedded = model.create_embeddings([texts[i] for i in batch])
            vectors.update(zip(batch, embedded, strict=True))
            logger.info(
                f"embedded {start + len(batch)}/{len(pending)} {label}"
            )

        return {
            start_index + i: Node(
                text=texts[i],
                index=start_index + i,
                children=set(children[i]) if children else set(),
                embeddings={model_name: vectors[i]},
                metadata=dict(node.metadata) if node.metadata else {},
            )
            for i, node in enumerate(llama_nodes)
        }

    def create_node(
        self,
        index: int,
        llama_node: TextNode,
        children_indices: set[int] | None = None,
    ) -> tuple[int, Node]:
        """노드 하나를 만든다. 여럿이면 create_nodes 가 배치로 처리한다."""
        nodes = self.create_nodes(
            index, [llama_node], [children_indices or set()]
        )
        return index, nodes[index]

    def summarize(self, text) -> str:
        """주어진 컨텍스트(텍스트)를 요약"""
        return self.summarization_model.summarize(text)

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
        on_layer_built: LayerCallback | None = None,
    ) -> Tree:
        """여러 문서의 청크 위에 트리 하나를 올린다.

        문서를 이어붙여 한 번에 청킹하면 잎마다 어느 문서에서 왔는지가 사라진다.
        문서별로 청킹해 출처를 박은 뒤, 그 잎 전체를 대상으로 트리를 쌓는다.
        """
        if not documents:
            raise ValueError("documents must not be empty")

        return self._assemble(
            self._chunk_documents(documents),
            use_multithreading,
            on_layer_built,
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

    def build_leaves_only(self, documents: Mapping[str, str]) -> Tree:
        """클러스터링과 요약 없이 잎 노드만 만든다.

        노트 하나를 고칠 때마다 트리를 다시 세울 수는 없다. 잎은 청킹과 로컬
        임베딩뿐이라 즉시 갱신할 수 있고, 트리 간선은 저장되지 않으므로 잎만
        갈아끼워도 적재된 요약 노드가 깨지지 않는다.
        """
        leaf_nodes = self._create_leaf_nodes(self._chunk_documents(documents))

        return Tree(
            all_nodes=dict(leaf_nodes),
            root_nodes=leaf_nodes,
            leaf_nodes=leaf_nodes,
            num_layers=0,
            layer_to_nodes={0: list(leaf_nodes.values())},
        )

    def build_from_text(
        self,
        text: str,
        use_multithreading: bool = True,
        on_layer_built: LayerCallback | None = None,
    ) -> Tree:
        """전체 텍스트에서 최종 Tree 객체를 생성"""
        if not text or not text.strip():
            raise ValueError("cannot build a tree from empty text")

        return self._assemble(
            self._chunk_into_nodes(text), use_multithreading, on_layer_built
        )

    def _create_leaf_nodes(self, nodes: list[TextNode]) -> dict[int, Node]:
        logger.info(
            f"embedding {len(nodes)} leaf chunks in batches of "
            f"{EMBEDDING_BATCH_SIZE}"
        )
        return self.create_nodes(0, nodes, label="leaf chunks")

    def _assemble(
        self,
        nodes: list[TextNode],
        use_multithreading: bool,
        on_layer_built: LayerCallback | None = None,
    ) -> Tree:
        leaf_nodes = self._create_leaf_nodes(nodes)

        # 노드 객체는 만들어진 뒤 바뀌지 않으므로 딕셔너리만 따로 두면 된다.
        # 깊은 복사는 임베딩까지 전부 두 벌로 만들어 메모리만 먹었다.
        all_nodes = dict(leaf_nodes)
        layer_to_nodes = {0: list(leaf_nodes.values())}
        logger.info(f"created {len(leaf_nodes)} leaf nodes")

        if on_layer_built:
            on_layer_built(0, layer_to_nodes[0])

        # 상위 노드 생성 및 트리 구축
        # 하위 클래스(ClusterTreeBuilder)에서 구현된 construct_tree 메서드 호출
        root_nodes = self.construct_tree(
            all_nodes, layer_to_nodes, use_multithreading, on_layer_built
        )
        final_depth = len(layer_to_nodes) - 1

        return Tree(
            all_nodes, root_nodes, leaf_nodes, final_depth, layer_to_nodes
        )

    @abstractmethod
    def construct_tree(
        self,
        all_tree_nodes: dict[int, Node],
        layer_to_nodes: dict[int, list[Node]],
        use_multithreading: bool = True,
        on_layer_built: LayerCallback | None = None,
    ) -> dict[int, Node]:
        pass
