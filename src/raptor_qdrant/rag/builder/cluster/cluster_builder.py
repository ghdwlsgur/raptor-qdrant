import concurrent.futures
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from llama_index.core.schema import TextNode

from raptor_qdrant.rag.builder.models.structure import Node
from raptor_qdrant.rag.builder.tree_builder import (
    LayerCallback,
    TreeBuilder,
    TreeBuilderConfig,
)
from raptor_qdrant.rag.builder.utils import (
    get_node_list,
    get_text,
)
from raptor_qdrant.rag.chunker.models.chunk_metadata import (
    ChunkingMethod,
    ChunkMetadata,
)
from raptor_qdrant.rag.constants import (
    CLUSTER_REDUCTION_DIMENSION,
    SOURCE_KEY,
    SOURCE_SET_KEY,
)
from raptor_qdrant.rag.summarizer import is_unusable_summary
from raptor_qdrant.rag.utils import count_tokens

from .raptor_clustering import RaptorClustering

logger = logging.getLogger(__name__)


def _covered_sources(cluster: list[Node]) -> list[str]:
    """클러스터가 덮는 원본 문서 이름을 모은다."""
    sources: set[str] = set()
    for node in cluster:
        metadata = node.metadata or {}
        sources.update(metadata.get(SOURCE_SET_KEY) or [])
        source = metadata.get(SOURCE_KEY)
        if source:
            sources.add(source)
    return sorted(sources)


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=CLUSTER_REDUCTION_DIMENSION,  # 차원 축소의 목표 차원 수
        clustering_algorithm=RaptorClustering,  # 클러스터링 알고리즘
        clustering_params=None,  # 선택한 클러스터링 알고리즘에 전달할 추가 매개변수
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params or {}

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    """계층적 트리를 클러스터링 방식으로 구축하는 클래스
    1. 비슷한 내용의 노드들을 그룹화(클러스터링)
    2. 각 클러스터의 내용을 요약하여 상위 노드(요약 노드) 생성
    3. 이 과정을 반복하여 트리의 최상위 노드에 도달
    """

    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params

    def _too_few_to_cluster(self, nodes: list[Node]) -> bool:
        return len(nodes) <= self.reduction_dimension + 1

    def construct_tree(
        self,
        all_tree_nodes: dict[int, Node],
        layer_to_nodes: dict[int, list[Node]],
        use_multithreading: bool = False,
        on_layer_built: LayerCallback | None = None,
    ) -> dict[int, Node]:
        """TreeBuilder의 추상 메서드 구현
        리프 노드로부터 시작하여 클러스터링과 요약을 반복하며 상위 레이어의 노드를 생성
        """

        # 시작점은 레이어 0, 즉 리프 노드
        current_level_nodes = {node.index: node for node in layer_to_nodes[0]}
        # 새로 만들 노드의 인덱스는 기존 노드 수 다음 번호부터 시작
        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster: list[Node],
            new_level_nodes: dict[int, Node],
            node_index: int,
            lock: Lock,
        ):
            """하나의 클러스터를 처리하여 요약 노드를 생성하고 새로운 부모 노드를 생성"""
            # 클러스터 내 모든 노드의 텍스트를 하나로 합침
            node_texts = get_text(cluster)

            # 합쳐진 텍스트를 요약하여 부모 노드의 텍스트로 사용
            summarized_text = self.summarize(text=node_texts)

            # 요약이 불충분하면 해당 클러스터를 건너뜀
            if is_unusable_summary(summarized_text):
                logger.info(
                    f"skipping cluster {node_index}: unusable summary "
                    f"{summarized_text.strip()[:40]!r}"
                )
                return

            logger.info(
                f"summarized text for node {node_index}: {summarized_text}"
            )

            token_count = count_tokens(summarized_text)
            chunk_metadata = ChunkMetadata(
                chunked_by=ChunkingMethod.SUMMARY, token_count=token_count
            )
            summary_node = TextNode(
                text=summarized_text,
                metadata={
                    **chunk_metadata.to_dict(),
                    SOURCE_SET_KEY: _covered_sources(cluster),
                },
            )

            # 요약된 텍스트와 자식 노드 인덱스를 사용해 새로운 부모 노드 생성
            _, new_parent_node = self.create_node(
                node_index,
                summary_node,
                {node.index for node in cluster},
            )

            with lock:
                new_level_nodes[node_index] = new_parent_node

        # 설정된 레이어 수만큼 아래에서 위로 반복하여 트리를 구축
        for layer in range(self.num_layers):
            new_level_nodes: dict[int, Node] = {}
            logger.info(f"constructing layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if self._too_few_to_cluster(node_list_current_layer):
                logger.info(
                    f"stopping at layer {layer} due to insufficient nodes for clustering"
                )
                break

            # 현재 레이어의 노드들을 클러스터링
            clustering_instance = self.clustering_algorithm(
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params,
            )
            clusters = clustering_instance.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
            )

            lock = Lock()
            layer_start_index = next_node_index

            if use_multithreading:
                max_workers = min(
                    self.summarization_max_workers, len(clusters)
                )

                try:
                    with ThreadPoolExecutor(
                        max_workers=max_workers
                    ) as executor:
                        # 모든 작업을 한 번에 제출
                        futures = []
                        cluster_args = []

                        for cluster in clusters:
                            args = (
                                cluster,
                                new_level_nodes,
                                next_node_index,
                                lock,
                            )
                            cluster_args.append(args)
                            futures.append(
                                executor.submit(process_cluster, *args)
                            )
                            next_node_index += 1

                        # 모든 작업 완료까지 안전하게 대기
                        completed_count = 0
                        for future in concurrent.futures.as_completed(
                            futures, timeout=600
                        ):
                            try:
                                future.result(
                                    timeout=300
                                )  # 개별 작업 5분 타임아웃
                                completed_count += 1
                            except concurrent.futures.TimeoutError:
                                logger.error("cluster processing timed out")
                            except Exception as e:
                                logger.error(f"cluster processing failed: {e}")

                        logger.info(
                            f"successfully processed {completed_count}/{len(futures)} clusters"
                        )

                except Exception as e:
                    logger.error(
                        f"critical error in multithreaded cluster processing: {e}"
                    )
                    logger.info(
                        "falling back to single-threaded cluster processing"
                    )
                    new_level_nodes.clear()
                    next_node_index = layer_start_index
                    for cluster in clusters:
                        try:
                            process_cluster(
                                cluster,
                                new_level_nodes,
                                next_node_index,
                                lock,
                            )
                            next_node_index += 1
                        except Exception as cluster_error:
                            logger.error(
                                f"failed to process cluster in fallback mode: {cluster_error}"
                            )

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)
            if on_layer_built:
                on_layer_built(layer + 1, layer_to_nodes[layer + 1])

        return current_level_nodes
