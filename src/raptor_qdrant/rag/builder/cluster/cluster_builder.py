import concurrent.futures
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

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
    SUMMARY_PROGRESS_STEPS,
)
from raptor_qdrant.rag.summarizer import is_unusable_summary
from raptor_qdrant.rag.utils import count_tokens

from .raptor_clustering import RaptorClustering

logger = logging.getLogger(__name__)

Cluster = list[Node]


def _covered_sources(cluster: Cluster) -> list[str]:
    """클러스터가 덮는 원본 문서 이름을 모은다."""
    sources: set[str] = set()
    for node in cluster:
        metadata = node.metadata or {}
        sources.update(metadata.get(SOURCE_SET_KEY) or [])
        source = metadata.get(SOURCE_KEY)
        if source:
            sources.add(source)
    return sorted(sources)


class _ProgressReporter:
    """정해진 단계마다 한 줄씩 진행률을 남긴다. 클러스터마다 찍으면 로그가 넘친다."""

    def __init__(
        self, label: str, total: int, steps: int = SUMMARY_PROGRESS_STEPS
    ):
        self.label = label
        self.total = total
        self.done = 0
        self.every = max(1, total // steps)

    def step(self) -> None:
        self.done += 1
        if self.done == self.total or self.done % self.every == 0:
            logger.info(
                f"{self.label}: {self.done}/{self.total} clusters summarized "
                f"({self.done / self.total:.0%})"
            )


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
        """잎에서 시작해 클러스터링과 요약을 반복하며 위로 레이어를 쌓는다."""
        current_level_nodes = {node.index: node for node in layer_to_nodes[0]}
        next_node_index = max(all_tree_nodes, default=-1) + 1

        for layer in range(self.num_layers):
            nodes = get_node_list(current_level_nodes)
            if self._too_few_to_cluster(nodes):
                logger.info(
                    f"stopping at layer {layer}: {len(nodes)} nodes are too "
                    "few to cluster"
                )
                break

            target = layer + 1
            clusters = self._cluster(nodes)
            logger.info(
                f"layer {target}: summarizing {len(clusters)} clusters "
                f"from {len(nodes)} nodes"
            )

            summaries = self._summarize_clusters(
                clusters, target, use_multithreading
            )
            if not summaries:
                logger.warning(
                    f"layer {target}: no usable summaries, stopping here"
                )
                break

            new_level_nodes = self.create_nodes(
                next_node_index,
                [
                    self._summary_text_node(cluster, text)
                    for cluster, text in summaries
                ],
                children=[
                    {node.index for node in cluster}
                    for cluster, _ in summaries
                ],
                label=f"layer {target} summaries",
            )
            next_node_index += len(new_level_nodes)

            layer_to_nodes[target] = list(new_level_nodes.values())
            all_tree_nodes.update(new_level_nodes)
            current_level_nodes = new_level_nodes
            logger.info(
                f"layer {target}: built {len(new_level_nodes)} summary nodes"
            )

            if on_layer_built:
                on_layer_built(target, layer_to_nodes[target])

        return current_level_nodes

    def _cluster(self, nodes: list[Node]) -> list[Cluster]:
        clustering = self.clustering_algorithm(
            reduction_dimension=self.reduction_dimension,
            **self.clustering_params,
        )
        return clustering.perform_clustering(
            nodes, self.cluster_embedding_model
        )

    def _summarize_clusters(
        self,
        clusters: Sequence[Cluster],
        layer: int,
        parallel: bool,
    ) -> list[tuple[Cluster, str]]:
        """클러스터를 요약하고 쓸 만한 것만 돌려준다.

        예전에는 레이어 전체에 10분 제한을 걸고, 넘기면 성공한 요약까지 버린
        채 처음부터 단일 스레드로 다시 돌렸다. 레이어 하나가 수십 분 걸리는
        볼트에선 매번 두 배 일을 하는 셈이었다. 이제 전체 제한은 없고(호출별
        제한은 LLM 클라이언트가 가진다), 실패한 클러스터만 한 번 더 시도한다.
        """
        total = len(clusters)
        results: dict[int, str] = {}
        failed: list[int] = []
        progress = _ProgressReporter(f"layer {layer}", total)

        def run(i: int) -> str:
            return self.summarize(get_text(clusters[i]))

        workers = min(self.summarization_max_workers, total)
        if parallel and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(run, i): i for i in range(total)}
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        results[i] = future.result()
                    except Exception as e:
                        logger.error(f"layer {layer}: cluster {i} failed: {e}")
                        failed.append(i)
                    progress.step()
        else:
            for i in range(total):
                try:
                    results[i] = run(i)
                except Exception as e:
                    logger.error(f"layer {layer}: cluster {i} failed: {e}")
                    failed.append(i)
                progress.step()

        for i in failed:
            try:
                results[i] = run(i)
            except Exception as e:
                logger.error(
                    f"layer {layer}: cluster {i} failed again, dropping it: {e}"
                )

        usable: list[tuple[Cluster, str]] = []
        skipped = 0
        for i, cluster in enumerate(clusters):
            text = results.get(i)
            if text is None or is_unusable_summary(text):
                skipped += 1
                logger.info(
                    f"layer {layer}: skipping cluster {i}: unusable summary "
                    f"{(text or '').strip()[:40]!r}"
                )
                continue
            logger.debug(f"layer {layer}: cluster {i} summary: {text}")
            usable.append((cluster, text))

        if skipped:
            logger.info(
                f"layer {layer}: {skipped}/{total} clusters had no usable "
                "summary"
            )
        return usable

    @staticmethod
    def _summary_text_node(cluster: Cluster, text: str) -> TextNode:
        metadata = ChunkMetadata(
            chunked_by=ChunkingMethod.SUMMARY, token_count=count_tokens(text)
        )
        return TextNode(
            text=text,
            metadata={
                **metadata.to_dict(),
                SOURCE_SET_KEY: _covered_sources(cluster),
            },
        )
