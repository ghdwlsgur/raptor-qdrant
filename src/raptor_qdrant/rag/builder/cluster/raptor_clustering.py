import logging
from abc import ABC, abstractmethod

import numpy as np
import umap

from raptor_qdrant.rag.builder.models.structure import Node
from raptor_qdrant.rag.constants import (
    CLUSTER_MAX_RECURSION_DEPTH,
    CLUSTER_MAX_TOKENS,
    CLUSTER_MIN_NODES_TO_SPLIT,
    CLUSTER_PROBABILITY_THRESHOLD,
    CLUSTER_REDUCTION_DIMENSION,
    UMAP_LOCAL_MAX_NEIGHBORS,
)
from raptor_qdrant.rag.utils import resolve_token_count

from .utils import (
    RANDOM_SEED,
    gmm_soft_cluster,
    reduce_embedding_dimensions,
)

logger = logging.getLogger(__name__)


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(
        self, nodes: list[Node], embedding_model_name: str
    ) -> list[list[Node]]:
        """Nodes 객체 리스트를 받아 그룹화된 Nodes 객체들의 리스트를 반환"""
        pass


class RaptorClustering(ClusteringAlgorithm):
    def __init__(
        self,
        max_length_in_cluster: int = CLUSTER_MAX_TOKENS,
        reduction_dimension: int = CLUSTER_REDUCTION_DIMENSION,
        threshold: float = CLUSTER_PROBABILITY_THRESHOLD,
    ):
        """
        RaptorClustering 인스턴스를 생성할 때 필요한 설정을 초기화하고 저장

        :param max_length_in_cluster: 한 클러스터가 가질 수 있는 최대 토큰 길이
        :param reduction_dimension: UMAP 차원 축소 시 목표 차원
        :param threshold: gmm_soft_cluster 함수에서 사용하는 임계값
        """
        self.max_length_in_cluster = max_length_in_cluster
        self.reduction_dimension = reduction_dimension
        self.threshold = threshold

    def _hierarchical_cluster(
        self, embeddings: np.ndarray
    ) -> list[np.ndarray]:
        """각 임베딩이 어떤 클러스터에 속하는지를 나타내는 인덱스 배열의 리스트를 반환

        자리(위치 인덱스)를 들고 다닌다. 임베딩 값을 키로 원래 자리를 되찾으려
        하면 1024 차원짜리 튜플을 노드 수만큼 쥐고 있어야 하고(잎 만 개면
        수백 MB), 값이 똑같은 노드 둘은 한 자리로 뭉개져 한쪽이 어느
        클러스터에도 못 들어간 채 상위 레이어에서 조용히 빠진다.
        """
        # 전체 임베딩의 차원을 축소하여 전체 구조 파악
        global_reduced_embeddings = reduce_embedding_dimensions(
            embeddings, min(self.reduction_dimension, len(embeddings) - 2)
        )
        # 차원이 축소된 임베딩을 사용해 전체적인 분포를 n_global_clusters 개로 나눔
        global_clusters, n_global_clusters = gmm_soft_cluster(
            global_reduced_embeddings, self.threshold
        )
        logger.info(f"global clusters found: {n_global_clusters}")

        # 최종 클러스터 단계를 저장할 리스트 초기화
        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        # 전체 클러스터 ID가 중복되지 않도록 관리하는 카운터 변수
        total_clusters = 0

        # 각 Global 클러스터를 순회하며 내부를 더 작은 클러스터로 세분화
        for i in range(n_global_clusters):
            positions = np.flatnonzero([i in gc for gc in global_clusters])
            if positions.size == 0:
                continue

            global_cluster_embeddings = embeddings[positions]

            # Global 클러스터가 너무 작으면 Local 클러스터링을 생략하고 단일 클러스터로 처리
            if len(positions) <= self.reduction_dimension + 1:
                local_clusters = [np.array([0])] * len(positions)
                n_local_clusters = 1
            else:
                # n_neighbors를 데이터 크기에 맞게 동적으로 설정
                n_neighbors = max(
                    2, min(UMAP_LOCAL_MAX_NEIGHBORS, len(positions) - 1)
                )
                reduced_embeddings_local = umap.UMAP(
                    n_neighbors=n_neighbors,
                    n_components=self.reduction_dimension,
                    metric="cosine",
                    random_state=RANDOM_SEED,
                ).fit_transform(global_cluster_embeddings)

                local_clusters, n_local_clusters = gmm_soft_cluster(
                    reduced_embeddings_local, self.threshold
                )

            # Local 클러스터링 결과를 전체 데이터의 자리에 다시 매핑
            for j in range(n_local_clusters):
                members = positions[
                    np.flatnonzero([j in lc for lc in local_clusters])
                ]
                for position in members:
                    # 중복되지 않는 고유한 클러스터 ID를 부여
                    all_local_clusters[position] = np.append(
                        all_local_clusters[position], j + total_clusters
                    )

            # 다음 Global 클러스터의 ID가 겹치지 않도록 offset 업데이트
            total_clusters += n_local_clusters

        logger.info(f"total local clusters found: {total_clusters}")
        return all_local_clusters

    def _get_node_token_count(self, node: Node) -> int:
        """노드의 토큰 수를 메타데이터에서 가져오되, 없으면 텍스트로 계산

        메타데이터를 그대로 믿고 None 을 합산하면 TypeError 로 클러스터링 전체가
        멈춘다. resolve_token_count 가 값이 없는 경우를 대신 계산해 준다.
        """
        return resolve_token_count(node.metadata, node.text)

    def perform_clustering(
        self,
        nodes: list[Node],
        embedding_model_name: str,
        recursion_depth: int = 0,
    ) -> list[list[Node]]:
        """Node 객체 리스트를 받아 그룹화된 Node 객체들의 리스트를 반환
        클러스터링의 전체 과정을 지휘하고 숫자 데이터를 다루는 _hierarchical_cluster 메서드
        를 호출하고, 그 결과를 다시 Node 객체로 매핑. 또한, 클러스터의 크기가
        너무 클 경우 재귀적으로 자신을 호출하여 클러스터를 더 작은 단위로 재분할

        Args:
            nodes (List[Node]): 클러스터링을 수행할 대상이 되는 Node 객체들의 리스트입니다.
            embedding_model_name (str): 각 Node 객체의 .embeddings 딕셔너리에서 어떤
                임베딩 벡터를 사용할지 지정하는 모델의 이름(키)입니다.

        Returns:
            List[List[Node]]: 클러스터링이 완료된 후, 그룹화된 Node 객체들의 리스트입니다.
                바깥쪽 리스트는 모든 클러스터들을 담고 있으며, 각각의 안쪽 리스트는
                하나의 클러스터에 속한 Node 객체들을 담고 있습니다.
                e.g., [[Node1, Node5], [Node2, Node3, Node4], ...]
        """
        # 입력으로 받은 노드 리스트가 비어있으면 빈 리스트 반환
        if not nodes:
            return []

        # 노드 수가 너무 적으면 더 이상 분할하지 않음
        if len(nodes) <= CLUSTER_MIN_NODES_TO_SPLIT:
            return [nodes]

        # 각 Node 객체에서 저장된 모델의 임베딩 벡터를 추출하여 Numpy 배열로 변환 (객체 -> 숫자)
        embeddings = np.array(
            [node.embeddings[embedding_model_name] for node in nodes]
        )

        # 계층적 클러스터링 수행, 각 노드가 어떤 클러스터 ID에 속하는지 나타내는 인덱스 배열의 리스트를 반환
        clusters_indices = self._hierarchical_cluster(embeddings)

        # 클러스터 ID를 key로, Node 리스트를 value로 갖는 딕셔너리를 생성
        clusters_map: dict[int, list[Node]] = {}
        for i, label_array in enumerate(clusters_indices):
            for label in label_array.astype(int):
                if label not in clusters_map:
                    clusters_map[label] = []
                clusters_map[label].append(nodes[i])

        # 딕셔너리의 value들로부터 초기 클러스터 리스트를 생성
        initial_clusters = list(clusters_map.values())

        node_clusters = []
        for cluster_nodes in initial_clusters:
            # ============================================== Base Case
            # 클러스터에 노드가 하나뿐이라면 더 이상 분할할 수 없으므로 최종 클러스터 리스트에 추가
            if len(cluster_nodes) <= 1:
                node_clusters.append(cluster_nodes)
                continue

            # 현재 클러스터에 속한 모든 노드들의 텍스트 길이를 토큰 단위로 합산
            total_length = sum(
                [self._get_node_token_count(node) for node in cluster_nodes]
            )

            # 클러스터가 너무 길고, 노드 수가 3개 이상이고, 재귀 깊이가 한계 내일 때만 재분할
            if (
                total_length > self.max_length_in_cluster
                and len(cluster_nodes) > CLUSTER_MIN_NODES_TO_SPLIT
                and recursion_depth < CLUSTER_MAX_RECURSION_DEPTH
            ):
                logger.info(
                    f"reclustering cluster with {len(cluster_nodes)} nodes (depth: {recursion_depth})"
                )
                # 재귀 호출의 결과는 하나 이상의 작은 클러스터이므로 extend 사용
                sub_clusters = self.perform_clustering(
                    cluster_nodes,
                    embedding_model_name,
                    recursion_depth + 1,
                )

                # 재귀 호출 결과가 원본과 동일하면 무한루프 방지를 위해 강제 종료
                if len(sub_clusters) == 1 and len(sub_clusters[0]) == len(
                    cluster_nodes
                ):
                    logger.warning(
                        f"clustering did not improve, forcing termination at depth {recursion_depth}"
                    )
                    node_clusters.append(cluster_nodes)
                else:
                    node_clusters.extend(sub_clusters)
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
