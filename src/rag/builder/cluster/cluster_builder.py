import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List

import tiktoken
from .raptor_clustering import RaptorClustering
from src.rag.builder.models.structure import Node, Tree
from src.rag.builder.tree_builder import TreeBuilder, TreeBuilderConfig
from src.rag.builder.utils import (
    get_node_list,
    get_text,
)

logger = logging.getLogger(__name__)


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,  # 차원 축소의 목표 차원 수
        clustering_algorithm=RaptorClustering,  # 클러스터링 알고리즘
        clustering_params={},  # 선택한 클러스터링 알고리즘에 전달할 추가 매개변수
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params

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
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def construct_tree(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        """TreeBuilder의 추상 메서드 구현
        리프 노드로부터 시작하여 클러스터링과 요약을 반복하며 상위 레이어의 노드를 생성
        """

        # 시작점은 레이어 0, 즉 리프 노드
        current_level_nodes = {node.index: node for node in layer_to_nodes[0]}
        # 새로 만들 노드의 인덱스는 기존 노드 수 다음 번호부터 시작
        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster: List[Node],
            new_level_nodes: Dict[int, Node],
            node_index: int,
            summarization_length: int,
            lock: Lock,
        ):
            """하나의 클러스터를 처리하여 요약 노드를 생성하고 새로운 부모 노드를 생성"""
            # 클러스터 내 모든 노드의 텍스트를 하나로 합침
            node_texts = get_text(cluster)
            # 합쳐진 텍스트를 요약하여 부모 노드의 텍스트로 사용
            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )
            logging.info(
                f"summarized text for node {node_index}: {summarized_text[:100]}..."
            )

            # 요약된 텍스트와 자식 노드 인덱스를 사용해 새로운 부모 노드 생성
            _, new_parent_node = self.create_node(
                node_index,
                summarized_text,
                {node.index for node in cluster},
            )

            with lock:
                new_level_nodes[node_index] = new_parent_node

        # 설정된 레이어 수만큼 아래에서 위로 반복하여 트리를 구축
        for layer in range(self.num_layers):
            new_level_nodes = {}
            logging.info(f"constructing layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            # 노드가 너무 적으면 의미 있는 클러스터링이 불가능하므로 중단
            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    "stopping at layer {layer} due to insufficient nodes for clustering"
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
            summarization_length = self.summarization_length

            if use_multithreading:
                # AWS Bedrock API 제한을 고려하여 동시 요청 수를 제한
                max_workers = min(10, len(clusters))  # 최대 10개 동시 요청
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            # Create tree structure for tracking (not returned)
            _ = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        return current_level_nodes
