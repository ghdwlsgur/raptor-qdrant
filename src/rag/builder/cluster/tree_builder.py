import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List

import tiktoken
from .utils import RaptorClustering
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


# 계층적 트리를 클러스터링 방식으로 구축하는 클래스
"""
1. 비슷한 내용의 노드들을 그룹화(클러스터링)
2. 각 클러스터의 내용을 요약하여 상위 노드(요약 노드) 생성
3. 이 과정을 반복하여 트리의 최상위 노드에 도달
"""


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logging.info(
            f"initialized ClusterTreeBuilder with config: {config.log_config()}"
        )

    def construct_tree(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("using cluster tree builder")

        # Initialize current_level_nodes with leaf nodes (layer 0)
        current_level_nodes = {node.index: node for node in layer_to_nodes[0]}
        # 새로 만들 노드의 인덱스는 기존 노드 수에서 시작
        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster,
            new_level_nodes,
            next_node_index,
            summarization_length,
            lock,
        ):
            node_texts = get_text(cluster)
            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logging.info(
                f"node texts character length: {len(node_texts)}"
            )
            logging.info(
                f"summarized text character length: {len(summarized_text)}"
            )

            _, new_parent_node = self.create_node(
                next_node_index,
                summarized_text,
                {node.index for node in cluster},
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        # 설정된 레이어 수만큼 반복
        for layer in range(self.num_layers):
            new_level_nodes = {}
            logging.info(f"constructing layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            # 노드가 너무 적으면 더 이상 레이어를 만들 수 없으므로 중지
            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    "stopping layer construction: cannot create more layers"
                )
                break

            # 현재 레이어의 노드들을 클러스터링
            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                tokenizer=self.tokenizer,
                **self.clustering_params,
            )

            lock = Lock()
            summarization_length = self.summarization_length
            logging.info(f"summarization length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
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
