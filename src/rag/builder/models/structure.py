from typing import List, Optional, Set, Dict, Any
from dataclasses import dataclass, field

"""RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

1. 문서 분할(Chunking): 긴 문서를 작은 텍스트 조각(chunk)으로 나눔, 이 조각들이 트리의 가장 아래 레벨인
'잎 노드(leaf nodes)'가 됨.
2. 요약 및 클러스터링: 비슷한 잎 노드들을 묶어 클러스터링하고 그 내용을 요약함. 이 내용으로 '부모 노드'를 만듦
3. 반복: 이 과정을 트리 꼭대기(root)에 도달할 때까지 반복하여 여러 계층(layer)을 가진 트리 구조를 완성
"""


@dataclass
class Node:
    text: str  # 노드가 담고 있는 텍스트 내용
    index: int  # 모든 노드를 구분하기 위한 고유 인덱스
    children: Set[int]  # 자식 노드들의 인덱스를 저장
    embeddings: Any  # 텍스트를 벡터로 변환한 임베딩 값
    metadata: Optional[Dict[str, Any]] = field(
        default_factory=dict, compare=False
    )  # ChunkMetadata 구조를 따르는 메타데이터


@dataclass
class Tree:
    all_nodes: List[Node]  # 트리에 포함된 모든 노드 객체
    root_nodes: List[Node]  # 트리의 최상위 노드
    leaf_nodes: List[Node]  # 트리의 최하위 노드
    num_layers: int  # 트리의 레이어 개수
    layer_to_nodes: Dict[
        int, List[Node]
    ]  # 각 레이어 번호에 어떤 노드들이 속해있는지가 매핑된 딕셔너리
    index_to_layer: Dict[int, int] = field(init=False)

    def __post_init__(self):
        self.index_to_layer = {
            node.index: layer
            for layer, nodes in self.layer_to_nodes.items()
            for node in nodes
        }

    def get_all_nodes_at_level(self, level: int) -> List[Node]:
        """주어진 레벨에 있는 모든 노드를 반환"""
        return self.layer_to_nodes.get(level, [])

    def get_node_layer(self, node_index: int) -> Optional[int]:
        """주어진 노드 인덱스(ID)에 해당하는 레이어 번호를 반환"""
        return self.index_to_layer.get(node_index)

    def __repr__(self) -> str:
        return f"Tree(num_layers={self.num_layers}, num_nodes={len(self.all_nodes)})"
