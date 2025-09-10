from typing import Set

from typing import List, Optional


class Node:
    """
    Represents a node in the hierarchical tree structure.
    """

    def __init__(
        self, text: str, index: int, children: Set[int], embeddings
    ) -> None:
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings


class Tree:
    def __init__(
        self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes
    ):
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes

        # <<<--- 이 부분을 추가해주세요 ---
        # 노드 인덱스(ID)를 레이어 번호에 매핑하는 딕셔너리를 미리 만들어 둡니다.
        # 이렇게 하면 매번 검색할 필요 없이 빠르게 찾을 수 있습니다.
        self.index_to_layer = {
            node.index: layer
            for layer, nodes in self.layer_to_nodes.items()
            for node in nodes
        }
        # --- 여기까지 추가 ---

    def get_all_nodes_at_level(self, level: int) -> List[Node]:
        """
        주어진 레벨에 있는 모든 노드를 반환합니다.
        """
        return self.layer_to_nodes.get(level, [])

    # <<<--- 이 메서드를 추가해주세요 ---
    def get_node_layer(self, node_index: int) -> Optional[int]:
        """
        주어진 노드 인덱스(ID)에 해당하는 레이어 번호를 반환합니다.
        """
        return self.index_to_layer.get(node_index)

    # --- 여기까지 추가 ---

    def __repr__(self) -> str:
        return f"Tree(num_layers={self.num_layers}, num_nodes={len(self.all_nodes)})"
