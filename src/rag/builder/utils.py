import numpy as np
from typing import Dict, List

from .models.structure import Node


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    딕셔너리 형태의 노드들을 정렬된 리스트로 변환

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_text(node_list: List[Node]) -> str:
    """
    Node 리스트에서 모든 텍스트를 추출해서 하나의 문자열로 합침

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text
