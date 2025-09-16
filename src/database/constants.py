"""Qdrant 관련 상수들"""
from qdrant_client import models

# Sparse Vector 설정 상수
SPARSE_VECTOR_NAME = "text-sparse-new"
DEFAULT_SPARSE_VECTOR_CONFIG = {
    SPARSE_VECTOR_NAME: models.SparseVectorParams(
        index=models.SparseIndexParams(on_disk=False)
    )
}

# Dense Vector 기본 설정
DEFAULT_DISTANCE_METRIC = models.Distance.COSINE


def create_vector_config(size: int, distance: models.Distance = DEFAULT_DISTANCE_METRIC) -> models.VectorParams:
    """Dense vector 설정을 생성합니다."""
    return models.VectorParams(size=size, distance=distance)


def get_sparse_vector_config() -> dict:
    """Sparse vector 설정을 반환합니다."""
    return DEFAULT_SPARSE_VECTOR_CONFIG.copy()