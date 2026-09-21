import numpy as np
import pytest

from raptor_qdrant.rag.builder.cluster.utils import (
    get_optimal_cluster_count,
    gmm_soft_cluster,
    reduce_embedding_dimensions,
)


@pytest.fixture
def two_far_apart_blobs() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.vstack(
        [rng.normal(0, 0.1, (20, 8)), rng.normal(10, 0.1, (20, 8))]
    )


@pytest.mark.parametrize("size", [1, 2, 3])
def test_too_few_points_collapse_to_one_cluster(size):
    assert get_optimal_cluster_count(np.zeros((size, 4))) == 1


def test_finds_more_than_one_cluster_in_separated_data(two_far_apart_blobs):
    assert get_optimal_cluster_count(two_far_apart_blobs) > 1


def test_cluster_count_never_exceeds_half_the_points():
    rng = np.random.default_rng(1)
    points = rng.normal(0, 1, (10, 4))

    assert get_optimal_cluster_count(points) <= 5


def test_soft_cluster_labels_every_point(two_far_apart_blobs):
    labels, n_clusters = gmm_soft_cluster(two_far_apart_blobs, threshold=0.5)

    assert len(labels) == len(two_far_apart_blobs)
    assert n_clusters >= 1
    assert all(isinstance(label, np.ndarray) for label in labels)


def test_soft_cluster_ids_stay_within_range(two_far_apart_blobs):
    labels, n_clusters = gmm_soft_cluster(two_far_apart_blobs, threshold=0.5)

    assert all((label < n_clusters).all() for label in labels if label.size)


@pytest.mark.parametrize("size", [1, 2])
def test_reduction_passes_tiny_inputs_through(size):
    embeddings = np.ones((size, 16))

    assert len(reduce_embedding_dimensions(embeddings, dim=4)) == size
