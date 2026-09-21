from qdrant_client import models

from raptor_qdrant.database.qdrant_manager import QdrantManager
from raptor_qdrant.rag.engine import IndexHealth


def test_single_generation_is_a_finished_tree():
    health = IndexHealth(
        leaf_nodes=10, summary_nodes=3, leaves_outside_tree=0, generations=1
    )

    assert not health.has_mixed_generations
    assert "generations" not in health.summary


def test_mixed_generations_are_called_out():
    health = IndexHealth(
        leaf_nodes=10, summary_nodes=3, leaves_outside_tree=0, generations=2
    )

    assert health.has_mixed_generations
    assert "2 tree generations mixed" in health.summary


def test_drift_ratio_and_rebuild_threshold():
    fresh = IndexHealth(leaf_nodes=10, summary_nodes=2, leaves_outside_tree=1)
    stale = IndexHealth(leaf_nodes=10, summary_nodes=2, leaves_outside_tree=3)

    assert (fresh.drift, fresh.needs_rebuild) == (0.1, False)
    assert (stale.drift, stale.needs_rebuild) == (0.3, True)


def test_empty_index_has_no_drift():
    assert IndexHealth(0, 0, 0).drift == 0.0


def test_match_filter_selects_or_excludes_a_value():
    keep = QdrantManager._match_filter("tree_generation", "g1", negate=False)
    drop = QdrantManager._match_filter("tree_generation", "g1", negate=True)

    assert isinstance(keep.must[0], models.FieldCondition)
    assert keep.must[0].match == models.MatchValue(value="g1")
    assert keep.must_not is None
    assert drop.must is None
    assert drop.must_not[0].key == "tree_generation"
