from fakes import CountingEmbedding, FixedSummary, WholeNoteChunker
from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.builder.tree_builder import TreeBuilderConfig
from raptor_qdrant.rag.constants import SUMMARIZATION_MAX_WORKERS


def config(**overrides) -> TreeBuilderConfig:
    return TreeBuilderConfig(
        summarization_model=FixedSummary(),
        embedding_models={"fake": CountingEmbedding()},
        cluster_embedding_model="fake",
        chunker=WholeNoteChunker(),
        **overrides,
    )


def test_provider_default_applies_when_nothing_is_set(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "SUMMARY_WORKERS", 0)

    assert (
        config().summarization_max_workers
        == SUMMARIZATION_MAX_WORKERS["ollama"]
    )


def test_setting_overrides_the_provider_default(monkeypatch):
    monkeypatch.setattr(settings, "SUMMARY_WORKERS", 3)

    assert config().summarization_max_workers == 3


def test_explicit_argument_wins_over_the_setting(monkeypatch):
    monkeypatch.setattr(settings, "SUMMARY_WORKERS", 3)

    assert config(summarization_max_workers=5).summarization_max_workers == 5
