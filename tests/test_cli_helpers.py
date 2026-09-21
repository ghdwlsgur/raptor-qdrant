import pytest

from raptor_qdrant.cli import (
    as_documents,
    as_hashes,
    guard_remote_llm,
    load_vault,
)
from raptor_qdrant.rag.engine import QueryResult
from raptor_qdrant.vault.loader import VaultNote


def note(path: str) -> VaultNote:
    return VaultNote(
        path=path, title=path, text=f"{path} 본문", content_hash=f"h-{path}"
    )


def test_remote_provider_is_blocked_without_consent():
    assert not guard_remote_llm("bedrock", allowed=False)


def test_remote_provider_passes_with_consent():
    assert guard_remote_llm("bedrock", allowed=True)


@pytest.mark.parametrize("provider", ["ollama", "OLLAMA"])
def test_local_provider_needs_no_consent(provider):
    assert guard_remote_llm(provider, allowed=False)


def test_guard_falls_back_to_the_configured_provider(monkeypatch):
    from raptor_qdrant.core.config import settings

    monkeypatch.setattr(settings, "LLM_PROVIDER", "bedrock")

    assert not guard_remote_llm(None, allowed=False)


def test_documents_and_hashes_are_keyed_by_path():
    notes = [note("a.md"), note("메모/b.md")]

    assert as_documents(notes) == {
        "a.md": "a.md 본문",
        "메모/b.md": "메모/b.md 본문",
    }
    assert as_hashes(notes) == {
        "a.md": "h-a.md",
        "메모/b.md": "h-메모/b.md",
    }


def test_loading_a_vault_without_notes_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no indexable notes"):
        load_vault(str(tmp_path))


def test_query_result_dedupes_sources_in_order():
    result = QueryResult(
        question="q",
        answer="a",
        context="c",
        chunks=[
            {"sources": ["b.md", "a.md"]},
            {"sources": ["a.md", "c.md"]},
            {"sources": []},
        ],
    )

    assert result.sources == ["b.md", "a.md", "c.md"]


def test_query_result_without_sources():
    assert QueryResult("q", "a", "c").sources == []
