import pytest

from raptor_qdrant.cli import (
    as_documents,
    as_hashes,
    catch_up,
    format_elapsed,
    guard_remote_llm,
    load_vault,
    summary_model_override,
)
from raptor_qdrant.rag.engine import QueryResult
from raptor_qdrant.vault.loader import VaultLoader, VaultNote


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


class FakeEngine:
    def __init__(self) -> None:
        self.upserted: dict[str, str] = {}
        self.removed: list[str] = []

    def upsert_notes(self, documents, note_hashes=None) -> int:
        self.upserted.update(documents)
        return len(documents)

    def remove_notes(self, paths) -> int:
        self.removed.extend(paths)
        return len(self.removed)


def test_catch_up_applies_only_what_changed_during_the_build(tmp_path):
    (tmp_path / "a.md").write_text("처음 내용")
    (tmp_path / "b.md").write_text("그대로인 내용")
    (tmp_path / "gone.md").write_text("지워질 내용")
    loader = VaultLoader(tmp_path)
    snapshot = as_hashes(loader.load())

    (tmp_path / "a.md").write_text("빌드 중 바뀐 내용")
    (tmp_path / "c.md").write_text("빌드 중 생긴 노트")
    (tmp_path / "gone.md").unlink()
    engine = FakeEngine()

    catch_up(engine, loader, snapshot)

    assert set(engine.upserted) == {"a.md", "c.md"}
    assert engine.removed == ["gone.md"]


def test_catch_up_does_nothing_when_the_vault_is_unchanged(tmp_path):
    (tmp_path / "a.md").write_text("내용")
    loader = VaultLoader(tmp_path)
    engine = FakeEngine()

    catch_up(engine, loader, as_hashes(loader.load()))

    assert not engine.upserted and not engine.removed


def test_elapsed_is_shown_without_microseconds():
    from datetime import timedelta

    assert format_elapsed(timedelta(hours=2, minutes=3, seconds=4.5)) == (
        "2:03:04"
    )


@pytest.mark.parametrize(
    ("provider", "answer_model", "summary_model", "expected"),
    [
        ("ollama", "qwen2.5:7b", "qwen2.5:3b", "qwen2.5:3b"),
        ("ollama", "qwen2.5:7b", "qwen2.5:7b", None),
        ("ollama", "qwen2.5:7b", "", None),
        ("ollama", "qwen2.5:7b", "   ", None),
        ("bedrock", "qwen2.5:7b", "qwen2.5:3b", None),
    ],
)
def test_summary_model_override(
    monkeypatch, provider, answer_model, summary_model, expected
):
    from raptor_qdrant.core.config import settings

    monkeypatch.setattr(settings, "OLLAMA_MODEL", answer_model)
    monkeypatch.setattr(settings, "OLLAMA_SUMMARY_MODEL", summary_model)

    assert summary_model_override(provider) == expected
