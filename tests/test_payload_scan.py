from typing import Any

from raptor_qdrant.database.qdrant_manager import QdrantManager


class FakeCount:
    def __init__(self, count: int) -> None:
        self.count = count


class FakeClient:
    def __init__(self) -> None:
        self.scrolls: list[dict[str, Any]] = []
        self.payloads: list[dict[str, Any]] = []
        self.counts = 0

    def collection_exists(self, collection_name: str) -> bool:
        return True

    def scroll(self, **kwargs: Any) -> tuple[list[Any], None]:
        self.scrolls.append(kwargs)
        return [], None

    def count(self, **kwargs: Any) -> FakeCount:
        return FakeCount(self.counts)

    def set_payload(self, **kwargs: Any) -> None:
        self.payloads.append(kwargs)


def scanning(monkeypatch) -> tuple[QdrantManager, FakeClient]:
    manager = QdrantManager()
    client = FakeClient()
    monkeypatch.setattr(manager, "client", client)
    return manager, client


def test_scan_asks_only_for_the_requested_fields(monkeypatch):
    manager, client = scanning(monkeypatch)

    list(
        manager.iter_payloads("obsidian", fields=("layer", "tree_generation"))
    )

    assert client.scrolls[0]["with_payload"] == ["layer", "tree_generation"]


def test_scan_without_fields_takes_the_whole_payload(monkeypatch):
    manager, client = scanning(monkeypatch)

    list(manager.iter_payloads("obsidian"))

    assert client.scrolls[0]["with_payload"] is True


def test_listing_documents_never_pulls_the_note_text(monkeypatch):
    manager, client = scanning(monkeypatch)

    manager.list_document_names("obsidian")

    assert client.scrolls[0]["with_payload"] == ["document_name"]


def test_marking_points_sets_the_payload_on_the_match(monkeypatch):
    manager, client = scanning(monkeypatch)
    client.counts = 3

    marked = manager.set_payload_where(
        "obsidian", "source_notes", "지워진.md", {"stale": True}
    )

    assert marked == 3
    assert client.payloads[0]["payload"] == {"stale": True}


def test_marking_nothing_touches_no_points(monkeypatch):
    manager, client = scanning(monkeypatch)
    client.counts = 0

    assert (
        manager.set_payload_where(
            "obsidian", "source_notes", "없는.md", {"stale": True}
        )
        == 0
    )
    assert client.payloads == []
