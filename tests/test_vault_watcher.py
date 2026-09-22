import threading
from pathlib import Path

import pytest

from raptor_qdrant.vault.watcher import (
    _DebouncedMarkdownHandler,
    _markdown_path,
    changed_note_paths,
)


class FakeEvent:
    def __init__(self, src_path, dest_path=None):
        self.src_path = src_path
        self.dest_path = dest_path


@pytest.mark.parametrize(
    "raw",
    [
        "/vault/note.txt",
        "/vault/.obsidian/workspace.md",
        "/vault/.trash/old.md",
        "/vault/.git/COMMIT_EDITMSG.md",
        None,
        12,
    ],
)
def test_ignores_non_notes(raw):
    assert _markdown_path(raw) is None


@pytest.mark.parametrize("raw", ["/vault/a.md", b"/vault/b.md"])
def test_accepts_markdown(raw):
    assert _markdown_path(raw) is not None


def collect(debounce=0.05):
    seen: list[set[Path]] = []
    done = threading.Event()

    def on_settled(paths: set[Path]) -> None:
        seen.append(paths)
        done.set()

    return _DebouncedMarkdownHandler(on_settled, debounce), seen, done


def test_batches_rapid_saves_into_one_callback():
    handler, seen, done = collect()

    for _ in range(5):
        handler.on_any_event(FakeEvent("/vault/a.md"))
    handler.on_any_event(FakeEvent("/vault/b.md"))

    assert done.wait(2)
    assert seen == [{Path("/vault/a.md"), Path("/vault/b.md")}]


def test_rename_reports_both_sides():
    handler, seen, done = collect()

    handler.on_any_event(FakeEvent("/vault/old.md", "/vault/new.md"))

    assert done.wait(2)
    assert seen[0] == {Path("/vault/old.md"), Path("/vault/new.md")}


def test_flush_delivers_pending_immediately():
    handler, seen, _ = collect(debounce=60)

    handler.on_any_event(FakeEvent("/vault/a.md"))
    handler.flush()

    assert seen == [{Path("/vault/a.md")}]


def test_flush_without_pending_does_nothing():
    handler, seen, _ = collect(debounce=60)

    handler.flush()

    assert seen == []


def test_callback_errors_do_not_escape():
    def boom(_: set[Path]) -> None:
        raise RuntimeError("적용 실패")

    handler = _DebouncedMarkdownHandler(boom, 60)
    handler.on_any_event(FakeEvent("/vault/a.md"))

    handler.flush()


def test_paths_become_vault_relative(tmp_path):
    (tmp_path / "메모").mkdir()
    note = tmp_path / "메모" / "a.md"
    note.write_text("x", encoding="utf-8")

    assert changed_note_paths([note], tmp_path.resolve()) == {"메모/a.md"}


def test_paths_outside_the_vault_are_dropped(tmp_path):
    assert (
        changed_note_paths([Path("/elsewhere/a.md")], tmp_path.resolve())
        == set()
    )


def test_deferred_changes_collect_until_drained():
    from raptor_qdrant.vault.watcher import DeferredChanges

    held = DeferredChanges()

    assert held.add([Path("/v/a.md"), Path("/v/b.md")]) == 2
    assert held.add([Path("/v/a.md")]) == 2
    assert len(held) == 2

    assert held.drain() == {Path("/v/a.md"), Path("/v/b.md")}
    assert len(held) == 0
    assert held.drain() == set()


def test_tick_runs_periodically_while_watching(tmp_path):
    from raptor_qdrant.vault.watcher import watch_vault

    stop = threading.Event()
    ticks: list[int] = []

    def tick() -> None:
        ticks.append(1)
        if len(ticks) >= 3:
            stop.set()

    watch_vault(
        tmp_path,
        lambda paths: None,
        debounce_seconds=0.05,
        stop=stop,
        tick=tick,
        tick_seconds=0.02,
    )

    assert len(ticks) >= 3


def test_a_busy_file_does_not_hold_back_a_settled_one():
    import time

    handler, seen, _ = collect(debounce=60)

    handler.on_any_event(FakeEvent("/vault/quiet.md"))
    handler._deadlines[Path("/vault/quiet.md")] = time.monotonic()
    handler.on_any_event(FakeEvent("/vault/busy.md"))

    handler._sweep()

    assert seen == [{Path("/vault/quiet.md")}]
    assert Path("/vault/busy.md") in handler._deadlines


def test_a_still_busy_file_keeps_waiting():
    handler, seen, _ = collect(debounce=60)

    handler.on_any_event(FakeEvent("/vault/busy.md"))
    handler._sweep()

    assert seen == []
    assert Path("/vault/busy.md") in handler._deadlines
