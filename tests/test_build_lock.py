import json
import subprocess
import sys

import pytest

from raptor_qdrant.vault.lock import BuildInProgress, BuildLock


def make_lock(tmp_path) -> BuildLock:
    return BuildLock.for_collection("obsidian", tmp_path)


def dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_missing_lock_is_not_held(tmp_path):
    lock = make_lock(tmp_path)

    assert lock.read() is None
    assert not lock.is_held()
    assert lock.stale() is None


def test_acquire_records_the_build_and_holds(tmp_path):
    lock = make_lock(tmp_path)

    info = lock.acquire("obsidian", "gen-1")

    assert lock.is_held()
    assert (info.collection, info.generation) == ("obsidian", "gen-1")
    assert lock.read() == info
    assert info.elapsed.total_seconds() >= 0


def test_live_lock_refuses_another_build(tmp_path):
    lock = make_lock(tmp_path)
    lock.acquire("obsidian", "gen-1")

    with pytest.raises(BuildInProgress, match="already running"):
        lock.acquire("obsidian", "gen-2")


def test_dead_owner_makes_the_lock_stale(tmp_path):
    lock = make_lock(tmp_path)
    lock.path.parent.mkdir(parents=True, exist_ok=True)
    lock.path.write_text(
        json.dumps(
            {
                "pid": dead_pid(),
                "collection": "obsidian",
                "generation": "gen-old",
                "started_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )

    stale = lock.stale()

    assert not lock.is_held()
    assert stale is not None and stale.generation == "gen-old"
    assert lock.acquire("obsidian", "gen-new").generation == "gen-new"


def test_release_removes_the_file(tmp_path):
    lock = make_lock(tmp_path)
    lock.acquire("obsidian", "gen-1")

    lock.release()
    lock.release()  # 두 번 불러도 조용하다

    assert not lock.path.exists()


def test_hold_releases_even_when_the_build_fails(tmp_path):
    lock = make_lock(tmp_path)

    with pytest.raises(RuntimeError, match="boom"), lock.hold("obsidian", "g"):
        assert lock.is_held()
        raise RuntimeError("boom")

    assert not lock.path.exists()


@pytest.mark.parametrize("garbage", ["", "not json", '{"pid": "x"}', "[]"])
def test_unreadable_lock_is_ignored(tmp_path, garbage):
    lock = make_lock(tmp_path)
    lock.path.parent.mkdir(parents=True, exist_ok=True)
    lock.path.write_text(garbage)

    assert lock.read() is None
    assert not lock.is_held()


def test_collection_name_becomes_a_safe_file_name(tmp_path):
    lock = BuildLock.for_collection("팀/노트 v2", tmp_path)

    assert lock.path.parent == tmp_path
    assert "/" not in lock.path.name
    assert lock.path.name.endswith(".build.json")
