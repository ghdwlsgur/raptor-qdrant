"""전체 재구축이 돌고 있는지 알려주는 락 파일.

index 가 몇 시간 도는 동안 sync 나 watch 가 같은 컬렉션을 건드리면 서로
느려지고, 재구축이 끝나며 세대를 정리할 때 그 사이 넣은 잎이 지워진다.
컬렉션마다 락 파일 하나를 두고 pid 와 세대를 적어둔다. pid 가 죽어 있으면
끝나지 않은 빌드가 남긴 흔적이므로, 다음 index 가 그 세대를 치우고 시작한다.
"""

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class BuildInProgress(RuntimeError):
    """같은 컬렉션의 재구축이 이미 돌고 있다."""


@dataclass(frozen=True)
class BuildInfo:
    pid: int
    collection: str
    generation: str
    started_at: str  # ISO 8601, UTC

    @property
    def is_live(self) -> bool:
        return _pid_alive(self.pid)

    @property
    def elapsed(self) -> timedelta:
        started = datetime.fromisoformat(self.started_at)
        return datetime.now(UTC) - started


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _filesystem_safe(name: str) -> str:
    cleaned = "".join(
        c if c.isalnum() or c in "-_." else "_" for c in name.strip()
    )
    return cleaned or "default"


class BuildLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def for_collection(
        cls, collection: str, state_dir: str | Path
    ) -> "BuildLock":
        root = Path(state_dir).expanduser()
        return cls(root / f"{_filesystem_safe(collection)}.build.json")

    def read(self) -> BuildInfo | None:
        """락 파일 내용. 없거나 망가졋으면 None."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            logger.warning(f"unreadable build lock at {self.path}, ignoring")
            return None

        try:
            return BuildInfo(
                pid=int(raw["pid"]),
                collection=str(raw["collection"]),
                generation=str(raw["generation"]),
                started_at=str(raw["started_at"]),
            )
        except (KeyError, TypeError, ValueError):
            logger.warning(f"malformed build lock at {self.path}, ignoring")
            return None

    def is_held(self) -> bool:
        """살아 있는 프로세스가 잡고 있는가."""
        info = self.read()
        return bool(info and info.is_live)

    def stale(self) -> BuildInfo | None:
        """죽은 프로세스가 남긴 락. 끝나지 않은 빌드의 흔적이다."""
        info = self.read()
        if info and not info.is_live:
            return info
        return None

    def acquire(self, collection: str, generation: str) -> BuildInfo:
        current = self.read()
        if current and current.is_live:
            raise BuildInProgress(
                f"index is already running for '{collection}' "
                f"(pid {current.pid}, started {current.started_at})"
            )

        info = BuildInfo(
            pid=os.getpid(),
            collection=collection,
            generation=generation,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(info)), encoding="utf-8")
        return info

    def release(self) -> None:
        with suppress(FileNotFoundError):
            self.path.unlink()

    @contextmanager
    def hold(self, collection: str, generation: str) -> Iterator[BuildInfo]:
        info = self.acquire(collection, generation)
        try:
            yield info
        finally:
            self.release()
