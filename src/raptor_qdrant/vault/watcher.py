import logging
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 3.0
# 재구축이 끝났는지 이 주기로 확인한다
TICK_SECONDS = 15.0


class _DebouncedMarkdownHandler(FileSystemEventHandler):
    """옵시디언은 타이핑 중에도 자주 저장한다. 파일별로 조용해질 때까지 기다린다."""

    def __init__(
        self,
        on_settled: Callable[[set[Path]], None],
        debounce_seconds: float,
    ):
        self._on_settled = on_settled
        self._debounce_seconds = debounce_seconds
        self._lock = threading.Lock()
        self._pending: set[Path] = set()
        self._timer: threading.Timer | None = None

    def on_any_event(self, event: FileSystemEvent) -> None:
        for raw in (event.src_path, getattr(event, "dest_path", None)):
            path = _markdown_path(raw)
            if path:
                self._schedule(path)

    def _schedule(self, path: Path) -> None:
        with self._lock:
            self._pending.add(path)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_seconds, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            settled, self._pending = self._pending, set()
            self._timer = None

        if not settled:
            return

        try:
            self._on_settled(settled)
        except Exception as e:
            logger.error(f"failed to apply vault changes: {e}")

    def flush(self) -> None:
        with self._lock:
            timer, self._timer = self._timer, None
        if timer:
            timer.cancel()
        self._fire()


def _markdown_path(raw: object) -> Path | None:
    if not isinstance(raw, str | bytes):
        return None
    text = raw.decode() if isinstance(raw, bytes) else raw
    path = Path(text)
    if path.suffix != ".md":
        return None
    if {".obsidian", ".trash", ".git"}.intersection(path.parts):
        return None
    return path


class DeferredChanges:
    """재구축이 도는 동안 들어온 변경을 모아둔다. 끝나면 한 번에 적용한다.

    watch 콜백은 워처 스레드에서, 비우는 쪽은 메인 루프에서 부르므로 잠근다.
    """

    def __init__(self) -> None:
        self._paths: set[Path] = set()
        self._lock = threading.Lock()

    def add(self, paths: Iterable[Path]) -> int:
        with self._lock:
            self._paths.update(paths)
            return len(self._paths)

    def drain(self) -> set[Path]:
        with self._lock:
            paths, self._paths = self._paths, set()
        return paths

    def __len__(self) -> int:
        with self._lock:
            return len(self._paths)


def watch_vault(
    vault_path: Path,
    on_settled: Callable[[set[Path]], None],
    debounce_seconds: float = DEBOUNCE_SECONDS,
    stop: threading.Event | None = None,
    tick: Callable[[], None] | None = None,
    tick_seconds: float = TICK_SECONDS,
) -> None:
    """볼트를 감시하며 조용해진 파일 묶음을 콜백으로 넘긴다.

    tick 을 주면 감시 중 주기적으로 부른다. 재구축이 끝났는지 확인하고
    미뤄둔 변경을 적용하는 용도다.
    """
    handler = _DebouncedMarkdownHandler(on_settled, debounce_seconds)
    observer = Observer()
    observer.schedule(handler, str(vault_path), recursive=True)
    observer.start()
    logger.info(f"watching {vault_path} (debounce {debounce_seconds}s)")

    waiter = stop or threading.Event()
    try:
        if tick is None:
            waiter.wait()
        else:
            while not waiter.wait(tick_seconds):
                try:
                    tick()
                except Exception as e:
                    logger.error(f"periodic check failed: {e}")
    except KeyboardInterrupt:
        logger.info("stopping watcher")
    finally:
        observer.stop()
        observer.join(timeout=5)
        handler.flush()


def changed_note_paths(paths: Iterable[Path], vault_root: Path) -> set[str]:
    relative = set()
    for path in paths:
        try:
            relative.add(path.resolve().relative_to(vault_root).as_posix())
        except ValueError:
            continue
    return relative
