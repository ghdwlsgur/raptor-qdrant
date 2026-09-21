import argparse
import logging
import os
import uuid
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path

from raptor_qdrant.core.config import settings
from raptor_qdrant.core.logger import configure_logging
from raptor_qdrant.database.qdrant_manager import QdrantManager
from raptor_qdrant.rag.engine import EngineConfig, QueryResult, RaptorEngine
from raptor_qdrant.rag.llm import BaseChatbotModel, create_chatbot
from raptor_qdrant.rag.summarizer import LLMSummarizer
from raptor_qdrant.vault import (
    BuildLock,
    DeferredChanges,
    VaultLoader,
    VaultNote,
    changed_note_paths,
    diff_vault,
    watch_vault,
)

logger = logging.getLogger(__name__)

LLAMA_INDEX_CACHE_DIR = "/tmp/llama_index_cache"
REMOTE_PROVIDERS = frozenset({"bedrock"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="raptor-qdrant",
        description="RAPTOR 트리를 Qdrant 에 적재하고 질의한다",
    )
    parser.add_argument(
        "--collection",
        default=settings.COLLECTION_NAME,
        help=f"Qdrant 컬렉션 (기본: {settings.COLLECTION_NAME})",
    )
    parser.add_argument(
        "--llm",
        choices=["ollama", "bedrock"],
        default=None,
        help=f"LLM 공급자 (기본: {settings.LLM_PROVIDER})",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="볼트 전체를 새로 인덱싱한다")
    index.add_argument("--vault", default=settings.VAULT_PATH)
    index.add_argument(
        "--allow-remote-llm",
        action="store_true",
        help="원격 LLM 으로 볼트를 인덱싱하는 것을 허용한다",
    )

    sync = commands.add_parser("sync", help="변경된 노트만 즉시 반영한다")
    sync.add_argument("--vault", default=settings.VAULT_PATH)
    sync.add_argument(
        "--dry-run", action="store_true", help="무엇이 바뀌었는지만 보여준다"
    )
    sync.add_argument(
        "--rebuild-tree",
        action="store_true",
        help="잎만 갱신하지 않고 요약 레이어까지 다시 세운다 (느리다)",
    )
    sync.add_argument("--allow-remote-llm", action="store_true")

    watch = commands.add_parser(
        "watch", help="볼트를 감시하며 저장될 때마다 잎을 갱신한다"
    )
    watch.add_argument("--vault", default=settings.VAULT_PATH)
    watch.add_argument("--debounce", type=float, default=3.0)
    watch.add_argument("--allow-remote-llm", action="store_true")

    ask = commands.add_parser("ask", help="적재된 내용에 질문한다")
    ask.add_argument("question")
    ask.add_argument("--show-chunks", type=int, default=3)

    commands.add_parser("status", help="컬렉션에 무엇이 들어 있는지 보여준다")

    return parser.parse_args()


def ready_chatbot(
    provider: str | None, model: str | None = None
) -> BaseChatbotModel:
    llm = create_chatbot(provider, model)
    health_check = getattr(llm, "health_check", None)
    if health_check:
        health_check()
    logger.info(f"llm ready: {llm.describe}")
    return llm


def summary_model_override(provider: str | None) -> str | None:
    """요약에 답변 모델과 다른 Ollama 모델을 쓰기로 했으면 그 이름."""
    name = (provider or settings.LLM_PROVIDER).strip().lower()
    wanted = settings.OLLAMA_SUMMARY_MODEL.strip()
    if name == "ollama" and wanted and wanted != settings.OLLAMA_MODEL:
        return wanted
    return None


def ready_summarizer(
    provider: str | None, llm: BaseChatbotModel
) -> LLMSummarizer:
    """요약기를 준비한다. 전용 모델이 지정돼 있으면 그것도 health check 한다."""
    override = summary_model_override(provider)
    if override:
        return LLMSummarizer(ready_chatbot(provider, override))
    return LLMSummarizer(llm)


def guard_remote_llm(provider: str | None, allowed: bool) -> bool:
    """볼트 본문을 원격으로 보내기 전에 명시적 동의를 요구한다."""
    name = (provider or settings.LLM_PROVIDER).lower()
    if name in REMOTE_PROVIDERS and not allowed:
        logger.error(
            f"'{name}' is a remote provider and vault notes would be sent to "
            "it. pass --allow-remote-llm if that is intended, or use ollama"
        )
        return False
    return True


def load_vault(path: str) -> tuple[VaultLoader, list[VaultNote]]:
    loader = VaultLoader(path)
    notes = loader.load()
    if not notes:
        raise ValueError(f"no indexable notes under {loader.vault_path}")
    return loader, notes


def as_documents(notes: list[VaultNote]) -> dict[str, str]:
    return {note.path: note.text for note in notes}


def as_hashes(notes: list[VaultNote]) -> dict[str, str]:
    return {note.path: note.content_hash for note in notes}


def print_result(result: QueryResult, chunk_preview: int) -> None:
    print(f"\n질문: {result.question}")
    print(f"\n{result.answer}")

    if result.sources:
        print("\n근거 노트:")
        for source in result.sources:
            print(f"  [[{Path(source).stem}]]  ({source})")

    if chunk_preview and result.chunks:
        print(
            f"\n검색된 청크 {len(result.chunks)}개 중 상위 {chunk_preview}개:"
        )
        for i, info in enumerate(result.chunks[:chunk_preview], start=1):
            print(
                f"  {i}. layer {info['layer_number']} / "
                f"score {info['score']:.3f} / "
                f"{info['chunked_by']} / {info['token_count']} tokens"
            )


def build_lock(collection: str) -> BuildLock:
    return BuildLock.for_collection(collection, settings.STATE_DIR)


def refuse_if_building(lock: BuildLock, collection: str) -> bool:
    """재구축이 돌고 있으면 알리고 True 를 돌려준다."""
    info = lock.read()
    if info and info.is_live:
        logger.error(
            f"index is already running for '{collection}' "
            f"(pid {info.pid}, {format_elapsed(info.elapsed)} elapsed). "
            "wait for it to finish"
        )
        return True
    return False


def format_elapsed(elapsed: timedelta) -> str:
    return str(elapsed).split(".")[0]


def rebuild(
    engine: RaptorEngine, loader: VaultLoader, notes: list[VaultNote]
) -> int:
    """볼트 전체로 트리를 새로 세운다. 락을 잡고, 끝나면 그 사이 변경을 따라잡는다."""
    lock = build_lock(engine.collection_name)
    if refuse_if_building(lock, engine.collection_name):
        return 1

    stale = lock.stale()
    if stale:
        dropped = engine.discard_generation(stale.generation)
        logger.warning(
            f"previous index (pid {stale.pid}) did not finish. dropped "
            f"{dropped} points of generation {stale.generation[:8]}"
        )

    snapshot = as_hashes(notes)
    generation = uuid.uuid4().hex
    with lock.hold(engine.collection_name, generation):
        indexed = engine.add_corpus(
            as_documents(notes), note_hashes=snapshot, generation=generation
        )
        logger.info(f"indexed {len(notes)} notes into {indexed} nodes")
        catch_up(engine, loader, snapshot)
    return 0


def catch_up(
    engine: RaptorEngine, loader: VaultLoader, snapshot: Mapping[str, str]
) -> None:
    """빌드 동안 바뀐 노트를 잎만 갱신해 따라잡는다.

    index 는 시작 시점의 스냅샷으로 트리를 세운다. 몇 시간 도는 사이 사용자가
    고친 노트는 트리에 없고, 세대 정리 때 watch 가 넣어둔 잎도 함께 지워졌다.
    끝난 뒤 스냅샷과 지금 볼트를 비교해 그 차이만 잎으로 넣는다.
    """
    diff = diff_vault(loader.load(), snapshot)
    if not diff.has_changes:
        logger.info("no notes changed while the tree was being built")
        return

    logger.info(
        f"catching up on notes changed during the build: {diff.summary}"
    )
    apply_leaf_changes(engine, diff.added + diff.changed, diff.removed)


def run_index(engine: RaptorEngine, args: argparse.Namespace) -> int:
    loader, notes = load_vault(args.vault)
    return rebuild(engine, loader, notes)


def run_sync(engine: RaptorEngine, args: argparse.Namespace) -> int:
    loader, notes = load_vault(args.vault)
    if refuse_if_building(
        build_lock(engine.collection_name), engine.collection_name
    ):
        return 1

    diff = diff_vault(notes, engine.indexed_content_hashes())
    logger.info(f"vault diff: {diff.summary}")

    if not diff.has_changes and not args.rebuild_tree:
        report_drift(engine)
        return 0

    for note in diff.added:
        print(f"  + {note.path}")
    for note in diff.changed:
        print(f"  ~ {note.path}")
    for path in diff.removed:
        print(f"  - {path}")

    if args.dry_run:
        return 0

    if args.rebuild_tree:
        return rebuild(engine, loader, notes)

    apply_leaf_changes(engine, diff.added + diff.changed, diff.removed)
    report_drift(engine)
    return 0


def apply_leaf_changes(
    engine: RaptorEngine,
    upserted: tuple[VaultNote, ...],
    removed: tuple[str, ...],
) -> None:
    if removed:
        deleted = engine.remove_notes(removed)
        logger.info(f"removed {len(removed)} notes ({deleted} points)")

    if upserted:
        nodes = engine.upsert_notes(
            as_documents(list(upserted)), as_hashes(list(upserted))
        )
        logger.info(f"updated {len(upserted)} notes ({nodes} leaf chunks)")


def report_drift(engine: RaptorEngine) -> None:
    health = engine.drift()
    logger.info(f"index: {health.summary}")
    if health.needs_rebuild:
        logger.warning(
            "summary layers are stale. run `raptor-qdrant sync --rebuild-tree`"
        )


def apply_paths(
    engine: RaptorEngine, loader: VaultLoader, paths: set[Path]
) -> None:
    relative = changed_note_paths(paths, loader.vault_path)
    if not relative:
        return

    alive = {
        note.path: note
        for note in (
            loader.load_note(loader.vault_path / name)
            for name in relative
            if (loader.vault_path / name).is_file()
        )
        if not note.is_empty
    }
    gone = tuple(sorted(relative - alive.keys()))

    apply_leaf_changes(engine, tuple(alive.values()), gone)


def run_watch(engine: RaptorEngine, args: argparse.Namespace) -> int:
    loader, _ = load_vault(args.vault)
    lock = build_lock(engine.collection_name)
    held = DeferredChanges()

    def apply(paths: set[Path]) -> None:
        # 재구축 중에 잎을 넣으면 서로 느려지고 세대 정리 때 지워진다.
        # 모아뒀다가 끝난 뒤 tick 에서 적용한다.
        if lock.is_held():
            count = held.add(paths)
            logger.info(
                f"index is running, holding {count} changed note(s) until it "
                "finishes"
            )
            return
        apply_paths(engine, loader, paths)

    def tick() -> None:
        if len(held) and not lock.is_held():
            paths = held.drain()
            logger.info(f"index finished, applying {len(paths)} held note(s)")
            apply_paths(engine, loader, paths)

    watch_vault(
        loader.vault_path, apply, debounce_seconds=args.debounce, tick=tick
    )
    return 0


def run_ask(engine: RaptorEngine, args: argparse.Namespace) -> int:
    print_result(engine.query(args.question), args.show_chunks)
    return 0


def run_status(engine: RaptorEngine, _: argparse.Namespace) -> int:
    documents = engine.list_documents()
    health = engine.drift()
    print(f"컬렉션: {engine.collection_name}")
    print(f"문서: {len(documents)}개")
    print(
        f"잎 노드: {health.leaf_nodes}개 / 요약 노드: {health.summary_nodes}개"
    )
    print(
        f"현재 트리 밖의 잎: {health.leaves_outside_tree}개 ({health.drift:.0%})"
    )
    if health.needs_rebuild:
        print("  요약 레이어가 낡았다. sync --rebuild-tree 를 권한다")
    if health.has_mixed_generations:
        print(f"  트리 세대가 {health.generations}개 섞여 있다")

    info = build_lock(engine.collection_name).read()
    if info and info.is_live:
        print(
            f"빌드 중: pid {info.pid}, {format_elapsed(info.elapsed)} 경과 "
            f"(세대 {info.generation[:8]})"
        )
    elif info:
        print(
            f"  끝나지 않은 빌드 흔적: pid {info.pid}, 세대 "
            f"{info.generation[:8]}. 다음 index 가 정리한다"
        )
    for name in documents[:20]:
        print(f"  {name}")
    if len(documents) > 20:
        print(f"  ... 외 {len(documents) - 20}개")
    return 0


COMMANDS = {
    "index": run_index,
    "sync": run_sync,
    "watch": run_watch,
    "ask": run_ask,
    "status": run_status,
}

NEEDS_VAULT_GUARD = frozenset({"index", "sync", "watch"})


def main() -> int:
    args = parse_args()

    os.environ["LLAMA_INDEX_CACHE_DIR"] = LLAMA_INDEX_CACHE_DIR
    configure_logging()

    if args.command in NEEDS_VAULT_GUARD and not guard_remote_llm(
        args.llm, args.allow_remote_llm
    ):
        return 1

    try:
        QdrantManager().connect()
    except Exception as e:
        logger.error(f"cannot reach Qdrant: {e}")
        return 1

    try:
        llm = ready_chatbot(args.llm)
        summarizer = ready_summarizer(args.llm, llm)
    except Exception as e:
        logger.error(f"LLM is not usable: {e}")
        return 1

    engine = RaptorEngine(
        EngineConfig(
            collection_name=args.collection,
            llm=llm,
            summarization_model=summarizer,
        )
    )

    try:
        return COMMANDS[args.command](engine, args)
    except Exception as e:
        logger.error(f"{args.command} failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
