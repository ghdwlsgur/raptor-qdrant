import argparse
import logging
import os
from pathlib import Path

from raptor_qdrant.core.config import settings
from raptor_qdrant.core.logger import configure_logging
from raptor_qdrant.database.qdrant_manager import QdrantManager
from raptor_qdrant.rag.engine import EngineConfig, QueryResult, RaptorEngine
from raptor_qdrant.rag.llm import BaseChatbotModel, create_chatbot
from raptor_qdrant.vault import (
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


def ready_chatbot(provider: str | None) -> BaseChatbotModel:
    llm = create_chatbot(provider)
    health_check = getattr(llm, "health_check", None)
    if health_check:
        health_check()
    logger.info(f"llm ready: {llm.describe}")
    return llm


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


def run_index(engine: RaptorEngine, args: argparse.Namespace) -> int:
    _, notes = load_vault(args.vault)
    indexed = engine.add_corpus(
        as_documents(notes),
        recreate_collection=True,
        note_hashes=as_hashes(notes),
    )
    logger.info(f"indexed {len(notes)} notes into {indexed} nodes")
    return 0


def run_sync(engine: RaptorEngine, args: argparse.Namespace) -> int:
    _, notes = load_vault(args.vault)
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
        indexed = engine.add_corpus(
            as_documents(notes),
            recreate_collection=True,
            note_hashes=as_hashes(notes),
        )
        logger.info(
            f"rebuilt the tree: {len(notes)} notes into {indexed} nodes"
        )
        return 0

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


def run_watch(engine: RaptorEngine, args: argparse.Namespace) -> int:
    loader, _ = load_vault(args.vault)

    def apply(paths: set[Path]) -> None:
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

    watch_vault(loader.vault_path, apply, debounce_seconds=args.debounce)
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
    except Exception as e:
        logger.error(f"LLM is not usable: {e}")
        return 1

    engine = RaptorEngine(
        EngineConfig(collection_name=args.collection, llm=llm)
    )

    try:
        return COMMANDS[args.command](engine, args)
    except Exception as e:
        logger.error(f"{args.command} failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
