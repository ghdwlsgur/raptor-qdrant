import argparse
import logging
import os
from pathlib import Path

from raptor_qdrant.core.config import settings
from raptor_qdrant.core.logger import configure_logging
from raptor_qdrant.database.qdrant_manager import QdrantManager
from raptor_qdrant.rag.engine import EngineConfig, QueryResult, RaptorEngine
from raptor_qdrant.rag.llm import create_chatbot

logger = logging.getLogger(__name__)

LLAMA_INDEX_CACHE_DIR = "/tmp/llama_index_cache"
DEFAULT_DOCUMENT = "data/sample_ko.txt"
DEFAULT_COLLECTION = "sample"

DEFAULT_QUESTIONS = [
    "신데렐라는 누구인가요?",
    "신데렐라의 의붓언니들은 축제 전에 신데렐라에게 무엇을 하라고 시켰나요?",
    "왕자는 신데렐라를 찾기 위해 무엇을 사용했나요?",
    "마지막에 의붓언니들은 어떻게 벌을 받았나요?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAPTOR 트리를 만들어 Qdrant 에 적재하고 질의하는 데모"
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_DOCUMENT,
        help=f"인덱싱할 문서 경로 (기본: {DEFAULT_DOCUMENT})",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant 컬렉션 이름 (기본: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--document-name",
        default=None,
        help="문서 구분 이름 (기본: 파일 이름)",
    )
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="질문. 여러 번 지정할 수 있다 (기본: 내장 예시 질문)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="인덱싱을 건너뛰고 이미 적재된 컬렉션에 질의만 한다",
    )
    parser.add_argument(
        "--llm",
        choices=["ollama", "bedrock"],
        default=None,
        help=f"요약·답변에 쓸 LLM 공급자 (기본: {settings.LLM_PROVIDER})",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="적재 전에 컬렉션을 통째로 지운다 (같은 컬렉션의 다른 문서도 함께 사라진다)",
    )
    return parser.parse_args()


def ready_chatbot(provider: str | None):
    llm = create_chatbot(provider)
    if hasattr(llm, "health_check"):
        llm.health_check()
    logger.info(f"llm ready: {llm.describe}")
    return llm


def print_result(result: QueryResult, chunk_preview: int = 3) -> None:
    print(f"\n🔍 Retrieved Context (from {len(result.chunks)} chunks):")
    for i, info in enumerate(result.chunks[:chunk_preview], start=1):
        print(
            f" Chunk {i}: Layer {info['layer_number']}, "
            f"Score: {info['score']:.3f}, "
            f"Chunked by: {info['chunked_by']}, "
            f"Token: {info['token_count']}"
        )
    print("\n" + "=" * 50)
    print(f"❓ Question: {result.question}")
    print(f"✅ Answer: {result.answer}")
    print("=" * 50)


def index_document(engine: RaptorEngine, args: argparse.Namespace) -> bool:
    """문서를 읽어 인덱싱한다. 성공하면 True."""
    path = Path(args.file)
    if not path.exists():
        logger.error(f"file not found: {path}")
        return False

    document_name = args.document_name or path.stem
    text = path.read_text(encoding="utf-8")

    if not args.recreate and document_name in engine.list_documents():
        logger.info(
            f"document '{document_name}' is already indexed, replacing it"
        )
        indexed = engine.update_document(text, document_name=document_name)
    else:
        logger.info(f"indexing '{path}' as '{document_name}'...")
        indexed = engine.add_document(
            text,
            document_name=document_name,
            recreate_collection=args.recreate,
        )

    logger.info(f"document indexing finished: {indexed} nodes")
    return True


def main() -> int:
    args = parse_args()

    os.environ["LLAMA_INDEX_CACHE_DIR"] = LLAMA_INDEX_CACHE_DIR
    configure_logging()

    try:
        QdrantManager().connect()
        logger.info("qdrant connection successful")
    except Exception as e:
        logger.error(
            f"Failed to connect to Qdrant. Please ensure Qdrant is running. Error: {e}"
        )
        return 1

    try:
        llm = ready_chatbot(args.llm)
    except Exception as e:
        logger.error(f"LLM is not usable: {e}")
        return 1

    engine = RaptorEngine(
        EngineConfig(collection_name=args.collection, llm=llm)
    )
    logger.info("rag pipeline initialized")

    if not args.skip_index and not index_document(engine, args):
        return 1

    for question in args.questions or DEFAULT_QUESTIONS:
        print_result(engine.query(question))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
