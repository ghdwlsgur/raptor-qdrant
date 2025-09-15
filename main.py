import logging
import os

from src.rag.main import EngineConfig, RaptorEngine
from src.database.qdrant_manager import QdrantManager
from src.core.logger import configure_logging

logger = logging.getLogger(__name__)


def main():
    # 1. 로깅 시스템 설정
    configure_logging()

    # 2. Qdrant 데이터베이스 연결 (싱글톤 패턴)
    try:
        qdrant_manager = QdrantManager()
        qdrant_manager.connect()  # settings.py 또는 환경변수에 설정된 값으로 연결
        logger.info("qdrant connection successful")
    except Exception as e:
        logger.error(
            f"Failed to connect to Qdrant. Please ensure Qdrant is running. Error: {e}"
        )
        return

    # 3. RAG 파이프라인 설정
    config = EngineConfig(
        collection_name="cinderella_story",  # 컬렉션 이름을 문서 내용에 맞게 변경
        top_k=10,  # 검색할 청크 개수 (증가)
        max_tokens_per_chunk=512,  # 텍스트를 나눌 청크의 최대 토큰 크기 (증가)
        max_context_tokens=4000,  # 컨텍스트 토큰 수 (증가)
    )

    # 4. RAG 파이프라인 객체 생성
    engine = RaptorEngine(config)
    logger.info("rag pipeline initialized")


    # 5. 파일에서 문서 읽어오기
    file_path = "data/sample_ko.txt"
    if not os.path.exists(file_path):
        logger.error(
            f"File not found: {file_path}. Please create this file with the Cinderella story."
        )
        return

    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    # 6. 문서 인덱싱 실행
    # 이 과정에서 RAPTOR 트리가 생성되고, 모든 노드가 Qdrant에 저장됩니다.
    logger.info("starting to add and index the document...")
    engine.add_document(document_text, document_name="Cinderella Story")
    logger.info("document indexing finished")

    # 7. 문서 내용에 대한 질문 및 답변 생성
    # questions = [
    #     "Who is Cinderella?",
    #     "What did Cinderella's step-sisters ask her to do before the festival?",
    #     "What did the prince use to find Cinderella?",
    #     "How were the step-sisters punished in the end?",
    # ]
    questions = [
        "신데렐라는 누구인가요?",
        "신데렐라의 의붓언니들은 축제 전에 신데렐라에게 무엇을 하라고 시켰나요?",
        "왕자는 신데렐라를 찾기 위해 무엇을 사용했나요?",
        "마지막에 의붓언니들은 어떻게 벌을 받았나요?",
    ]

    for question in questions:
        answer = engine.answer(question)
        _, layer_info = engine.retrieve(question)
        print(f"\n🔍 Retrieved Context (from {len(layer_info)} chunks):")
        for i, info in enumerate(layer_info[:3]):  # 상위 3개만 출력
            print(
                f"  Chunk {i+1}: Layer {info['layer_number']}, Score: {info['score']:.3f}"
            )
        print("\n" + "=" * 50)
        print(f"❓ Question: {question}")
        print(f"✅ Answer: {answer}")
        print("=" * 50)

    # points = engine.get_document_points("Cinderella Story")
    # print(
    #     f"Retrieved {len(points)} points for document_name: 'Cinderella Story'"
    # )

    # for point in points:
    #     print(f"  Point ID: {point['id']}, Payload: {point['payload']}")

    # names = engine.list_documents()

    # print("Documents in the collection:")
    # for name in names:
    #     print(f" - {name}")

    # collections = engine.list_collections()
    # print("Collections in the database:")
    # for collection in collections:
    #     print(f" - {collection}")


if __name__ == "__main__":
    main()
