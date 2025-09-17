import logging
import os

from src.rag.main import EngineConfig, RaptorEngine
from src.database.qdrant_manager import QdrantManager
from src.core.logger import configure_logging


logger = logging.getLogger(__name__)


def main():
    # LlamaIndex 설정
    os.environ["LLAMA_INDEX_CACHE_DIR"] = "/tmp/llama_index_cache"

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
        collection_name="sample",  # 컬렉션 이름을 문서 내용에 맞게 변경
    )

    # 4. RAG 파이프라인 객체 생성
    engine = RaptorEngine(config)
    logger.info("rag pipeline initialized")

    # 5. 파일에서 문서 읽어오기
    file_path = "data/sample.md"
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
    engine.add_document(document_text, document_name="Sample Test")
    logger.info("document indexing finished")

    # 7. 문서 내용에 대한 질문 및 답변 생성
    # questions = [
    #     "Who is Cinderella?",
    #     "What did Cinderella's step-sisters ask her to do before the festival?",
    #     "What did the prince use to find Cinderella?",
    #     "How were the step-sisters punished in the end?",
    # ]
    # questions = [
    #     "신데렐라는 누구인가요?",
    #     "신데렐라의 의붓언니들은 축제 전에 신데렐라에게 무엇을 하라고 시켰나요?",
    #     "왕자는 신데렐라를 찾기 위해 무엇을 사용했나요?",
    #     "마지막에 의붓언니들은 어떻게 벌을 받았나요?",
    # ]
    questions = [
        "텍스트 마이닝의 4단계 과정을 순서대로 알려주세요",
        "비정형(Unstructured) 데이터가 정형(Structured) 데이터로 변환되는 예시 표에서, 이름이 'Linh'인 사람의 나이는 몇 살인가요?",
        "천연 화장품'의 연관어 분석 표에서, '효능/효과'의 세부 키워드 중 가장 수치가 높은 것은 무엇이며 그 값은 얼마인가요?",
        "텍스트 분석의 '과업(Task)'으로 언급되지 않은 것을 고르세요: 1) 문서 요약, 2) 감성 분석, 3) 이미지 인식, 4) 기계 번역",
        "텍스트 데이터 수집에서 '인간'은 어떤 역할을 담당하며, 이는 온도계나 위치 센서와 같은 일반적인 센서와 어떻게 다른가요?",
    ]

    for question in questions:
        # answer 메서드가 내부적으로 retrieve를 호출하므로 중복 제거
        answer = engine.answer(question)
        # 디버그용으로 검색 정보를 별도로 가져옴 (실제로는 중복이지만 정보 표시용)
        _, layer_info = engine.retrieve(question)
        print(f"\n🔍 Retrieved Context (from {len(layer_info)} chunks):")
        for i, info in enumerate(layer_info[:3]):  # 상위 3개만 출력
            print(
                f" Chunk {i+1}: Layer {info['layer_number']}, Score: {info['score']:.3f}, Chunked by: {info['chunked_by']}, Token: {info['token_count']}"
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
