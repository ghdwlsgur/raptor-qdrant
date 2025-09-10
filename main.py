import logging
import os

# RAG 파이프라인의 핵심 클래스들을 임포트합니다.
from src.rag.retrieval_augmentation import RAGConfig, RetrievalAugmentation
from src.database.qdrant_manager import QdrantManager
from src.core.logger import configure_logging

# 로거 설정
logger = logging.getLogger(__name__)


def main():
    # 1. 로깅 시스템 설정
    configure_logging()

    # 2. Qdrant 데이터베이스 연결 (싱글톤 패턴)
    try:
        qdrant_manager = QdrantManager()
        qdrant_manager.connect()  # settings.py 또는 환경변수에 설정된 값으로 연결
        logger.info("Qdrant connection successful.")
    except Exception as e:
        logger.error(
            f"Failed to connect to Qdrant. Please ensure Qdrant is running. Error: {e}"
        )
        return

    # 3. RAG 파이프라인 설정
    config = RAGConfig(
        collection_name="cinderella_story",  # 컬렉션 이름을 문서 내용에 맞게 변경
        top_k=10,  # 검색할 청크 개수 (증가)
        max_tokens_per_chunk=512,  # 텍스트를 나눌 청크의 최대 토큰 크기 (증가)
        max_context_tokens=4000,  # 컨텍스트 토큰 수 (증가)
    )

    # 4. RAG 파이프라인 객체 생성
    rag_pipeline = RetrievalAugmentation(config)
    logger.info("RAG pipeline initialized.")

    # 5. 파일에서 문서 읽어오기
    file_path = "data/sample.txt"
    if not os.path.exists(file_path):
        logger.error(
            f"File not found: {file_path}. Please create this file with the Cinderella story."
        )
        return

    logger.info(f"Reading document from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    # 6. 문서 인덱싱 실행
    # 이 과정에서 RAPTOR 트리가 생성되고, 모든 노드가 Qdrant에 저장됩니다.
    logger.info("Starting to add and index the document...")
    rag_pipeline.add_documents(document_text)
    logger.info("Document indexing finished.")

    # 7. 문서 내용에 대한 질문 및 답변 생성
    questions = [
        "Who is Cinderella?",
        "What did Cinderella's step-sisters ask her to do before the festival?",
        "What did the prince use to find Cinderella?",
        "How were the step-sisters punished in the end?",
    ]

    for question in questions:
        logger.info(f"Answering question: '{question}'")

        # 검색 결과를 확인하기 위해 retrieve 메서드를 직접 호출
        context, layer_info = rag_pipeline.retrieve(question)
        
        # 검색된 컨텍스트 정보 출력
        print(f"\n🔍 Retrieved Context (from {len(layer_info)} chunks):")
        for i, info in enumerate(layer_info[:3]):  # 상위 3개만 출력
            print(f"  Chunk {i+1}: Layer {info['layer_number']}, Score: {info['score']:.3f}")
        
        # QA 모델로 답변 생성
        answer = rag_pipeline.answer_question(question)

        print("\n" + "=" * 50)
        print(f"❓ Question: {question}")
        print(f"✅ Answer: {answer}")
        print("=" * 50)


if __name__ == "__main__":
    main()
