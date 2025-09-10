import logging
import os
from src.rag.retrieval_augmentation import RAGConfig, RetrievalAugmentation
from src.database.qdrant_manager import QdrantManager
from src.core.logger import configure_logging

logger = logging.getLogger(__name__)

def test_layer_retrieval():
    """층별 검색과 전체 검색을 비교 테스트"""
    
    configure_logging()
    
    # Qdrant 연결
    try:
        qdrant_manager = QdrantManager()
        qdrant_manager.connect()
        logger.info("Qdrant connection successful.")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return

    # RAG 설정
    config = RAGConfig(
        collection_name="cinderella_story",
        top_k=8,
        max_tokens_per_chunk=512,
        max_context_tokens=4000,
    )

    rag_pipeline = RetrievalAugmentation(config)
    
    # 테스트 질문
    question = "What did the prince use to find Cinderella?"
    
    print(f"\n🔍 Testing question: '{question}'")
    print("=" * 60)
    
    # 1. 전체 트리 검색 (기본)
    print("\n1️⃣ Full Tree Search (collapse_tree=True):")
    context1, layer_info1 = rag_pipeline.retrieve(question, collapse_tree=True)
    print(f"Retrieved {len(layer_info1)} chunks:")
    for i, info in enumerate(layer_info1[:5]):
        print(f"  Chunk {i+1}: Layer {info['layer_number']}, Score: {info['score']:.3f}")
    
    # 2. 상위 레이어만 검색
    print("\n2️⃣ Top Layer Only Search (start_layer=2):")
    try:
        context2, layer_info2 = rag_pipeline.retrieve(question, collapse_tree=False, start_layer=2)
        print(f"Retrieved {len(layer_info2)} chunks:")
        for i, info in enumerate(layer_info2[:5]):
            print(f"  Chunk {i+1}: Layer {info['layer_number']}, Score: {info['score']:.3f}")
    except Exception as e:
        print(f"Layer 2 search failed: {e}")
    
    # 3. 원본 레이어만 검색
    print("\n3️⃣ Leaf Layer Only Search (start_layer=0):")
    try:
        context3, layer_info3 = rag_pipeline.retrieve(question, collapse_tree=False, start_layer=0)
        print(f"Retrieved {len(layer_info3)} chunks:")
        for i, info in enumerate(layer_info3[:5]):
            print(f"  Chunk {i+1}: Layer {info['layer_number']}, Score: {info['score']:.3f}")
    except Exception as e:
        print(f"Layer 0 search failed: {e}")
    
    # 4. 각 방법으로 답변 생성
    print("\n📝 Answer Comparison:")
    
    print("\n🌳 Full Tree Answer:")
    answer1 = rag_pipeline.answer_question(question, collapse_tree=True)
    print(answer1)
    
    print("\n🔝 Top Layer Answer:")
    try:
        answer2 = rag_pipeline.answer_question(question, collapse_tree=False, start_layer=2)
        print(answer2)
    except:
        print("Top layer answer failed")
    
    print("\n🍃 Leaf Layer Answer:")
    try:
        answer3 = rag_pipeline.answer_question(question, collapse_tree=False, start_layer=0)
        print(answer3)
    except:
        print("Leaf layer answer failed")

if __name__ == "__main__":
    test_layer_retrieval()