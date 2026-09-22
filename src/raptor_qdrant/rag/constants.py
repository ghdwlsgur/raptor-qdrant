# ========================================= Retriever
DEFAULT_MAX_TOKENS = 4096
# 예산이 병목이 되는 지점. 5 로 두면 4096 토큰 중 36% 만 쓰고 끝난다.
# 12 면 74%, 16 으로 더 올려도 80% 라 거기서부터는 예산이 먼저 찬다
DEFAULT_TOP_K = 12
# 상위 top_k 안에 남겨 둘 요약 노드 자리. 잎이 후보를 쓸어가는 것을 막는다
DEFAULT_SUMMARY_QUOTA = 4
# 레이어를 섞으려면 top_k 보다 넉넉히 받아 와야 고를 것이 생긴다. 실측으로
# 개괄 질문의 첫 요약이 24 등까지 내려간 적이 있다
DEFAULT_CANDIDATE_MULTIPLIER = 4
DEFAULT_COLLECTION_NAME = "default_collection"
DEFAULT_HYBRID_ALPHA = 0.8
DEFAULT_BATCH_SIZE = 64

# ========================================= Tokenizer
DEFAULT_ENCODING = "cl100k_base"

# ========================================= Chunker
CHUNK_MAX_TOKENS = 512
# 이보다 작은 조각은 이웃에 붙인다. 본문 없는 제목 한 줄이 짧은 질의와
# 가까워 상위권을 차지하면서 컨텍스트에는 아무것도 보태지 못한다
CHUNK_MIN_TOKENS = 32
SEMANTIC_CHUNK_BUFFER_SIZE = 1
SEMANTIC_BREAKPOINT_PERCENTILE = 95

# ========================================= Bedrock
BEDROCK_MAX_TOKENS = 2048
BEDROCK_TEMPERATURE = 0.1
BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"
BEDROCK_MAX_POOL_CONNECTIONS = 50
BEDROCK_MAX_RETRIES = 3

# ========================================= Ollama
OLLAMA_MAX_TOKENS = 2048
OLLAMA_TEMPERATURE = 0.1
OLLAMA_MAX_RETRIES = 3
OLLAMA_TIMEOUT_SECONDS = 300

# ========================================= Anthropic
# Opus 5 는 사고 토큰이 이 예산을 함께 쓴다. 요약 한 문단 길이로 잡으면
# 사고에 다 쓰고 본문이 잘린 채 stop_reason=max_tokens 로 끝난다
ANTHROPIC_MAX_TOKENS = 8192
ANTHROPIC_MAX_RETRIES = 3
# 안전 분류기가 거절하면 같은 호출 안에서 대체 모델로 한 번 더 태운다
ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Authorization: Bearer 인증을 여는 플래그. SDK 는 자격증명 공급자 경로에서만
# 이걸 자동으로 붙이므로, 토큰을 직접 넘길 때는 우리가 넣어야 한다
ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20"

# ========================================= OpenAI
OPENAI_MAX_TOKENS = 2048
OPENAI_TEMPERATURE = 0.1
OPENAI_MAX_RETRIES = 3

# ========================================= Tree builder
SUMMARIZATION_MAX_WORKERS = {
    "ollama": 2,
    "bedrock": 10,
    "anthropic": 8,
    "openai": 8,
}
DEFAULT_SUMMARIZATION_MAX_WORKERS = 4

# ========================================= Clustering
CLUSTER_MAX_TOKENS = 3500
CLUSTER_REDUCTION_DIMENSION = 10
CLUSTER_PROBABILITY_THRESHOLD = 0.5
CLUSTER_MAX_RECURSION_DEPTH = 10
CLUSTER_MIN_NODES_TO_SPLIT = 3
UMAP_LOCAL_MAX_NEIGHBORS = 10

# ========================================= Corpus
SOURCE_KEY = "document_name"
SOURCE_SET_KEY = "source_notes"
CONTENT_HASH_KEY = "content_hash"
TREE_GENERATION_KEY = "tree_generation"
# 원문이 사라진 요약. 재구축 전까지 검색에서 뺀다
STALE_KEY = "stale"

# 요약 레이어가 이만큼 낡으면 재구축을 권한다
TREE_DRIFT_WARN_RATIO = 0.2

# ========================================= Embedding
# 잎·요약 노드를 임베딩할 때 한 번에 encode 에 넘기는 텍스트 수
EMBEDDING_BATCH_SIZE = 64

# 레이어 요약 진행률을 이 횟수만큼 나눠 로그로 남긴다 (10 이면 10% 단위)
SUMMARY_PROGRESS_STEPS = 10
