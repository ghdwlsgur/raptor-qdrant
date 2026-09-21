# ========================================= Retriever
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_K = 5
DEFAULT_COLLECTION_NAME = "default_collection"
DEFAULT_HYBRID_ALPHA = 0.8
DEFAULT_BATCH_SIZE = 64

# ========================================= Tokenizer
DEFAULT_ENCODING = "cl100k_base"

# ========================================= Chunker
CHUNK_MAX_TOKENS = 512
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

# ========================================= Tree builder
SUMMARIZATION_MAX_WORKERS = {"ollama": 2, "bedrock": 10}
DEFAULT_SUMMARIZATION_MAX_WORKERS = 4

# ========================================= Clustering
CLUSTER_MAX_TOKENS = 3500
CLUSTER_REDUCTION_DIMENSION = 10
CLUSTER_PROBABILITY_THRESHOLD = 0.5
CLUSTER_MAX_RECURSION_DEPTH = 10
CLUSTER_MIN_NODES_TO_SPLIT = 3
UMAP_LOCAL_MAX_NEIGHBORS = 10
