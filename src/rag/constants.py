# ========================================= Retriever
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_K = 5
DEFAULT_COLLECTION_NAME = "default_collection"
DEFAULT_HYBRID_ALPHA = 0.8
DEFAULT_BATCH_SIZE = 64
MAX_SCROLL_LIMIT = 10000

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
