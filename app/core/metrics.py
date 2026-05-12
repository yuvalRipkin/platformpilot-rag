from prometheus_client import Counter, Histogram

rag_queries_total = Counter(
    "rag_queries_total",
    "Total /query and /search requests",
    ["endpoint", "status"],
)

rag_retrieval_duration_seconds = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval latency",
)

rag_llm_duration_seconds = Histogram(
    "rag_llm_duration_seconds",
    "LLM generation latency",
)

rag_chunks_retrieved = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks returned from retrieval",
    buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)
