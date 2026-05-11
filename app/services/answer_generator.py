import tiktoken

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.metrics import (
    rag_chunks_retrieved,
    rag_llm_duration_seconds,
    rag_retrieval_duration_seconds,
)
from app.services.llm_client import LLMClient
from app.services.retriever import RetrievedChunk, Retriever

_ENCODING = tiktoken.get_encoding("cl100k_base")

_FALLBACK_ANSWER = (
    "I don't have information about that in the indexed documents."
)

_SYSTEM_PROMPT = """\
You are a documentation assistant for the PlatformPilot project.
Answer questions strictly using the provided context.
Rules:
1. Use only information from the numbered context below.
2. Cite sources by their numbers, e.g. "as described in [1]".
3. If the context does not contain the answer, say "I don't have that information in the indexed documents."
4. Do not invent details, file names, or commands not present in the context.
5. Be concise. Prefer short, direct answers."""


def _format_chunk(idx: int, chunk: RetrievedChunk) -> str:
    return (
        f"[{idx}] Source: {chunk.source} (chunk {chunk.chunk_index})\n"
        f"{chunk.text}"
    )


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODING.decode(tokens[:max_tokens])


class AnswerGenerator:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    async def generate_answer(
        self,
        db: AsyncSession,
        query: str,
    ) -> tuple[str, list[RetrievedChunk]]:
        with rag_retrieval_duration_seconds.time():
            chunks = await self.retriever.retrieve(
                db,
                query,
                k=self.settings.top_k,
                threshold=self.settings.similarity_threshold,
            )
        rag_chunks_retrieved.observe(len(chunks))

        if not chunks:
            return _FALLBACK_ANSWER, []

        formatted = "\n\n".join(
            _format_chunk(i + 1, c) for i, c in enumerate(chunks)
        )
        prefix = f"Question: {query}\n\nContext:\n"
        prefix_tokens = len(_ENCODING.encode(prefix))
        budget = max(0, self.settings.max_context_tokens - prefix_tokens)
        formatted = _truncate_to_tokens(formatted, budget)
        user_message = prefix + formatted

        with rag_llm_duration_seconds.time():
            answer = await self.llm.generate(
                system=_SYSTEM_PROMPT,
                user=user_message,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
            )
        return answer, chunks
