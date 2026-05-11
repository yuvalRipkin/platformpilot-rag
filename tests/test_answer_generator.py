from dataclasses import dataclass
from uuid import uuid4

import tiktoken

from app.services.answer_generator import AnswerGenerator
from app.services.llm_client import LLMClient
from app.services.retriever import RetrievedChunk

_ENCODING = tiktoken.get_encoding("cl100k_base")


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.last_call: tuple[str, int, float] | None = None

    async def retrieve(self, db, query, k, threshold):
        self.last_call = (query, k, threshold)
        return list(self._chunks)


class FakeLLM(LLMClient):
    def __init__(self, response: str = "generated answer") -> None:
        self.response = response
        self.system: str | None = None
        self.user: str | None = None
        self.max_tokens: int | None = None
        self.temperature: float | None = None

    async def generate(self, system, user, max_tokens, temperature):
        self.system = system
        self.user = user
        self.max_tokens = max_tokens
        self.temperature = temperature
        return self.response


@dataclass
class StubSettings:
    top_k: int = 4
    similarity_threshold: float = 0.5
    max_context_tokens: int = 8000
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0


def _chunk(idx: int, source: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        source=source,
        text=text,
        chunk_index=idx,
        similarity=0.9 - idx * 0.01,
    )


async def test_no_chunks_returns_fallback():
    retriever = FakeRetriever([])
    llm = FakeLLM()
    gen = AnswerGenerator(retriever, llm, StubSettings())

    answer, chunks = await gen.generate_answer(db=None, query="anything")

    assert chunks == []
    assert "indexed documents" in answer.lower()
    assert llm.user is None, "LLM must not be called when no chunks retrieved"


async def test_with_chunks_calls_llm_with_numbered_context():
    chunks = [
        _chunk(0, "alpha.md", "alpha content goes here"),
        _chunk(1, "beta.md", "beta content goes here"),
        _chunk(2, "gamma.md", "gamma content goes here"),
    ]
    retriever = FakeRetriever(chunks)
    llm = FakeLLM(response="my generated answer")
    gen = AnswerGenerator(retriever, llm, StubSettings())

    answer, returned = await gen.generate_answer(
        db=None, query="what is alpha?"
    )

    assert answer == "my generated answer"
    assert returned == chunks

    # System prompt enumerates the five rules.
    assert llm.system is not None
    sys = llm.system.lower()
    for snippet in (
        "documentation assistant",
        "only information",
        "cite sources",
        "indexed documents",
        "do not invent",
        "concise",
    ):
        assert snippet in sys, f"system prompt missing: {snippet!r}"

    # User message has numbered citations and chunk texts.
    assert llm.user is not None
    assert "[1]" in llm.user
    assert "[2]" in llm.user
    assert "[3]" in llm.user
    assert "alpha content goes here" in llm.user
    assert "beta content goes here" in llm.user
    assert "gamma content goes here" in llm.user
    assert "what is alpha?" in llm.user

    # Settings flowed through to the LLM call unchanged.
    assert llm.max_tokens == 1024
    assert llm.temperature == 0.0


async def test_context_truncation():
    big_text = ("repeated phrase " * 500).strip()
    chunks = [_chunk(i, "big.md", big_text) for i in range(5)]
    retriever = FakeRetriever(chunks)
    llm = FakeLLM()
    settings = StubSettings(max_context_tokens=1000)
    gen = AnswerGenerator(retriever, llm, settings)

    await gen.generate_answer(db=None, query="q")

    assert llm.user is not None
    user_tokens = len(_ENCODING.encode(llm.user))
    assert user_tokens <= settings.max_context_tokens, (
        f"user message {user_tokens} tokens exceeds cap "
        f"{settings.max_context_tokens}"
    )
