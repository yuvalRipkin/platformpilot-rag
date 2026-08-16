import asyncio
import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings

logger = logging.getLogger(__name__)

# Ceiling on the *derived* worker count. Past this, concurrent embeds are
# throughput-bound anyway and each extra thread costs torch working memory;
# queueing is the right answer to a load spike, not more threads.
MAX_DERIVED_WORKERS = 8


class Embedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        torch_threads: int | None = None,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        threads = (
            torch_threads
            if torch_threads is not None
            else settings.embedder_torch_threads
        )
        if threads > 0:
            # The executor below supplies the concurrency, so torch must not
            # also fan each inference out across every core — the two
            # multiply, and N workers x C intra-op threads oversubscribes the
            # CPU N-fold. One thread per job makes max_workers a truthful
            # statement about how much CPU embedding can consume.
            torch.set_num_threads(threads)

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()


def _available_cpus() -> int:
    """CPUs this process may use.

    None of these read a cgroup CPU *quota*: under a Kubernetes CPU limit this
    still reports the node's core count. That is why the derived default is
    clamped and why EMBEDDER_MAX_WORKERS should be set explicitly in a
    deployment to match the pod's CPU limit.
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if process_cpu_count is not None:
        return process_cpu_count() or 1
    try:
        return len(os.sched_getaffinity(0))  # Linux
    except AttributeError:  # macOS, Windows
        return os.cpu_count() or 1


def resolve_max_workers(configured: int | None) -> int:
    if configured is not None:
        if configured < 1:
            raise ValueError("embedder max_workers must be >= 1")
        return configured
    # Leave a core for the event loop. /query holds a 1-2s Anthropic
    # round-trip open per request, and the loop thread has to stay schedulable
    # to drive those sockets while embeds run.
    return max(1, min(_available_cpus() - 1, MAX_DERIVED_WORKERS))


_executor: ThreadPoolExecutor | None = None


def init_embed_executor(max_workers: int | None = None) -> ThreadPoolExecutor:
    """Create (or replace) the process-wide embedding thread pool."""
    global _executor
    shutdown_embed_executor()
    workers = resolve_max_workers(
        max_workers if max_workers is not None else settings.embedder_max_workers
    )
    _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="embed")
    logger.info(
        "Embedding pool ready",
        extra={
            "embed_workers": workers,
            "torch_threads": settings.embedder_torch_threads,
        },
    )
    return _executor


def get_embed_executor() -> ThreadPoolExecutor:
    # No await between the check and the assignment, so concurrent first-time
    # callers on the event loop cannot race here.
    if _executor is None:
        return init_embed_executor()
    return _executor


def shutdown_embed_executor() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


async def encode_async(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Run a blocking encode() on the bounded embedding pool.

    Every async path into the embedder goes through here: inline calls stall
    the event loop for the whole process, and asyncio.to_thread would use the
    default unbounded-ish executor (min(32, cpu_count + 4)), admitting far more
    simultaneous inferences than the CPU can serve.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_embed_executor(), embedder.encode, texts)
