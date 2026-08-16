import asyncio
import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

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


CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass(frozen=True)
class CpuBudget:
    """How much CPU this process may actually use, and where we learned it."""

    cpus: int
    source: str  # "cgroup" | "affinity"
    affinity_cpus: int
    quota_cpus: float | None


def affinity_cpus() -> int:
    """CPUs this process is *allowed to run on*.

    This is the scheduler's affinity mask. It does NOT observe a cgroup CPU
    quota: a container limited to one core still has every core in its mask
    and is throttled at CFS period boundaries instead.
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if process_cpu_count is not None:
        return process_cpu_count() or 1
    try:
        return len(os.sched_getaffinity(0))  # Linux
    except AttributeError:  # macOS, Windows
        return os.cpu_count() or 1


def cgroup_quota_cpus(root: Path = CGROUP_ROOT) -> float | None:
    """Cores permitted by the cgroup CFS quota, or None if unlimited/absent.

    cgroup v2 `cpu.max` holds "<quota> <period>" in microseconds, where quota
    is the literal "max" when unthrottled: `--cpus=1.5` reads "150000 100000".
    cgroup v1 splits the same numbers across cpu.cfs_quota_us (-1 when
    unlimited) and cpu.cfs_period_us.
    """
    try:
        fields = (root / "cpu.max").read_text().split()
        if fields[0] != "max":
            quota, period = float(fields[0]), float(fields[1])
            if quota > 0 and period > 0:
                return quota / period
        return None
    except (OSError, ValueError, IndexError):
        pass

    try:
        quota = float((root / "cpu" / "cpu.cfs_quota_us").read_text().strip())
        period = float((root / "cpu" / "cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def cpu_budget(root: Path = CGROUP_ROOT) -> CpuBudget:
    affinity = affinity_cpus()
    quota = cgroup_quota_cpus(root)
    if quota is not None and quota < affinity:
        # Fractional quotas floor to whole workers; a 1.5-core pod gets one
        # embed at a time rather than two that throttle each other.
        return CpuBudget(max(1, int(quota)), "cgroup", affinity, quota)
    return CpuBudget(affinity, "affinity", affinity, quota)


def resolve_max_workers(configured: int | None, budget: CpuBudget | None = None) -> int:
    if configured is not None:
        if configured < 1:
            raise ValueError("embedder max_workers must be >= 1")
        return configured
    # Leave a core for the event loop. /query holds a 1-2s Anthropic
    # round-trip open per request, and the loop thread has to stay schedulable
    # to drive those sockets while embeds run.
    cpus = (budget or cpu_budget()).cpus
    return max(1, min(cpus - 1, MAX_DERIVED_WORKERS))


_executor: ThreadPoolExecutor | None = None


def init_embed_executor(max_workers: int | None = None) -> ThreadPoolExecutor:
    """Create (or replace) the process-wide embedding thread pool."""
    global _executor
    shutdown_embed_executor()
    configured = (
        max_workers if max_workers is not None else settings.embedder_max_workers
    )
    budget = cpu_budget()
    workers = resolve_max_workers(configured, budget)
    _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="embed")
    logger.info(
        "Embedding pool ready",
        extra={
            "embed_workers": workers,
            # Which input decided the bound. "affinity" inside a container
            # means no CPU limit was set on the pod — worth noticing.
            "workers_source": "config" if configured is not None else budget.source,
            "affinity_cpus": budget.affinity_cpus,
            "cgroup_quota_cpus": budget.quota_cpus,
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
