"""Measurement harness for the embedder concurrency matrix.

Fires N concurrent POST /search requests through the real endpoint and the
real Retriever, with a fake Embedder standing in for the model, and measures
wall time, CPU utilisation and event-loop responsiveness.

Four dispatch configurations, which are the shipped states of this repo plus
one diagnostic:

    blocking          embedder.encode() called inline   torch: all cores
    unbounded         asyncio.to_thread (default pool)  torch: all cores
    unbounded+pinned  asyncio.to_thread (default pool)  torch: 1 thread
    bounded           dedicated bounded pool            torch: 1 thread

"blocking" and "unbounded" bundle two changes each (dispatch *and* torch
threading) because that is how they shipped. "unbounded+pinned" exists only to
separate the two effects: comparing it against "unbounded" isolates the value
of pinning torch, and against "bounded" isolates the value of the pool.

Two fakes, because they measure different things:

    SleepEmbedder  time.sleep() — consumes no CPU, so threads never contend.
                   Measures queueing and event-loop stall ONLY. Under it,
                   oversubscription is invisible and "unbounded" looks best.
    CpuEmbedder    torch.mm — real CPU work that releases the GIL and honours
                   torch.set_num_threads. This is the fake that tests the
                   oversubscription hypothesis.

Run inside a CPU-limited container; host numbers do not transfer. See
`make bench-concurrency`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_retriever
from app.db.session import get_db
from app.services import retriever as retriever_module
from app.services.embedder import (
    Embedder,
    cpu_budget,
    encode_async,
    init_embed_executor,
    pin_torch_threads,
    resolve_max_workers,
    shutdown_embed_executor,
)
from app.services.retriever import Retriever

SLEEP_SECONDS = 0.05
TARGET_CPU_SECONDS = 0.05
HEARTBEAT_INTERVAL = 0.005
CALIBRATION_DIMS = (256, 384, 512, 640, 768, 896, 1024, 1280, 1536)
# Keep a single matmul small enough that its working set stays cache-resident,
# then repeat it to reach the target. One matmul large enough to take 50ms
# would be memory-bandwidth-bound, and concurrent copies would contend on
# bandwidth rather than on CPU — the opposite of MiniLM, whose forward pass is
# a chain of small matmuls over a 90MB model.
MAX_MATMUL_SECONDS = 0.005

# Captured before anything calls set_num_threads, so the "all cores" configs
# can be restored to torch's own default rather than a guess.
try:
    import torch

    DEFAULT_TORCH_THREADS: int | None = torch.get_num_threads()
except ImportError:  # pragma: no cover - torch is a hard dep of the service
    torch = None  # type: ignore[assignment]
    DEFAULT_TORCH_THREADS = None


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class SleepEmbedder(Embedder):
    """Blocks without consuming CPU. Measures queueing, not contention."""

    def __init__(self, seconds: float = SLEEP_SECONDS) -> None:
        self.seconds = seconds

    def encode(self, texts: list[str]) -> list[list[float]]:
        time.sleep(self.seconds)
        return [[0.1] * 384 for _ in texts]


class CpuEmbedder(Embedder):
    """Burns real CPU via torch.mm, which releases the GIL like the model does.

    Matrices are allocated once per instance so each encode() costs a
    predictable ~2*dim^3 flops with no allocation noise, and the cost scales
    with torch.set_num_threads exactly as inference does.
    """

    def __init__(self, dim: int, repeats: int) -> None:
        self.dim = dim
        self.repeats = repeats
        self._a = torch.rand(dim, dim)
        self._b = torch.rand(dim, dim)

    def encode(self, texts: list[str]) -> list[list[float]]:
        for _ in range(self.repeats):
            product = torch.mm(self._a, self._b)
        float(product[0, 0])  # force materialisation
        return [[0.1] * 384 for _ in texts]


@dataclass(frozen=True)
class Calibration:
    dim: int
    repeats: int
    matmul_ms: float
    encode_ms: float


def calibrate_cpu_embedder(
    target: float = TARGET_CPU_SECONDS,
    dims: tuple[int, ...] = CALIBRATION_DIMS,
) -> Calibration:
    """Size the CPU fake to ~`target` seconds of single-threaded work.

    Picks the largest cache-friendly matmul, then repeats it to hit the
    target. Must run in the environment being measured: BLAS kernels and clock
    speeds differ enough that a calibration from the host does not transfer
    into a CPU-limited container.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:

        def time_encode(embedder: CpuEmbedder) -> float:
            embedder.encode(["warmup"])  # first mm pays lazy init
            return statistics.median(
                _time(lambda: embedder.encode(["x"])) for _ in range(3)
            )

        chosen_dim, chosen_seconds = dims[0], time_encode(CpuEmbedder(dims[0], 1))
        for dim in dims[1:]:
            seconds = time_encode(CpuEmbedder(dim, 1))
            if seconds > MAX_MATMUL_SECONDS:
                break
            chosen_dim, chosen_seconds = dim, seconds

        repeats = max(1, round(target / chosen_seconds))
        embedder = CpuEmbedder(chosen_dim, repeats)
        return Calibration(
            dim=chosen_dim,
            repeats=repeats,
            matmul_ms=chosen_seconds * 1000,
            encode_ms=time_encode(embedder) * 1000,
        )
    finally:
        torch.set_num_threads(previous)


def _time(fn: Callable[[], object]) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


# --------------------------------------------------------------------------
# Dispatch configurations
# --------------------------------------------------------------------------

Dispatch = Callable[[Embedder, list[str]], Awaitable[list[list[float]]]]


async def dispatch_inline(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Pre-fix: blocking call straight from the coroutine."""
    return embedder.encode(texts)


async def dispatch_to_thread(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """First fix: offloaded, but to the loop's default (unbounded-ish) pool."""
    return await asyncio.to_thread(embedder.encode, texts)


@dataclass(frozen=True)
class Config:
    name: str
    dispatch: Dispatch
    torch_threads: int | None  # None = torch's own default (all cores)


CONFIGS: tuple[Config, ...] = (
    Config("blocking", dispatch_inline, None),
    Config("unbounded", dispatch_to_thread, None),
    Config("unbounded+pinned", dispatch_to_thread, 1),
    Config("bounded", encode_async, 1),
)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass
class RunResult:
    wall_s: float
    cores_used: float
    max_loop_lag_ms: float


@dataclass
class Cell:
    fake: str
    config: str
    concurrency: int
    median_s: float
    min_s: float
    max_s: float
    cores_used: float
    max_loop_lag_ms: float
    runs: int
    spread_pct: float = field(default=0.0)

    @property
    def noisy(self) -> bool:
        return self.spread_pct > 20.0


class _StubResult:
    def mappings(self):
        return self

    def all(self):
        return []


class _StubSession:
    """Stands in for AsyncSession: the DB is not what we are measuring."""

    async def execute(self, *args, **kwargs):
        return _StubResult()


async def _heartbeat(stop: asyncio.Event, lags: list[float]) -> None:
    """Records how late a 5ms timer fires — i.e. how blocked the loop is."""
    while not stop.is_set():
        started = time.perf_counter()
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        lags.append(time.perf_counter() - started - HEARTBEAT_INTERVAL)


async def _run_once(client: AsyncClient, concurrency: int) -> RunResult:
    lags: list[float] = []
    stop = asyncio.Event()
    beat = asyncio.create_task(_heartbeat(stop, lags))
    await asyncio.sleep(0.02)  # let the heartbeat settle

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    responses = await asyncio.gather(
        *(
            client.post("/search", json={"query": f"query {i}"})
            for i in range(concurrency)
        )
    )
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start

    stop.set()
    await beat

    for response in responses:
        if response.status_code != 200:
            raise RuntimeError(f"/search returned {response.status_code}")

    return RunResult(
        wall_s=wall,
        # process_time() sums CPU across every thread, so this is the mean
        # number of cores the process kept busy over the run.
        cores_used=cpu / wall if wall > 0 else 0.0,
        max_loop_lag_ms=max(lags) * 1000 if lags else 0.0,
    )


def default_executor_size() -> int:
    """What asyncio.to_thread gets when the loop makes its own executor."""
    return min(32, (os.cpu_count() or 1) + 4)


def _apply(config: Config, bounded_workers: int | None) -> None:
    """Install a dispatch mode and pin torch *in the threads that will run it*.

    torch resolves intra-op threads per thread on first use, so pinning has to
    happen inside each worker; setting it from here would silently leave the
    workers on all cores. See app.services.embedder.pin_torch_threads.
    """
    retriever_module.encode_async = config.dispatch

    if config.dispatch is dispatch_inline and torch is not None:
        # Runs on the event-loop thread, so this thread is the one to set.
        torch.set_num_threads(config.torch_threads or DEFAULT_TORCH_THREADS or 1)

    if config.dispatch is dispatch_to_thread and config.torch_threads:
        # Same size the loop would have picked, but with pinned workers, so
        # this row differs from "unbounded" in exactly one variable.
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(
                max_workers=default_executor_size(),
                thread_name_prefix="to-thread-pinned",
                initializer=pin_torch_threads,
                initargs=(config.torch_threads,),
            )
        )

    if config.dispatch is encode_async:
        # init_embed_executor pins its own workers from settings.
        init_embed_executor(bounded_workers)


async def measure_cell(
    config: Config,
    fake_name: str,
    make_fake: Callable[[], Embedder],
    concurrency: int,
    reps: int,
    warmups: int,
    bounded_workers: int | None,
) -> Cell:
    from app.main import app

    _apply(config, bounded_workers)
    embedder = make_fake()
    app.dependency_overrides[get_retriever] = lambda: Retriever(embedder)
    app.dependency_overrides[get_db] = _stub_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://bench") as client:
            results: list[RunResult] = []
            for index in range(warmups + reps):
                result = await _run_once(client, concurrency)
                if index >= warmups:
                    results.append(result)
    finally:
        app.dependency_overrides.clear()
        retriever_module.encode_async = encode_async

    walls = [r.wall_s for r in results]
    median = statistics.median(walls)
    return Cell(
        fake=fake_name,
        config=config.name,
        concurrency=concurrency,
        median_s=median,
        min_s=min(walls),
        max_s=max(walls),
        cores_used=statistics.median([r.cores_used for r in results]),
        max_loop_lag_ms=max(r.max_loop_lag_ms for r in results),
        runs=len(results),
        spread_pct=((max(walls) - min(walls)) / median * 100) if median else 0.0,
    )


async def _stub_db():
    yield _StubSession()


def _default_executor_workers() -> int:
    """The size asyncio.to_thread actually gets, read from a live loop."""

    async def probe() -> int:
        await asyncio.to_thread(lambda: None)
        executor = asyncio.get_running_loop()._default_executor
        return getattr(executor, "_max_workers", default_executor_size())

    return asyncio.run(probe())


def environment(label: str, calibration: Calibration) -> dict:
    budget = cpu_budget()
    return {
        "label": label,
        "affinity_cpus": budget.affinity_cpus,
        "cgroup_quota_cpus": budget.quota_cpus,
        "budget_cpus": budget.cpus,
        "budget_source": budget.source,
        "os_cpu_count": os.cpu_count(),
        "default_executor_workers": _default_executor_workers(),
        "bounded_workers": resolve_max_workers(None, budget),
        "torch_default_threads": DEFAULT_TORCH_THREADS,
        "cpu_fake_dim": calibration.dim,
        "cpu_fake_repeats": calibration.repeats,
        "cpu_fake_matmul_ms": round(calibration.matmul_ms, 2),
        "cpu_fake_single_thread_ms": round(calibration.encode_ms, 1),
        "sleep_fake_ms": SLEEP_SECONDS * 1000,
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None),
    }


async def run_cell(
    fake_name: str,
    config_name: str,
    concurrencies: tuple[int, ...],
    reps: int,
    warmups: int,
    calibration: Calibration,
    bounded_workers: int | None,
) -> list[Cell]:
    """Every concurrency level for one (fake, config), in this process."""
    config = {c.name: c for c in CONFIGS}[config_name]
    make_fake: Callable[[], Embedder] = (
        SleepEmbedder
        if fake_name == "sleep"
        else lambda: CpuEmbedder(calibration.dim, calibration.repeats)
    )

    cells = []
    for concurrency in concurrencies:
        cell = await measure_cell(
            config, fake_name, make_fake, concurrency, reps, warmups, bounded_workers
        )
        cells.append(cell)
        print(
            f"  {fake_name:6s} {config.name:17s} N={concurrency:<3d} "
            f"{cell.median_s:7.3f}s  cores={cell.cores_used:4.1f}  "
            f"lag={cell.max_loop_lag_ms:7.1f}ms"
            f"{'  NOISY' if cell.noisy else ''}",
            file=sys.stderr,
            flush=True,
        )
    return cells


def run_matrix_isolated(args, calibration: Calibration) -> list[Cell]:
    """One subprocess per (fake, config).

    Configs cannot share a process: torch's per-thread intra-op pools and the
    executors themselves outlive a cell, so a config that ran wide leaves spin
    -waiting threads that inflate the next config's CPU accounting.
    """
    cells: list[Cell] = []
    for fake_name in ("sleep", "cpu"):
        for config in CONFIGS:
            command = [
                sys.executable,
                "-m",
                "tests.concurrency_harness",
                "--cell",
                f"{fake_name}:{config.name}",
                "--dim",
                str(calibration.dim),
                "--repeats",
                str(calibration.repeats),
                "--concurrency",
                args.concurrency,
                "--reps",
                str(args.reps),
                "--warmups",
                str(args.warmups),
            ]
            if args.workers is not None:
                command += ["--workers", str(args.workers)]
            completed = subprocess.run(command, capture_output=True, text=True)
            sys.stderr.write(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"cell {fake_name}:{config.name} failed ({completed.returncode})"
                )
            cells += [Cell(**raw) for raw in json.loads(completed.stdout)]
    return cells


def render_markdown(env: dict, cells: list[Cell]) -> str:
    """Table caption plus one block per fake, ready to paste into the README."""
    quota = env["cgroup_quota_cpus"]
    lines = [
        f"### {env['label']}",
        "",
        f"CPU quota {quota if quota else 'none'} "
        f"(affinity sees {env['affinity_cpus']}), "
        f"default executor {env['default_executor_workers']} threads, "
        f"bounded pool {env['bounded_workers']} threads, "
        f"torch default {env['torch_default_threads']} threads. "
        f"CPU fake: {env['cpu_fake_dim']}x{env['cpu_fake_dim']} matmul "
        f"x{env['cpu_fake_repeats']} = {env['cpu_fake_single_thread_ms']}ms "
        f"single-threaded. Median of {cells[0].runs} runs after 1 warmup.",
    ]
    for fake in dict.fromkeys(cell.fake for cell in cells):
        lines += [
            "",
            f"**{fake} fake**",
            "",
            "| config | N | median | min-max | cores | max loop lag |",
            "|---|---|---|---|---|---|",
        ]
        for cell in cells:
            if cell.fake != fake:
                continue
            flag = " ⚠" if cell.noisy else ""
            lines.append(
                f"| {cell.config} | {cell.concurrency} | "
                f"{cell.median_s:.3f}s | "
                f"{cell.min_s:.3f}-{cell.max_s:.3f}{flag} | "
                f"{cell.cores_used:.1f} | {cell.max_loop_lag_ms:.0f}ms |"
            )
    if any(cell.noisy for cell in cells):
        lines += ["", "⚠ min-max spread exceeds 20% of the median."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="host", help="environment label")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--concurrency", default="1,10,50")
    parser.add_argument("--workers", type=int, default=None, help="bounded pool size")
    parser.add_argument("--json", default=None, help="write results here")
    parser.add_argument("--cell", default=None, help="internal: run one fake:config")
    parser.add_argument("--dim", type=int, default=None, help="internal: matmul dim")
    parser.add_argument("--repeats", type=int, default=None, help="internal: matmuls")
    args = parser.parse_args()

    # Per-request INFO logging from the endpoint and httpx would both pollute
    # stdout and show up in the measurements.
    logging.disable(logging.INFO)

    concurrencies = tuple(int(n) for n in args.concurrency.split(","))

    if args.cell:
        fake_name, config_name = args.cell.split(":")
        # The matrix driver passes a calibration down so every cell measures
        # identical work; a cell run directly has to calibrate for itself.
        calibration = (
            Calibration(args.dim, args.repeats, 0.0, 0.0)
            if args.dim is not None
            else calibrate_cpu_embedder()
        )
        print(
            f"  fake: dim={calibration.dim} x{calibration.repeats} "
            f"= {calibration.encode_ms:.1f}ms single-threaded",
            file=sys.stderr,
        )
        cells = asyncio.run(
            run_cell(
                fake_name,
                config_name,
                concurrencies,
                args.reps,
                args.warmups,
                calibration,
                args.workers,
            )
        )
        shutdown_embed_executor()
        print(json.dumps([asdict(c) for c in cells]))
        return

    target_ms = TARGET_CPU_SECONDS * 1000
    print(f"calibrating torch.mm for ~{target_ms:.0f}ms...", file=sys.stderr)
    calibration = calibrate_cpu_embedder()
    print(
        f"  dim={calibration.dim} x {calibration.repeats} "
        f"({calibration.matmul_ms:.2f}ms each) "
        f"-> {calibration.encode_ms:.1f}ms single-threaded",
        file=sys.stderr,
    )

    env = environment(args.label, calibration)
    print(json.dumps(env, indent=2), file=sys.stderr)

    cells = run_matrix_isolated(args, calibration)

    payload = {"env": env, "cells": [asdict(c) for c in cells]}
    if args.json:
        with open(args.json, "w") as handle:
            handle.write(json.dumps(payload, indent=2))
    print(render_markdown(env, cells))


if __name__ == "__main__":
    main()
