"""Proves the embedder no longer blocks the event loop, and stays bounded.

Marked `concurrency` and excluded from the default suite: these tests measure
wall time, so they are slow by construction and meaningless on a machine that
is already saturated.

    uv run pytest -m concurrency

Assertions here are deliberately coarse — they encode the *shape* of the fix
(serialized vs. concurrent, loop stalled vs. responsive), not performance
numbers. The quantitative matrix lives in tests/concurrency_harness.py and is
meant to be run in a CPU-limited container; see `make bench-concurrency`.

Every case uses SleepEmbedder rather than the CPU fake. time.sleep consumes no
CPU, so the results do not depend on how many cores the runner has or what
else is running on them — which is what makes the thresholds below safe on a
loaded CI box. The tradeoff is that a sleeping thread never contends for CPU,
so these tests cannot see oversubscription at all; that is the harness's job.
"""

import pytest

from tests.concurrency_harness import (
    CONFIGS,
    SLEEP_SECONDS,
    SleepEmbedder,
    measure_cell,
)

pytestmark = pytest.mark.concurrency

CONCURRENCY = 10
WORKERS = 4

# With WORKERS=4 and N=10 the bounded pool needs ceil(10/4) = 3 waves, so the
# floor is 3 * 50ms = 150ms. 400ms is ~2.7x that floor: enough headroom for a
# slow, contended CI runner while still being far below the 500ms that
# serialized execution cannot beat. The two bands cannot overlap.
SERIALIZED_FLOOR_S = 0.8 * CONCURRENCY * SLEEP_SECONDS  # 0.40s
CONCURRENT_CEILING_S = 0.40

# A blocking encode holds the loop for a full 50ms; a heartbeat scheduled
# every 5ms therefore fires ~50ms late. 25ms cleanly separates "stalled by an
# encode" from "merely descheduled", and 20ms of slack over the 5ms interval
# absorbs GC pauses and scheduler jitter on a busy runner.
STALLED_LOOP_LAG_MS = 25.0

_BY_NAME = {config.name: config for config in CONFIGS}


async def _measure(config_name: str):
    return await measure_cell(
        _BY_NAME[config_name],
        fake_name="sleep",
        make_fake=SleepEmbedder,
        concurrency=CONCURRENCY,
        reps=1,
        warmups=1,
        bounded_workers=WORKERS,
    )


async def test_inline_encode_serializes_requests():
    """The bug: 10 concurrent requests cost 10x one request."""
    cell = await _measure("blocking")
    assert cell.median_s >= SERIALIZED_FLOOR_S, (
        f"expected ~{CONCURRENCY * SLEEP_SECONDS:.2f}s of serialized work, "
        f"got {cell.median_s:.3f}s"
    )


async def test_inline_encode_stalls_the_event_loop():
    """The reason it matters: nothing else on the loop can run either."""
    cell = await _measure("blocking")
    assert cell.max_loop_lag_ms >= STALLED_LOOP_LAG_MS, (
        f"expected the loop to stall for ~{SLEEP_SECONDS * 1000:.0f}ms, "
        f"saw {cell.max_loop_lag_ms:.1f}ms"
    )


async def test_bounded_pool_runs_requests_concurrently():
    cell = await _measure("bounded")
    assert cell.median_s < CONCURRENT_CEILING_S, (
        f"expected under {CONCURRENT_CEILING_S:.2f}s with {WORKERS} workers, "
        f"got {cell.median_s:.3f}s"
    )


async def test_bounded_pool_keeps_the_event_loop_responsive():
    cell = await _measure("bounded")
    assert cell.max_loop_lag_ms < STALLED_LOOP_LAG_MS, (
        f"loop stalled for {cell.max_loop_lag_ms:.1f}ms despite offloading"
    )


async def test_bounded_pool_admits_at_most_max_workers():
    """The pool must queue past WORKERS, not widen.

    N=10 through 4 workers is 3 waves, so it cannot beat 2 sleeps end to end;
    an unbounded pool would finish in roughly one.
    """
    cell = await _measure("bounded")
    assert cell.median_s >= 2 * SLEEP_SECONDS, (
        f"{cell.median_s:.3f}s is too fast for {WORKERS} workers — "
        "the bound is not being applied"
    )
