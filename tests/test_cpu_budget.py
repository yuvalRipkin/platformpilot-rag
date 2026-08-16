"""Tests for container-aware CPU budgeting.

The real files live under /sys/fs/cgroup and differ between cgroup v1 and v2
hosts, so every case here points the reader at a fixture directory instead.
Formats verified against `docker run --cpus=N python:3.12-slim`:

    --cpus=1     ->  cpu.max == "100000 100000"
    --cpus=1.5   ->  cpu.max == "150000 100000"
    (no limit)   ->  cpu.max == "max 100000"
"""

from pathlib import Path

import pytest

from app.services.embedder import (
    MAX_DERIVED_WORKERS,
    CpuBudget,
    cgroup_quota_cpus,
    cpu_budget,
    resolve_max_workers,
)


def _v2(tmp_path: Path, contents: str) -> Path:
    (tmp_path / "cpu.max").write_text(contents)
    return tmp_path


def _v1(tmp_path: Path, quota: str, period: str) -> Path:
    cpu_dir = tmp_path / "cpu"
    cpu_dir.mkdir()
    (cpu_dir / "cpu.cfs_quota_us").write_text(quota)
    (cpu_dir / "cpu.cfs_period_us").write_text(period)
    return tmp_path


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("100000 100000\n", 1.0),
        ("150000 100000\n", 1.5),
        ("200000 100000\n", 2.0),
        ("50000 100000\n", 0.5),
        ("max 100000\n", None),
    ],
)
def test_reads_cgroup_v2(tmp_path: Path, contents: str, expected: float | None):
    assert cgroup_quota_cpus(_v2(tmp_path, contents)) == expected


@pytest.mark.parametrize(
    ("quota", "period", "expected"),
    [
        ("100000\n", "100000\n", 1.0),
        ("150000\n", "100000\n", 1.5),
        ("-1\n", "100000\n", None),  # v1 spells "unlimited" as -1
    ],
)
def test_reads_cgroup_v1(
    tmp_path: Path, quota: str, period: str, expected: float | None
):
    assert cgroup_quota_cpus(_v1(tmp_path, quota, period)) == expected


def test_v2_wins_when_both_present(tmp_path: Path):
    _v1(tmp_path, "400000\n", "100000\n")
    _v2(tmp_path, "100000 100000\n")
    assert cgroup_quota_cpus(tmp_path) == 1.0


def test_missing_and_malformed_files_are_not_fatal(tmp_path: Path):
    assert cgroup_quota_cpus(tmp_path / "nonexistent") is None
    assert cgroup_quota_cpus(_v2(tmp_path, "garbage\n")) is None
    assert cgroup_quota_cpus(_v2(tmp_path, "")) is None
    assert cgroup_quota_cpus(_v2(tmp_path, "100000\n")) is None  # period missing
    assert cgroup_quota_cpus(_v2(tmp_path, "0 0\n")) is None


def test_quota_below_affinity_wins(tmp_path: Path):
    """The K8s case: limits.cpu=1 on a many-core node."""
    budget = cpu_budget(_v2(tmp_path, "100000 100000\n"))
    assert budget.cpus == 1
    assert budget.source == "cgroup"
    assert budget.quota_cpus == 1.0
    assert budget.affinity_cpus >= 1


def test_fractional_quota_floors(tmp_path: Path):
    budget = cpu_budget(_v2(tmp_path, "150000 100000\n"))
    assert budget.cpus == 1
    assert budget.source == "cgroup"


def test_sub_core_quota_still_yields_one(tmp_path: Path):
    budget = cpu_budget(_v2(tmp_path, "10000 100000\n"))
    assert budget.cpus == 1


def test_falls_back_to_affinity_without_quota(tmp_path: Path):
    budget = cpu_budget(_v2(tmp_path, "max 100000\n"))
    assert budget.source == "affinity"
    assert budget.cpus == budget.affinity_cpus
    assert budget.quota_cpus is None


def test_quota_above_affinity_does_not_inflate(tmp_path: Path):
    """A quota larger than the visible cores must not raise the bound."""
    budget = cpu_budget(_v2(tmp_path, "6400000 100000\n"))
    assert budget.source == "affinity"
    assert budget.cpus == budget.affinity_cpus


def _budget(cpus: int) -> CpuBudget:
    return CpuBudget(cpus=cpus, source="cgroup", affinity_cpus=64, quota_cpus=cpus)


@pytest.mark.parametrize(
    ("cpus", "expected"),
    [
        (1, 1),  # single-core pod
        (2, 2),  # one worker per core; reserving one was measured and lost
        (4, 4),
        (64, MAX_DERIVED_WORKERS),  # clamped
    ],
)
def test_derived_worker_counts(cpus: int, expected: int):
    assert resolve_max_workers(None, _budget(cpus)) == expected


def test_explicit_config_overrides_derivation():
    assert resolve_max_workers(16, _budget(1)) == 16


def test_rejects_nonpositive_worker_count():
    with pytest.raises(ValueError):
        resolve_max_workers(0, _budget(8))
