"""Tests for optimizer.py — greedy knapsack assignment."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quant_sim import QUANT_LEVELS, LEVELS_BY_NAME
from src.snr_profiler import TensorProfile, LevelResult
from src.optimizer import (
    compute_model_snr,
    compute_total_bytes,
    optimize_for_snr,
    optimize_for_size,
    Assignment,
    _per_tensor_floor_assignment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_profile(
    name: str,
    num_params: int = 4096,
    policy: str = "optimize",
    snr_by_level: dict | None = None,
) -> TensorProfile:
    """Build a TensorProfile with synthetic SNR/byte values."""
    if snr_by_level is None:
        # Default: realistic SNR curve covering all levels
        snr_by_level = {
            "Q2_K":    18.0,
            "Q3_K_S":  20.0,
            "Q3_K_M":  24.0,
            "Q3_K_L":  26.0,
            "Q4_0":    28.0,
            "Q4_K_S":  29.0,
            "Q4_K_M":  31.0,
            "Q5_0":    34.0,
            "Q5_K_S":  36.0,
            "Q5_K_M":  38.0,
            "Q6_K":    44.0,
            "Q8_0":    51.0,
            "F16":     96.0,
        }
    levels = {
        name: LevelResult(
            snr_db=snr,
            estimated_bytes=LEVELS_BY_NAME[name].estimated_bytes(num_params),
        )
        for name, snr in snr_by_level.items()
        if name in LEVELS_BY_NAME
    }
    return TensorProfile(
        name=name,
        shape=[num_params],
        num_params=num_params,
        tensor_class="mlp_ffn",
        policy=policy,
        levels=levels,
    )


def make_profiles(n: int = 10, **kwargs) -> list[TensorProfile]:
    return [_make_profile(f"layer.{i}.weight", **kwargs) for i in range(n)]


# ---------------------------------------------------------------------------
# compute_model_snr
# ---------------------------------------------------------------------------

class TestComputeModelSnr:
    def test_uniform_assignment_returns_that_snr(self):
        profiles = make_profiles(10)
        levels = {p.name: "Q4_K_M" for p in profiles}
        snr = compute_model_snr(levels, profiles)
        assert abs(snr - 31.0) < 0.1

    def test_p5_catches_worst_tensors(self):
        """One tensor with terrible SNR should pull down the P5."""
        profiles = make_profiles(20)
        # Give tensor 0 a catastrophically bad Q4_K_M SNR
        profiles[0] = _make_profile(
            "layer.0.weight", num_params=100_000,
            snr_by_level={"Q2_K": 5.0, "Q3_K_M": 8.0, "Q4_K_M": 10.0,
                          "Q5_K_M": 15.0, "Q6_K": 20.0, "Q8_0": 28.0, "F16": 96.0},
        )
        levels = {p.name: "Q4_K_M" for p in profiles}
        snr = compute_model_snr(levels, profiles)
        # Most tensors are at 31 dB, but the big bad tensor should pull P5 down
        assert snr < 31.0

    def test_empty_returns_zero(self):
        assert compute_model_snr({}, []) == 0.0


# ---------------------------------------------------------------------------
# compute_total_bytes
# ---------------------------------------------------------------------------

class TestComputeTotalBytes:
    def test_sum_of_individual_bytes(self):
        profiles = make_profiles(4, num_params=1024)
        levels = {p.name: "Q4_K_M" for p in profiles}
        expected = sum(LEVELS_BY_NAME["Q4_K_M"].estimated_bytes(1024) for _ in range(4))
        assert compute_total_bytes(levels, profiles) == expected

    def test_mixed_levels(self):
        p1 = _make_profile("a", num_params=1024)
        p2 = _make_profile("b", num_params=2048)
        levels = {"a": "Q2_K", "b": "F16"}
        expected = (
            LEVELS_BY_NAME["Q2_K"].estimated_bytes(1024)
            + LEVELS_BY_NAME["F16"].estimated_bytes(2048)
        )
        assert compute_total_bytes(levels, profiles=[p1, p2]) == expected


# ---------------------------------------------------------------------------
# optimize_for_snr
# ---------------------------------------------------------------------------

class TestOptimizeForSnr:
    def test_meets_floor(self):
        profiles = make_profiles(20)
        result = optimize_for_snr(profiles, snr_floor=30.0)
        assert result.model_snr_db >= 30.0

    def test_cheaper_than_f16(self):
        """The optimizer should find something smaller than all-F16."""
        profiles = make_profiles(20)
        all_f16_bytes = compute_total_bytes(
            {p.name: "F16" for p in profiles}, profiles
        )
        result = optimize_for_snr(profiles, snr_floor=30.0)
        assert result.total_bytes < all_f16_bytes

    def test_high_floor_upgrades_to_q8(self):
        """A very high SNR floor should push most tensors to Q8 or F16."""
        profiles = make_profiles(10)
        result = optimize_for_snr(profiles, snr_floor=50.0)
        counts = result.level_counts()
        high_quality = counts.get("Q8_0", 0) + counts.get("F16", 0)
        assert high_quality == 10

    def test_per_tensor_floor_guarantees_model_snr(self):
        """Per-tensor assignment guarantees model P5 >= floor."""
        profiles = make_profiles(50)
        for floor in [20.0, 30.0, 40.0]:
            result = optimize_for_snr(profiles, snr_floor=floor, refine=False)
            assert result.model_snr_db >= floor, (
                f"floor={floor}: got model_snr={result.model_snr_db:.1f}"
            )

    def test_large_tensors_not_over_upgraded(self):
        """Large tensors should not be pushed to F16 when Q4_K_M meets the floor."""
        # 10 large tensors where Q4_K_M SNR = 31 dB
        profiles = make_profiles(10, num_params=50_000_000)
        result = optimize_for_snr(profiles, snr_floor=30.0, refine=False)
        counts = result.level_counts()
        # Should be at Q4_K_M, not F16
        assert counts.get("F16", 0) == 0
        assert counts.get("Q4_K_M", 0) == 10

    def test_fixed_f16_tensors_unchanged(self):
        profiles = make_profiles(5)
        fixed = _make_profile("norm.weight", num_params=4096, policy="always_f16")
        fixed.levels.clear()
        fixed.levels["F16"] = LevelResult(snr_db=96.0, estimated_bytes=8192)
        all_profiles = profiles + [fixed]
        result = optimize_for_snr(all_profiles, snr_floor=25.0)
        assert result.levels["norm.weight"] == "F16"

    def test_embedding_never_below_q8(self):
        embed = _make_profile("embed.weight", policy="always_q8_min", num_params=1024)
        # Remove ALL levels below Q8_0 — this is what snr_profiler does for
        # always_q8_min tensors (skips any level cheaper than Q8_0 by bpw).
        q8_bpw = LEVELS_BY_NAME["Q8_0"].bits_per_weight
        for lvl in list(embed.levels.keys()):
            if lvl in LEVELS_BY_NAME and LEVELS_BY_NAME[lvl].bits_per_weight < q8_bpw:
                embed.levels.pop(lvl)
        profiles = [embed] + make_profiles(5)
        result = optimize_for_snr(profiles, snr_floor=20.0)
        assigned = result.levels["embed.weight"]
        level_names = [q.name for q in QUANT_LEVELS]
        assert level_names.index(assigned) >= level_names.index("Q8_0")


# ---------------------------------------------------------------------------
# optimize_for_size
# ---------------------------------------------------------------------------

class TestOptimizeForSize:
    def test_fits_budget(self):
        profiles = make_profiles(20, num_params=4096)
        all_f16 = compute_total_bytes(
            {p.name: "F16" for p in profiles}, profiles
        )
        budget = all_f16 // 3
        result = optimize_for_size(profiles, budget)
        assert result.total_bytes <= budget * 1.05  # allow 5% overshoot from rounding

    def test_small_budget_maximizes_snr_within_budget(self):
        """Two budgets: larger budget should get equal or better SNR."""
        profiles = make_profiles(10, num_params=4096)
        f16_bytes = compute_total_bytes({p.name: "F16" for p in profiles}, profiles)
        result_big = optimize_for_size(profiles, f16_bytes // 2)
        result_small = optimize_for_size(profiles, f16_bytes // 4)
        assert result_big.model_snr_db >= result_small.model_snr_db

    def test_returns_assignment_type(self):
        profiles = make_profiles(5)
        result = optimize_for_size(profiles, 10_000_000)
        assert isinstance(result, Assignment)
        assert len(result.levels) == 5
