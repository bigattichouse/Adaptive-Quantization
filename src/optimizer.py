"""
optimizer.py — Mixed-precision optimizer.

Two modes:
  - snr_floor:   find smallest model with model_snr >= floor
  - size_budget: find highest model_snr that fits in budget (bytes)

The optimizer only moves "optimize" policy tensors.  Fixed tensors
(always_f16, always_q8_min) contribute their fixed SNR to model_snr
but are never reassigned.

Model SNR is defined as the P5 weighted SNR:
  - Sort all tensors by SNR ascending
  - Walk from lowest to highest, accumulating num_params weights
  - Model SNR = SNR value when 5% of total params have been covered
This catches the worst 5% of parameters by weight — conservative
but not dominated by a single small tensor.

SNR-floor optimizer strategy (why not greedy snr_gain/byte_cost):
  The naive greedy metric "snr_gain / byte_cost" gives efficiency ∝ 1/num_params
  because byte_cost ∝ num_params.  Small tensors always win, get upgraded to F16
  while large tensors (which actually determine the P5) stay at Q2_K.

  Instead we use per-tensor floor assignment:
    1. Each tensor gets the cheapest level where its own SNR >= floor.
       If every tensor individually meets the floor, model P5 >= floor — guaranteed.
    2. Refinement: try downgrading each tensor one level.  If P5 still >= floor,
       keep the downgrade (small tensors below the individual floor don't affect P5
       if their num_params < 5% of total).
  This is O(N*K) for phase 1 and O(N*K) per refinement pass — much faster and
  more correct than the greedy approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .quant_sim import QUANT_LEVELS, LEVELS_BY_NAME, next_level, prev_level
from .snr_profiler import TensorProfile, LevelResult


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

@dataclass
class Assignment:
    """Quant level assigned to each tensor, plus aggregate metrics."""
    levels: dict[str, str]          # tensor_name -> level_name
    total_bytes: int
    model_snr_db: float
    model_snr_percentile: float = 5.0

    def level_counts(self) -> dict[str, int]:
        """Return {level_name: tensor_count}, ordered cheapest-first."""
        order = [q.name for q in QUANT_LEVELS]
        counts: dict[str, int] = {}
        for lvl in self.levels.values():
            counts[lvl] = counts.get(lvl, 0) + 1
        return {k: counts[k] for k in order if k in counts}


# ---------------------------------------------------------------------------
# Model SNR calculation
# ---------------------------------------------------------------------------

def compute_model_snr(
    levels: dict[str, str],
    profiles: list[TensorProfile],
    percentile: float = 5.0,
) -> float:
    """Compute weighted-percentile SNR across all tensors.

    Args:
        levels:     {tensor_name: level_name} assignment.
        profiles:   Full list of TensorProfile objects.
        percentile: Which percentile to use (5 = conservative P5).

    Returns:
        SNR in dB at the given percentile, weighted by num_params.
    """
    pairs: list[tuple[float, int]] = []
    for p in profiles:
        lvl = levels.get(p.name)
        if lvl is None or lvl not in p.levels:
            continue
        pairs.append((p.levels[lvl].snr_db, p.num_params))

    if not pairs:
        return 0.0

    pairs.sort(key=lambda x: x[0])
    total_weight = sum(w for _, w in pairs)
    target = total_weight * percentile / 100.0

    cumulative = 0.0
    for snr, w in pairs:
        cumulative += w
        if cumulative >= target:
            return snr
    return pairs[-1][0]


def compute_total_bytes(
    levels: dict[str, str],
    profiles: list[TensorProfile],
) -> int:
    total = 0
    for p in profiles:
        lvl = levels.get(p.name)
        if lvl and lvl in p.levels:
            total += p.levels[lvl].estimated_bytes
    return total


# ---------------------------------------------------------------------------
# Level-agnostic helpers (work with GGUF levels, codebook levels, or any mix)
# ---------------------------------------------------------------------------

def _levels_by_cost(levels_dict: dict[str, LevelResult]) -> list[tuple[str, LevelResult]]:
    """Return (name, result) pairs sorted cheapest-first by estimated_bytes."""
    return sorted(levels_dict.items(), key=lambda kv: kv[1].estimated_bytes)


def _cheapest_level(levels_dict: dict[str, LevelResult]) -> str:
    """Name of the cheapest level in the profile."""
    return min(levels_dict.items(), key=lambda kv: kv[1].estimated_bytes)[0]


def _next_cheaper(levels_dict: dict[str, LevelResult], current: str) -> str | None:
    """Name of the next cheaper level by estimated_bytes, or None if at minimum."""
    cur_bytes = levels_dict[current].estimated_bytes
    cheaper = {n: lr for n, lr in levels_dict.items() if lr.estimated_bytes < cur_bytes}
    if not cheaper:
        return None
    # Closest step down: the one with the most bytes among those cheaper
    return max(cheaper.items(), key=lambda kv: kv[1].estimated_bytes)[0]


# ---------------------------------------------------------------------------
# Phase 1: per-tensor floor assignment
# ---------------------------------------------------------------------------

def _per_tensor_floor_assignment(
    profiles: list[TensorProfile],
    snr_floor: float,
) -> dict[str, str]:
    """Assign each tensor to the cheapest level where its own SNR >= snr_floor.

    Works with any level scheme (GGUF, codebook, or mixed) — iterates the
    profile's own levels dict sorted by cost rather than a fixed level list.

    For always-fixed tensors the profile minimum is used regardless of the floor.
    For 'optimize' tensors: cheapest level with per-tensor SNR >= floor.
    If no level meets the floor (rare), F16 is used.
    """
    assignment: dict[str, str] = {}
    for p in profiles:
        if p.policy == "always_f16":
            assignment[p.name] = "F16"
            continue

        if p.policy == "always_q8_min":
            # Profile already excludes sub-minimum levels; just take the cheapest.
            assignment[p.name] = _cheapest_level(p.levels) if p.levels else "F16"
            continue

        # optimize: walk cheapest-first, pick first that meets the floor
        chosen = "F16"
        for name, lr in _levels_by_cost(p.levels):
            if lr.snr_db >= snr_floor:
                chosen = name
                break
        assignment[p.name] = chosen

    return assignment


# ---------------------------------------------------------------------------
# Phase 2: refinement — downgrade if P5 still holds
# ---------------------------------------------------------------------------

def _refine_downgrade(
    levels: dict[str, str],
    profiles: list[TensorProfile],
    snr_floor: float,
    percentile: float,
) -> dict[str, str]:
    """Try to downgrade each tensor one level; keep if model P5 still meets floor.

    Iterates until no more downgrades are possible.  Tries smallest tensors
    first since they have the least impact on the P5 and are most likely to
    be downgradeable.
    """
    optim = sorted(
        [p for p in profiles if p.policy == "optimize"],
        key=lambda p: p.num_params,   # small first — most likely to be free
    )

    changed = True
    while changed:
        changed = False
        for p in optim:
            current = levels[p.name]
            pv = _next_cheaper(p.levels, current)
            if pv is None:
                continue

            levels[p.name] = pv
            if compute_model_snr(levels, profiles, percentile) >= snr_floor:
                changed = True          # keep downgrade, continue scanning
            else:
                levels[p.name] = current    # revert

    return levels


# ---------------------------------------------------------------------------
# SNR-floor optimizer
# ---------------------------------------------------------------------------

def optimize_for_snr(
    profiles: list[TensorProfile],
    snr_floor: float,
    percentile: float = 5.0,
    refine: bool = True,
) -> Assignment:
    """Find smallest model with model_snr >= snr_floor.

    Phase 1: per-tensor floor assignment (each tensor individually meets floor).
    Phase 2: refinement — downgrade tensors that don't affect P5 (optional).

    Args:
        profiles:   Tensor profiles from Profiler.run().
        snr_floor:  Target model SNR in dB (P5 percentile).
        percentile: Percentile for model SNR (default 5.0).
        refine:     Whether to run the downgrade refinement pass.
    """
    levels = _per_tensor_floor_assignment(profiles, snr_floor)

    if refine:
        levels = _refine_downgrade(levels, profiles, snr_floor, percentile)

    return Assignment(
        levels=levels,
        total_bytes=compute_total_bytes(levels, profiles),
        model_snr_db=compute_model_snr(levels, profiles, percentile),
        model_snr_percentile=percentile,
    )


# ---------------------------------------------------------------------------
# Size-budget optimizer
# ---------------------------------------------------------------------------

def optimize_for_size(
    profiles: list[TensorProfile],
    size_budget_bytes: int,
    percentile: float = 5.0,
) -> Assignment:
    """Find highest model_snr that fits in size_budget_bytes.

    Strategy: start all 'optimize' tensors at F16, then greedily downgrade
    the tensor that loses the least SNR-per-byte-saved until budget is met.

    The SNR loss metric here is per-tensor SNR loss / byte saved — this
    works correctly for the size-constrained case because we're trying to
    preserve overall SNR while cutting bytes, not raise a P5.
    """
    # Start at F16 for optimizable tensors; policy minimum for fixed tensors
    levels: dict[str, str] = {}
    for p in profiles:
        if p.policy == "always_f16":
            levels[p.name] = "F16"
        elif p.policy == "always_q8_min":
            levels[p.name] = _cheapest_level(p.levels) if p.levels else "F16"
        else:
            levels[p.name] = "F16"

    optim_profiles = [p for p in profiles if p.policy == "optimize"]
    max_levels = max((len(p.levels) for p in optim_profiles), default=1)
    max_iterations = len(optim_profiles) * max_levels

    for _ in range(max_iterations):
        if compute_total_bytes(levels, profiles) <= size_budget_bytes:
            break

        best_name: str | None = None
        best_ratio: float = float("inf")   # minimize snr_loss / byte_save

        for p in optim_profiles:
            current = levels[p.name]
            pv = _next_cheaper(p.levels, current)
            if pv is None:
                continue

            snr_loss  = p.levels[current].snr_db          - p.levels[pv].snr_db
            byte_save = p.levels[current].estimated_bytes  - p.levels[pv].estimated_bytes

            if byte_save <= 0:
                continue

            ratio = snr_loss / byte_save
            if ratio < best_ratio:
                best_ratio = ratio
                best_name = p.name

        if best_name is None:
            break

        p_best = next(p for p in optim_profiles if p.name == best_name)
        levels[best_name] = _next_cheaper(p_best.levels, levels[best_name])

    return Assignment(
        levels=levels,
        total_bytes=compute_total_bytes(levels, profiles),
        model_snr_db=compute_model_snr(levels, profiles, percentile),
        model_snr_percentile=percentile,
    )
