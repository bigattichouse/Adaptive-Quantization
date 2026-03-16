"""
reporter.py — Survey output formatting and model naming.

The survey prints a unified table with two types of rows interleaved by SNR:

  STANDARD rows: every optimizable tensor at the same level (uniform quant).
                 Shows the SNR you actually get from a plain llama-quantize run.

  MIXED rows:    optimal per-tensor assignment at a given SNR floor.
                 Shows size, achieved SNR, and the level mix as percentages
                 of total parameter count.

Seeing both in the same SNR-sorted table makes the tradeoff immediately clear:
"Q4_K_M gets you 19 dB at 6.6 GB; mixed targeting 20 dB costs 7.4 GB but
actually delivers 20 dB instead of the 19.4 dB Q4_K_M gives you."

Model naming: BaseModel_3.8GB_30dB
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .quant_sim import QUANT_LEVELS, LEVELS_BY_NAME
from .snr_profiler import TensorProfile
from .optimizer import (
    Assignment,
    optimize_for_snr,
    optimize_for_size,
    compute_model_snr,
    compute_total_bytes,
)


# SNR floors to run the mixed optimizer at
MIXED_SNR_FLOORS = [55, 50, 45, 40, 35, 30, 25, 20, 15]


# ---------------------------------------------------------------------------
# Model naming
# ---------------------------------------------------------------------------

def model_name(base_name: str, total_bytes: int, model_snr_db: float) -> str:
    """Generate a self-documenting model name.

    Example:  model_name("Qwen3.5-9B", 4_100_000_000, 30.2)
              → "Qwen3.5-9B_3.8GB_30dB"
    """
    gb = total_bytes / 1e9
    return f"{base_name}_{gb:.1f}GB_{int(model_snr_db)}dB"


# ---------------------------------------------------------------------------
# Level mix percentage breakdown
# ---------------------------------------------------------------------------

def _level_mix_pct(
    assignment: dict[str, str],
    profiles: list[TensorProfile],
) -> dict[str, float]:
    """Return {level_name: pct_of_total_params} for a given assignment.

    Percentages are by parameter count, not tensor count, so a few large
    MLP tensors dominate rather than hundreds of tiny layernorms.
    """
    total_params = sum(p.num_params for p in profiles)
    if total_params == 0:
        return {}

    by_level: dict[str, int] = {}
    for p in profiles:
        lvl = assignment.get(p.name, "F16")
        by_level[lvl] = by_level.get(lvl, 0) + p.num_params

    return {lvl: 100.0 * params / total_params
            for lvl, params in by_level.items()}


def _format_mix(pct: dict[str, float], threshold: float = 1.0) -> str:
    """Format mix percentages as a compact string.

    Only levels above threshold% are shown.
    Levels are ordered cheapest-first (QUANT_LEVELS order).
    """
    order = [q.name for q in QUANT_LEVELS]
    parts = []
    for lvl in order:
        p = pct.get(lvl, 0.0)
        if p >= threshold:
            parts.append(f"{p:.0f}%{lvl}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Uniform-quant rows
# ---------------------------------------------------------------------------

def _uniform_assignment(
    profiles: list[TensorProfile], level_name: str
) -> dict[str, str]:
    """Assign all optimizable tensors to level_name (respecting policy minimums)."""
    level_names = [q.name for q in QUANT_LEVELS]
    target_idx = level_names.index(level_name)
    q8_idx = level_names.index("Q8_0")

    result: dict[str, str] = {}
    for p in profiles:
        if p.policy == "always_f16":
            result[p.name] = "F16"
        elif p.policy == "always_q8_min":
            result[p.name] = level_name if target_idx >= q8_idx else "Q8_0"
        else:
            # Only assign if the level is actually in this tensor's profile
            # (new levels may not be in cached profiles)
            if level_name in p.levels:
                result[p.name] = level_name
            else:
                result[p.name] = "F16"
    return result


def _build_uniform_rows(profiles: list[TensorProfile]) -> list[dict]:
    """One row per QUANT_LEVEL showing uniform-quant size and SNR."""
    rows = []
    for level in QUANT_LEVELS:
        assignment = _uniform_assignment(profiles, level.name)
        snr = compute_model_snr(assignment, profiles)
        total = compute_total_bytes(assignment, profiles)
        pct = _level_mix_pct(assignment, profiles)
        rows.append({
            "kind": "standard",
            "label": level.name,
            "total_bytes": total,
            "snr_db": snr,
            "mix_pct": pct,
        })
    return rows


# ---------------------------------------------------------------------------
# Mixed-precision rows
# ---------------------------------------------------------------------------

def _build_mixed_rows(profiles: list[TensorProfile]) -> list[dict]:
    """One row per SNR floor showing optimally-mixed size and level breakdown.

    Deduplicates rows where the optimizer produced the same result for multiple
    floors (happens when the model hits a quality ceiling).  Keeps the highest
    (most ambitious) floor for each unique (bytes, snr) pair so the table shows
    the tightest constraint that still achieves that result.

    Rows where the achieved SNR is below the requested floor have
    ``achievable=False`` — the model's ceiling prevents reaching the target.
    """
    rows = []
    for floor in MIXED_SNR_FLOORS:
        assignment = optimize_for_snr(profiles, float(floor))
        pct = _level_mix_pct(assignment.levels, profiles)
        rows.append({
            "kind": "mixed",
            "label": f"≥{floor}dB",
            "floor": floor,
            "total_bytes": assignment.total_bytes,
            "snr_db": assignment.model_snr_db,
            "mix_pct": pct,
            "achievable": assignment.model_snr_db >= floor - 0.1,
        })

    # Deduplicate: if two floors yield identical (bytes, snr), keep the highest
    # floor (most informative — shows the ceiling clearly).
    seen: dict[tuple, dict] = {}
    for row in rows:
        key = (row["total_bytes"], round(row["snr_db"], 1))
        if key not in seen or row["floor"] > seen[key]["floor"]:
            seen[key] = row
    deduped = sorted(seen.values(), key=lambda r: -r["floor"])
    return deduped


# ---------------------------------------------------------------------------
# Unified table
# ---------------------------------------------------------------------------

def _merge_and_sort(
    uniform_rows: list[dict],
    mixed_rows: list[dict],
) -> list[dict]:
    """Merge standard and mixed rows, sorted by achieved SNR descending."""
    all_rows = uniform_rows + mixed_rows
    # Sort descending by SNR; break ties: standard before mixed
    all_rows.sort(key=lambda r: (-r["snr_db"], 0 if r["kind"] == "standard" else 1))
    return all_rows


def _savings_vs_standard(
    row: dict,
    uniform_rows: list[dict],
) -> Optional[str]:
    """For a mixed row, find the nearest standard row by SNR and compute savings."""
    target_snr = row["snr_db"]
    # Find cheapest standard row that achieves >= target_snr
    candidates = [r for r in uniform_rows if r["snr_db"] >= target_snr - 0.5]
    if not candidates:
        return None
    ref = min(candidates, key=lambda r: r["total_bytes"])
    if ref["total_bytes"] == 0:
        return None
    delta_pct = (row["total_bytes"] - ref["total_bytes"]) / ref["total_bytes"] * 100
    sign = "+" if delta_pct > 0 else ""
    return f"{sign}{delta_pct:.0f}% vs {ref['label']}"


# ---------------------------------------------------------------------------
# Main survey printer
# ---------------------------------------------------------------------------

def print_survey(
    profiles: list[TensorProfile],
    model_path: Path,
) -> None:
    """Print the full survey report to stdout."""
    base_name = model_path.name

    n_total = len(profiles)
    n_fixed_f16 = sum(1 for p in profiles if p.policy == "always_f16")
    n_fixed_q8  = sum(1 for p in profiles if p.policy == "always_q8_min")
    n_optimize  = sum(1 for p in profiles if p.policy == "optimize")
    total_params = sum(p.num_params for p in profiles)

    f16_assignment = {p.name: "F16" for p in profiles}
    baseline_bytes = compute_total_bytes(f16_assignment, profiles)

    print()
    print("=" * 88)
    print(f"  SNR SURVEY: {base_name}")
    print("=" * 88)
    print(
        f"  {n_total} tensors  |  {total_params/1e9:.2f}B params  |  "
        f"{baseline_bytes/1e9:.1f} GB F16 baseline"
    )
    print(
        f"  {n_fixed_f16} always-F16 (layernorms/biases)  |  "
        f"{n_fixed_q8} always-Q8_0-min (embeddings)  |  "
        f"{n_optimize} optimizable"
    )
    print()

    # Build rows
    print("  Building uniform reference rows...", end="\r", flush=True)
    uniform_rows = _build_uniform_rows(profiles)
    print("  Running mixed optimizer at each SNR floor...", end="\r", flush=True)
    mixed_rows = _build_mixed_rows(profiles)
    all_rows = _merge_and_sort(uniform_rows, mixed_rows)
    print(" " * 50, end="\r")  # clear progress line

    # ---- Unified table header ----
    print(
        f"  {'Type':<10}  {'Label':<12}  {'Size':>8}  {'SNR':>8}  "
        f"{'vs nearest standard':>22}  {'Level mix (% of params)'}"
    )
    print(
        f"  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*8}  "
        f"{'-'*22}  {'-'*40}"
    )

    for row in all_rows:
        gb = row["total_bytes"] / 1e9
        snr = row["snr_db"]
        mix_str = _format_mix(row["mix_pct"])

        if row["kind"] == "standard":
            tag = "standard"
            vs_str = ""
            flag = ""
        else:
            tag = "mixed"
            vs_str = _savings_vs_standard(row, uniform_rows) or ""
            # ◀ = mixed saves space vs same-quality standard
            # † = requested floor exceeds model's achievable ceiling
            flag = ""
            if vs_str.startswith("-"):
                flag = " ◀"
            if not row.get("achievable", True):
                flag = " †"

        print(
            f"  {tag:<10}  {row['label']:<12}  {gb:>7.1f}GB  {snr:>6.1f}dB  "
            f"{vs_str:>22}{flag:<2}  {mix_str}"
        )

    print()
    print("  ◀ = mixed is smaller than nearest same-quality standard quant")
    print("  † = SNR floor exceeds model ceiling; best achievable shown")
    print()

    # ---- Suggested names ----
    print("  Suggested model names (mixed, at common SNR targets):")
    print()
    # Use the deduplicated mixed_rows; show achievable rows near common targets
    shown_floors = {45, 40, 35, 30, 25}
    for row in mixed_rows:
        if row["floor"] in shown_floors and row.get("achievable", True):
            name = model_name(base_name, row["total_bytes"], row["snr_db"])
            mix_str = _format_mix(row["mix_pct"], threshold=5.0)
            print(f"    {name:<40}  [{mix_str}]")
    print()
    print("=" * 88)
    print()


# ---------------------------------------------------------------------------
# Winner tables (shared by gguf_survey.py and unified_survey.py)
# ---------------------------------------------------------------------------

def row_tag(row: dict) -> str:
    """'GGUF std', 'CB mix', etc. — falls back gracefully when source absent."""
    src  = row.get("source", "")
    kind = "mix" if row["kind"] == "mixed" else "std"
    return f"{src} {kind}".strip()


def _suggested_name(base_name: Optional[str], gb: float, snr_db: float) -> str:
    """Short suggested output name: ModelName_8.3GB_30dB."""
    snr_int = int(snr_db)
    if base_name:
        return f"{base_name}_{gb:.1f}GB_{snr_int}dB"
    return f"{gb:.1f}GB_{snr_int}dB"


def print_snr_winners(
    all_rows: list[dict], base_name: Optional[str] = None
) -> None:
    """Smallest option at each 5 dB SNR step, deduped when ceiling repeats."""
    print("  Best size at each SNR target:")
    print()
    print(f"  {'Target':>8}  {'Type':<10}  {'Label':<12}  {'Size':>8}  {'Achieved':>9}  Suggested name")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*9}  {'-'*30}")

    # Collect winners at each 5 dB step, highest first
    entries = []
    prev_key = None
    for target in range(95, 0, -5):
        candidates = [r for r in all_rows if r["snr_db"] >= target]
        if not candidates:
            continue
        winner = min(candidates, key=lambda r: r["total_bytes"])
        key = (winner.get("source", ""), winner["label"], winner["total_bytes"])
        if key == prev_key:
            continue
        prev_key = key
        entries.append((target, winner))

    for target, winner in entries:
        gb   = winner["total_bytes"] / 1e9
        name = _suggested_name(base_name, gb, winner["snr_db"])
        print(
            f"  {target:>5} dB  {row_tag(winner):<10}  {winner['label']:<12}  "
            f"{gb:>7.1f}GB  {winner['snr_db']:>7.1f}dB  {name}"
        )
    print()


def print_size_winners(
    all_rows: list[dict], f16_bytes: int, base_name: Optional[str] = None
) -> None:
    """Best quality at each 0.5 GB size step, deduped when winner unchanged."""
    min_gb = min(r["total_bytes"] for r in all_rows) / 1e9
    max_gb = f16_bytes / 1e9
    start  = int(min_gb * 2) + 1
    end    = int(max_gb * 2) + 1

    print("  Best quality within each size budget:")
    print()
    print(f"  {'Budget':>8}  {'Type':<10}  {'Label':<12}  {'Achieved':>9}  {'Actual size':>11}  Suggested name")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*9}  {'-'*11}  {'-'*30}")

    # Collect winners at each 0.5 GB step, largest budget first
    entries = []
    prev_key = None
    for step in range(end - 1, start - 1, -1):
        budget_gb    = step * 0.5
        budget_bytes = int(budget_gb * 1e9)
        candidates   = [r for r in all_rows if r["total_bytes"] <= budget_bytes]
        if not candidates:
            continue
        winner = max(candidates, key=lambda r: (r["snr_db"], -r["total_bytes"]))
        key = (winner.get("source", ""), winner["label"])
        if key == prev_key:
            continue
        prev_key = key
        entries.append((budget_gb, winner))

    for budget_gb, winner in entries:
        gb   = winner["total_bytes"] / 1e9
        name = _suggested_name(base_name, gb, winner["snr_db"])
        print(
            f"  {budget_gb:>5.1f} GB  {row_tag(winner):<10}  {winner['label']:<12}  "
            f"{winner['snr_db']:>7.1f}dB  {gb:>9.1f}GB  {name}"
        )
    print()
