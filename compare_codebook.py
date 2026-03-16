#!/usr/bin/env python3
"""
compare_codebook.py — Compare k-means VQ codebook compression to GGUF mixed-precision.

Profiles each tensor at CB3–CB12 bit levels using the same sampling approach as
snr_quant.py, then runs the mixed-precision optimizer at each SNR floor and prints
a side-by-side size comparison.

Usage:
    python compare_codebook.py /path/to/model [--force] [--sample-size N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from src.tensor_loader import ModelLoader
from src.snr_profiler import Profiler, LevelResult, TensorProfile
from src.optimizer import optimize_for_snr, compute_model_snr, compute_total_bytes, Assignment
from src.codebook_sim import CODEBOOK_LEVELS, CODEBOOK_LEVELS_BY_NAME
from src.reporter import MIXED_SNR_FLOORS, model_name


# ---------------------------------------------------------------------------
# Codebook profiler
# ---------------------------------------------------------------------------

_CACHE_NAME = "codebook_profile.json"


def _snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    signal = float(np.mean(original.astype(np.float32) ** 2))
    noise  = float(np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2))
    if noise < 1e-12:
        return 96.0
    return float(np.clip(10.0 * np.log10(signal / noise), 0.0, 96.0))


def _profile_codebook(
    model_path: Path,
    gguf_profiles: list[TensorProfile],
    sample_size: int,
    force: bool,
    verbose: bool,
) -> list[TensorProfile]:
    """Profile all tensors at each codebook bit level.

    Reuses the tensor metadata (policy, num_params, etc.) from the existing
    GGUF profiles so we don't need to re-classify tensors.
    """
    cache_path = model_path / _CACHE_NAME
    if cache_path.exists() and not force:
        if verbose:
            print(f"  Loading codebook profile cache from {cache_path.name}")
        with open(cache_path) as f:
            raw = json.load(f)
        return _deserialise(raw)

    # Build lookup from GGUF profiles
    gguf_by_name = {p.name: p for p in gguf_profiles}

    loader = ModelLoader(model_path)
    metas  = list(loader.tensors())
    n      = len(metas)

    if verbose:
        print(f"  Profiling {n} tensors at {len(CODEBOOK_LEVELS)} codebook levels...")
        print(f"  (sampling up to {sample_size:,} elements per tensor)")

    profiles: list[TensorProfile] = []

    for i, meta in enumerate(metas):
        if verbose:
            bar = f"[{i+1:4d}/{n}  {(i+1)/n*100:5.1f}%]"
            print(f"  {bar}  {meta.name:<60}", end="\r", flush=True)

        gp = gguf_by_name.get(meta.name)
        policy = gp.policy if gp else "optimize"

        # F16 is always available; add it
        levels: dict[str, LevelResult] = {
            "F16": LevelResult(
                snr_db=96.0,
                estimated_bytes=meta.num_params * 2,
            )
        }

        if policy == "always_f16":
            # No point profiling; it stays F16
            profiles.append(TensorProfile(
                name=meta.name, shape=meta.shape,
                num_params=meta.num_params,
                tensor_class=meta.tensor_class,
                policy=policy, levels=levels,
            ))
            continue

        # Load tensor data
        raw = loader.load_tensor(meta)
        data = raw.astype(np.float32)

        # Sample if large
        if data.size > sample_size:
            idx  = np.random.default_rng(42).choice(data.size, sample_size, replace=False)
            flat = data.flatten()[idx]
        else:
            flat = data.flatten()

        for cb in CODEBOOK_LEVELS:
            # For always_q8_min, skip levels below CB8 (treat like Q8_0 policy)
            if policy == "always_q8_min" and cb.bits < 8:
                continue

            dq = cb.simulate(flat)
            snr = _snr(flat, dq)
            levels[cb.name] = LevelResult(
                snr_db=snr,
                estimated_bytes=cb.estimated_bytes(meta.num_params),
            )

        profiles.append(TensorProfile(
            name=meta.name, shape=meta.shape,
            num_params=meta.num_params,
            tensor_class=meta.tensor_class,
            policy=policy, levels=levels,
        ))

    if verbose:
        print(" " * 80, end="\r")
        print(f"  Codebook profile saved to {cache_path.name}")

    # Serialise cache
    raw = [
        {
            "name": p.name, "shape": p.shape, "num_params": p.num_params,
            "tensor_class": p.tensor_class, "policy": p.policy,
            "levels": {k: {"snr_db": v.snr_db, "estimated_bytes": v.estimated_bytes}
                       for k, v in p.levels.items()},
        }
        for p in profiles
    ]
    with open(cache_path, "w") as f:
        json.dump(raw, f, indent=2)

    return profiles


def _deserialise(raw: list[dict]) -> list[TensorProfile]:
    profiles = []
    for r in raw:
        levels = {k: LevelResult(**v) for k, v in r["levels"].items()}
        profiles.append(TensorProfile(
            name=r["name"], shape=r["shape"], num_params=r["num_params"],
            tensor_class=r["tensor_class"], policy=r["policy"], levels=levels,
        ))
    return profiles


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

SNR_TARGETS = [45, 40, 30, 25, 20, 15]


def _optimise_all(profiles: list[TensorProfile]) -> dict[int, Assignment]:
    results = {}
    for floor in SNR_TARGETS:
        results[floor] = optimize_for_snr(profiles, float(floor))
    return results


def _format_row(label: str, gb: float, snr: float, mix: str) -> str:
    return f"  {label:<14}  {gb:>7.1f} GB  {snr:>6.1f} dB  {mix}"


def print_comparison(
    gguf_profiles: list[TensorProfile],
    cb_profiles: list[TensorProfile],
    model_path: Path,
) -> None:
    base_name = model_path.name

    print()
    print("=" * 88)
    print(f"  CODEBOOK vs GGUF COMPARISON: {base_name}")
    print("=" * 88)

    # F16 baseline
    f16_bytes = sum(p.num_params * 2 for p in gguf_profiles)
    print(f"  F16 baseline: {f16_bytes/1e9:.1f} GB")
    print()

    # Header
    print(f"  {'Target':<8}  {'GGUF mixed':^22}  {'Codebook mixed':^22}  {'Delta':>10}")
    print(f"  {'':^8}  {'Size':>8}  {'SNR':>8}  {'':^4}  {'Size':>8}  {'SNR':>8}  {'':^4}  {'(cb-gguf)':>10}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'':^4}  {'-'*8}  {'-'*8}  {'':^4}  {'-'*10}")

    print("  Running GGUF optimizer ...", end="\r", flush=True)
    gguf_results = _optimise_all(gguf_profiles)
    print("  Running codebook optimizer...", end="\r", flush=True)
    cb_results   = _optimise_all(cb_profiles)
    print(" " * 40, end="\r")

    for floor in SNR_TARGETS:
        gr = gguf_results[floor]
        cr = cb_results[floor]
        delta_gb  = (cr.total_bytes - gr.total_bytes) / 1e9
        delta_pct = (cr.total_bytes - gr.total_bytes) / gr.total_bytes * 100
        sign = "+" if delta_pct >= 0 else ""

        # Best level names for codebook
        cb_mix = _top_levels(cr, cb_profiles, n=3)
        gu_mix = _top_levels(gr, gguf_profiles, n=3)

        g_flag = "◀" if gr.model_snr_db >= floor - 0.1 else "†"
        c_flag = "◀" if cr.model_snr_db >= floor - 0.1 else "†"

        print(
            f"  {floor:>3} dB    "
            f"{gr.total_bytes/1e9:>7.1f} GB  {gr.model_snr_db:>6.1f} dB  {g_flag:<4}"
            f"  {cr.total_bytes/1e9:>7.1f} GB  {cr.model_snr_db:>6.1f} dB  {c_flag:<4}"
            f"  {sign}{delta_pct:+.0f}% ({delta_gb:+.1f} GB)"
        )

    print()
    print("  ◀ = floor achieved    † = floor unachievable (ceiling)")
    print()

    # Per-level codebook detail
    print("  Codebook level sizes (on this model):")
    print()
    print(f"  {'Level':<8}  {'bpw':>5}  {'Size':>8}  {'Tensor SNR (P5)':>16}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*16}")

    f16_assign = {p.name: "F16" for p in cb_profiles}
    for cb in CODEBOOK_LEVELS:
        # Build uniform codebook assignment
        assign: dict[str, str] = {}
        for p in cb_profiles:
            if p.policy == "always_f16":
                assign[p.name] = "F16"
            elif p.policy == "always_q8_min":
                # Use CB8 minimum
                lvl = cb.name if cb.bits >= 8 else "CB8"
                assign[p.name] = lvl if lvl in p.levels else "F16"
            else:
                assign[p.name] = cb.name if cb.name in p.levels else "F16"

        snr   = compute_model_snr(assign, cb_profiles)
        total = compute_total_bytes(assign, cb_profiles)
        bpw   = cb.bits_per_weight

        # Compare to nearest GGUF level
        print(f"  {cb.name:<8}  {bpw:>5.1f}  {total/1e9:>7.1f} GB  {snr:>10.1f} dB")

    print()
    print("  (Uniform assignment: all optimizable tensors at that CB level)")
    print()

    # Suggested names
    print("  Suggested codebook model names:")
    print()
    for floor in [45, 40, 30, 25]:
        cr = cb_results[floor]
        if cr.model_snr_db >= floor - 0.1:
            name = model_name(base_name, cr.total_bytes, cr.model_snr_db)
            print(f"    {name}")
    print()
    print("=" * 88)
    print()


def _top_levels(result: Assignment, profiles: list[TensorProfile], n: int = 3) -> str:
    """Summarise the top N level names by param count."""
    by_level: dict[str, int] = {}
    total = sum(p.num_params for p in profiles)
    for p in profiles:
        lvl = result.levels.get(p.name, "F16")
        by_level[lvl] = by_level.get(lvl, 0) + p.num_params
    sorted_levels = sorted(by_level.items(), key=lambda x: -x[1])[:n]
    parts = [f"{100*v/total:.0f}%{k}" for k, v in sorted_levels if v/total >= 0.02]
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare k-means codebook vs GGUF mixed-precision quantization."
    )
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--force", action="store_true",
                        help="Re-profile even if codebook_profile.json exists")
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()

    # Load GGUF profiles (must already exist)
    verbose = not args.quiet
    gguf_profiler = Profiler(model_path, sample_size=args.sample_size, verbose=verbose)
    gguf_profiles = gguf_profiler.run(force=False)  # use existing cache

    # Profile codebook levels
    if verbose:
        print()
        print(f"  Profiling codebook levels (CB3–CB12) ...")
    t0 = time.time()
    cb_profiles = _profile_codebook(
        model_path, gguf_profiles,
        sample_size=args.sample_size,
        force=args.force,
        verbose=verbose,
    )
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    print_comparison(gguf_profiles, cb_profiles, model_path)


if __name__ == "__main__":
    main()
