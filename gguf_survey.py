#!/usr/bin/env python3
"""
gguf_survey.py — SNR-aware GGUF quantization survey for informed model selection.

Profiles every tensor at every GGUF quant level and shows:

  1. Full table — standard (uniform) and mixed-precision rows sorted by SNR,
     so you can see exactly what quality you get at every size point.

  2. Best size at each SNR target — if you have a quality requirement,
     this shows the smallest model that meets it.

  3. Best quality within each size budget — if you have a disk/RAM budget,
     this shows the best model you can fit.

Mixed-precision rows use the optimizer to assign each tensor individually
to the cheapest level that meets the SNR floor, rather than applying one
level uniformly.  These are often 2–5% smaller than the nearest uniform
quant at the same quality.

Usage:
    python gguf_survey.py /path/to/model
    python gguf_survey.py /path/to/model --force      # re-profile from scratch
    python gguf_survey.py /path/to/model --no-mixed   # standard rows only

Results are cached in snr_profile.json next to the model — subsequent runs
are fast.  Re-run with --force if you suspect stale data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.snr_profiler import Profiler, TensorProfile
from src.optimizer import compute_total_bytes
from src.reporter import (
    _build_uniform_rows,
    _build_mixed_rows,
    _format_mix,
    _savings_vs_standard,
    print_snr_winners,
    print_size_winners,
    row_tag,
)


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

def print_gguf_survey(
    profiles: list[TensorProfile],
    model_path: Path,
    show_mixed: bool = True,
) -> None:
    base         = model_path.name
    n_total      = len(profiles)
    n_f16        = sum(1 for p in profiles if p.policy == "always_f16")
    n_q8         = sum(1 for p in profiles if p.policy == "always_q8_min")
    n_opt        = sum(1 for p in profiles if p.policy == "optimize")
    total_params = sum(p.num_params for p in profiles)
    f16_bytes    = sum(p.num_params * 2 for p in profiles)

    print()
    print("=" * 92)
    print(f"  GGUF SURVEY: {base}")
    print("=" * 92)
    print(
        f"  {n_total} tensors  |  {total_params/1e9:.2f}B params  |  "
        f"{f16_bytes/1e9:.1f} GB F16 baseline"
    )
    print(
        f"  {n_f16} always-F16 (layernorms/biases)  |  "
        f"{n_q8} always-Q8_0-min (embeddings)  |  "
        f"{n_opt} optimizable"
    )
    print()

    print("  Building standard rows ...")
    std_rows = _build_uniform_rows(profiles)
    for r in std_rows:
        r["source"] = "GGUF"

    if show_mixed:
        print("  Running mixed optimizer ...")
        mixed_rows = _build_mixed_rows(profiles)
        for r in mixed_rows:
            r["source"] = "GGUF"
    else:
        mixed_rows = []
    print()

    all_rows = std_rows + mixed_rows
    all_rows.sort(key=lambda r: (-r["snr_db"], 0 if r["kind"] == "standard" else 1))

    print(
        f"  {'Type':<10}  {'Label':<12}  {'Size':>8}  {'SNR':>8}  "
        f"{'vs nearest std':>20}  {'Level mix (% of params)'}"
    )
    print(
        f"  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*8}  "
        f"{'-'*20}  {'-'*40}"
    )

    for row in all_rows:
        gb   = row["total_bytes"] / 1e9
        snr  = row["snr_db"]
        mix  = _format_mix(row["mix_pct"])

        if row["kind"] == "standard":
            tag    = "GGUF std"
            vs_str = ""
            flag   = ""
        else:
            tag    = "GGUF mix"
            vs_str = _savings_vs_standard(row, std_rows) or ""
            flag   = ""
            if vs_str.startswith("-"):
                flag = " ◀"
            if not row.get("achievable", True):
                flag = " †"

        print(
            f"  {tag:<10}  {row['label']:<12}  {gb:>7.1f}GB  {snr:>6.1f}dB  "
            f"{vs_str:>20}{flag:<2}  {mix}"
        )

    print()
    print("  ◀ = mixed is smaller than nearest same-quality standard quant")
    print("  † = SNR floor exceeds model ceiling; best achievable shown")
    print()

    print_snr_winners(all_rows, base_name=base)
    print_size_winners(all_rows, f16_bytes, base_name=base)

    print("=" * 92)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SNR-aware GGUF survey — standard and mixed-precision tradeoff table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_path", type=Path, help="Path to model directory")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-profile even if snr_profile.json cache exists",
    )
    parser.add_argument(
        "--no-mixed", action="store_true",
        help="Show standard rows only, skip the mixed-precision optimizer",
    )
    parser.add_argument(
        "--sample-size", type=int, default=100_000,
        help="Max elements to sample per tensor for SNR estimation (default: 100000)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        print(f"Error: {model_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    profiler = Profiler(
        model_path=model_path,
        sample_size=args.sample_size,
        verbose=not args.quiet,
    )
    profiles = profiler.run(force=args.force)
    print_gguf_survey(profiles, model_path, show_mixed=not args.no_mixed)


if __name__ == "__main__":
    main()
