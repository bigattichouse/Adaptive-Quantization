#!/usr/bin/env python3
"""
verify_gguf.py — Compare predicted vs actual SNR and tensor types for a quantized GGUF.

Loads the output GGUF, the plan file, and the model's snr_profile.json, then reports:
  - Actual file size vs predicted
  - Tensors that fell back to a different type than planned
  - Actual achieved model SNR (P5 weighted) vs predicted

Usage:
    python quantization/verify_gguf.py <gguf_path> <model_path> [--plan plan.txt]

If --plan is omitted, looks for a .txt file matching the GGUF basename in the same dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import hf_to_gguf_name from snr_quant.py (in the same directory)
from snr_quant import hf_to_gguf_name

# gguf package: try pip-installed first, then fall back to a llama.cpp checkout.
# Install with:  pip install gguf
# Or point --gguf-py to the gguf-py/ directory inside a llama.cpp checkout.
try:
    from gguf import GGUFReader
except ImportError:
    GGUFReader = None  # deferred; resolved after argument parsing


# Map GGUF tensor_type.name → the profile level key we'd use for SNR lookup.
# K-quant _S/_M variants share the same underlying type in the GGUF.
_GGUF_TYPE_TO_PROFILE_LEVEL: dict[str, str] = {
    "F16":   "F16",
    "BF16":  "F16",   # treat as F16 for SNR purposes
    "F32":   "F16",
    "Q8_0":  "Q8_0",
    "Q6_K":  "Q6_K",
    "Q5_K":  "Q5_K_M",
    "Q4_K":  "Q4_K_M",
    "Q3_K":  "Q3_K_M",
    "Q5_0":  "Q5_0",
    "Q4_0":  "Q4_0",
    "Q2_K":  "Q2_K",
    "Q1_K":  "Q2_K",  # rare; Q2_K as floor
}

# Normalise ggml type string for plan-file comparison
# plan file uses lowercase: "q6_K", "f16", etc.
_NORM = str.lower


def _plan_type_to_gguf_type(plan_type: str) -> str:
    """e.g. 'q6_K' → 'Q6_K', 'f16' → 'F16'."""
    return plan_type.upper().replace("_K_S", "_K").replace("_K_M", "_K").replace("_K_L", "_K")


def load_plan(plan_path: Path) -> dict[str, str]:
    """Load a plan .txt file → {gguf_tensor_name: ggml_type_name (uppercase)}."""
    result: dict[str, str] = {}
    with open(plan_path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            name, typ = line.split("=", 1)
            result[name.strip()] = _plan_type_to_gguf_type(typ.strip())
    return result


def load_snr_profile(model_path: Path) -> dict:
    """Load snr_profile.json → list of tensor dicts."""
    profile_path = model_path / "snr_profile.json"
    if not profile_path.exists():
        return {}
    with open(profile_path) as f:
        raw = json.load(f)
    return {t["name"]: t for t in raw}


def build_gguf_to_hf(hf_profiles: dict) -> dict[str, str]:
    """Build reverse map: gguf_name → hf_name from the SNR profile."""
    result: dict[str, str] = {}
    for hf_name in hf_profiles:
        gguf = hf_to_gguf_name(hf_name)
        if gguf:
            result[gguf] = hf_name
    return result


def compute_p5_snr(pairs: list[tuple[float, int]]) -> float:
    """P5 weighted SNR: sort by SNR, find 5th-percentile of cumulative params."""
    if not pairs:
        return 0.0
    pairs.sort(key=lambda x: x[0])
    total_w = sum(w for _, w in pairs)
    target = total_w * 5.0 / 100.0
    cumulative = 0.0
    for snr, w in pairs:
        cumulative += w
        if cumulative >= target:
            return snr
    return pairs[-1][0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify predicted vs actual SNR and tensor types for a quantized GGUF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("gguf_path", type=Path, help="Path to the quantized GGUF file")
    parser.add_argument("model_path", type=Path, help="Path to original HF model directory")
    parser.add_argument("--plan", type=Path, default=None,
                        help="Plan .txt file (default: <gguf_basename>.txt in same dir)")
    parser.add_argument("--predicted-snr", type=float, default=None,
                        help="Predicted model SNR from snr_quant (for comparison)")
    parser.add_argument("--predicted-gb", type=float, default=None,
                        help="Predicted size in GB from snr_quant (for comparison)")
    parser.add_argument(
        "--gguf-py", type=Path, default=None, metavar="DIR",
        help="Path to the gguf-py/ directory inside a llama.cpp checkout "
             "(only needed if the 'gguf' pip package is not installed). "
             "Example: --gguf-py /path/to/llama.cpp/gguf-py",
    )
    args = parser.parse_args()

    # Resolve gguf import
    global GGUFReader
    if GGUFReader is None:
        gguf_py_path = args.gguf_py
        if gguf_py_path is None:
            print(
                "Error: the 'gguf' Python package is not installed and --gguf-py was not given.\n"
                "Install it with:  pip install gguf\n"
                "Or pass:          --gguf-py /path/to/llama.cpp/gguf-py",
                file=sys.stderr,
            )
            sys.exit(1)
        sys.path.insert(0, str(gguf_py_path.expanduser().resolve()))
        try:
            from gguf import GGUFReader as _R
            GGUFReader = _R
        except ImportError as e:
            print(f"Error: could not import GGUFReader from {gguf_py_path}: {e}", file=sys.stderr)
            sys.exit(1)

    gguf_path  = args.gguf_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()

    if not gguf_path.exists():
        print(f"Error: GGUF not found: {gguf_path}", file=sys.stderr)
        sys.exit(1)

    # Auto-find plan file
    plan_path = args.plan
    if plan_path is None:
        candidate = gguf_path.with_suffix(".txt")
        if not candidate.exists():
            # Try same name but in current dir's plans/
            stem = gguf_path.stem
            candidate = Path("plans") / (stem + ".txt")
        plan_path = candidate if candidate.exists() else None

    # ── Load data ────────────────────────────────────────────────────────────

    print()
    print("=" * 80)
    print(f"  GGUF VERIFICATION: {gguf_path.name}")
    print("=" * 80)
    print()

    # GGUF tensor types
    print("  Reading GGUF tensor types ...", end="\r", flush=True)
    reader = GGUFReader(str(gguf_path))
    gguf_tensors: dict[str, str] = {
        t.name: t.tensor_type.name for t in reader.tensors
    }
    actual_bytes = gguf_path.stat().st_size
    print(f"  GGUF: {len(gguf_tensors)} tensors, {actual_bytes/1e9:.3f} GB on disk")

    if args.predicted_gb is not None:
        delta_pct = (actual_bytes/1e9 - args.predicted_gb) / args.predicted_gb * 100
        sign = "+" if delta_pct > 0 else ""
        print(f"  Predicted size : {args.predicted_gb:.3f} GB  ({sign}{delta_pct:.1f}% delta)")
    print()

    # Plan file
    plan: dict[str, str] = {}
    if plan_path and plan_path.exists():
        plan = load_plan(plan_path)
        print(f"  Plan file: {plan_path}  ({len(plan)} entries)")
    else:
        print("  No plan file found — skipping type comparison.")
    print()

    # SNR profile
    hf_profiles = load_snr_profile(model_path)
    gguf_to_hf  = build_gguf_to_hf(hf_profiles)
    print(f"  SNR profile: {len(hf_profiles)} tensors loaded")
    print(f"  Name map:    {len(gguf_to_hf)} GGUF names resolved to HF names")
    print()

    # ── Compare plan vs actual ────────────────────────────────────────────────

    if plan:
        mismatches: list[tuple[str, str, str]] = []   # (gguf_name, planned, actual)
        in_plan_not_gguf: list[str] = []
        for gguf_name, planned_type in sorted(plan.items()):
            actual_type = gguf_tensors.get(gguf_name)
            if actual_type is None:
                in_plan_not_gguf.append(gguf_name)
            elif actual_type != planned_type:
                mismatches.append((gguf_name, planned_type, actual_type))

        fallbacks: list[tuple[str, str]] = []   # (gguf_name, actual_type)
        for gguf_name, actual_type in sorted(gguf_tensors.items()):
            if gguf_name not in plan:
                fallbacks.append((gguf_name, actual_type))

        print(f"  Plan coverage:")
        print(f"    Tensors in plan    : {len(plan)}")
        print(f"    Matched (type OK)  : {len(plan) - len(mismatches) - len(in_plan_not_gguf)}")
        if mismatches:
            print(f"    Type mismatches    : {len(mismatches)}  ← planned type != actual type")
        if in_plan_not_gguf:
            print(f"    In plan, not GGUF  : {len(in_plan_not_gguf)}  ← tensors missing from output")
            for name in sorted(in_plan_not_gguf)[:20]:
                print(f"      {name}")
            if len(in_plan_not_gguf) > 20:
                print(f"      … and {len(in_plan_not_gguf)-20} more")
        print(f"    Not in plan (fallback): {len(fallbacks)}  ← llama-quantize chose type")
        print()

        if mismatches:
            print("  Type mismatches (planned → actual):")
            for name, planned, actual in mismatches[:20]:
                print(f"    {name:<45} {planned} → {actual}")
            if len(mismatches) > 20:
                print(f"    … and {len(mismatches)-20} more")
            print()

        if fallbacks:
            # Group fallbacks by type
            by_type: dict[str, int] = {}
            for _, t in fallbacks:
                by_type[t] = by_type.get(t, 0) + 1
            print("  Fallback tensors (not in plan — used llama-quantize default):")
            for t, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
                print(f"    {t:<10} {cnt} tensors")
            print()

    # ── Actual vs predicted SNR ───────────────────────────────────────────────

    snr_pairs: list[tuple[float, int]] = []
    unmapped_params = 0
    unmapped_count  = 0
    fallback_snr_impact: list[tuple[str, str, str, float, float, int]] = []

    for gguf_name, actual_type in gguf_tensors.items():
        hf_name = gguf_to_hf.get(gguf_name)
        if hf_name is None:
            unmapped_count += 1
            continue
        profile = hf_profiles.get(hf_name)
        if profile is None:
            unmapped_count += 1
            continue

        num_params   = profile.get("num_params", 0)
        levels_data  = profile.get("levels", {})
        profile_key  = _GGUF_TYPE_TO_PROFILE_LEVEL.get(actual_type)

        if profile_key and profile_key in levels_data:
            snr = levels_data[profile_key]["snr_db"]
            snr_pairs.append((snr, num_params))

            # Check if this tensor used a fallback (different from plan)
            planned_type = plan.get(gguf_name)
            if planned_type and planned_type != actual_type:
                planned_key = _GGUF_TYPE_TO_PROFILE_LEVEL.get(planned_type)
                if planned_key and planned_key in levels_data:
                    planned_snr = levels_data[planned_key]["snr_db"]
                    fallback_snr_impact.append(
                        (gguf_name, planned_type, actual_type,
                         planned_snr, snr, num_params)
                    )
        else:
            unmapped_params += num_params

    actual_model_snr = compute_p5_snr(snr_pairs)

    print(f"  SNR computation:")
    print(f"    Tensors with SNR data : {len(snr_pairs)}")
    if unmapped_count:
        print(f"    Tensors without profile data: {unmapped_count}  (skipped in P5)")
    print()
    print(f"  ┌─────────────────────────────────────────┐")
    if args.predicted_snr is not None:
        delta = actual_model_snr - args.predicted_snr
        sign  = "+" if delta >= 0 else ""
        print(f"  │  Predicted model SNR : {args.predicted_snr:>6.1f} dB              │")
        print(f"  │  Actual model SNR    : {actual_model_snr:>6.1f} dB  ({sign}{delta:.2f} dB delta) │")
    else:
        print(f"  │  Actual model SNR    : {actual_model_snr:>6.1f} dB              │")
    print(f"  └─────────────────────────────────────────┘")
    print()

    if fallback_snr_impact:
        print(f"  Tensors with SNR impact from type changes:")
        print(f"  {'Tensor':<42} {'Planned':>8}  {'Actual':>8}  {'Plan SNR':>9}  {'Act SNR':>9}  {'Params':>10}")
        print(f"  {'-'*42}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*10}")
        for name, planned, actual, psnr, asnr, params in sorted(
            fallback_snr_impact, key=lambda x: abs(x[4] - x[3]), reverse=True
        )[:20]:
            print(f"  {name:<42}  {planned:>8}  {actual:>8}  {psnr:>7.1f}dB  {asnr:>7.1f}dB  {params:>10,}")
        print()

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
