#!/usr/bin/env python3
"""
snr_quant.py — SNR-aware mixed-precision quantization planner.

Analyzes a model's weights and finds the optimal quantization assignment
for a given size budget or SNR quality floor.  Names output models by
actual measured size and signal quality rather than method name.

Usage:
  # Survey: show full size/SNR tradeoff curve
  python snr_quant.py /path/to/model --survey

  # SNR-constrained: find smallest model meeting a quality floor
  python snr_quant.py /path/to/model --db 30

  # Size-constrained: find best quality fitting in a budget
  python snr_quant.py /path/to/model --size 4G

  # Write plan + auto-named files to output directory (.txt for llama-quantize, .json summary)
  python snr_quant.py /path/to/model --db 30 --output-dir ./plans

  # Re-run analysis even if cache exists
  python snr_quant.py /path/to/model --survey --force
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Allow running from repo root or from quantization/ directly
sys.path.insert(0, str(Path(__file__).parent))

from src.snr_profiler import Profiler
from src.optimizer import optimize_for_snr, optimize_for_size
from src.reporter import print_survey, model_name


# ---------------------------------------------------------------------------
# Size parsing
# ---------------------------------------------------------------------------

def parse_size(s: str) -> int:
    """Parse a size string like '4G', '4.5GB', '4000M' into bytes."""
    s = s.strip().upper().rstrip("B")   # "4.5GB" → "4.5G"
    suffixes = {"G": 1_000_000_000, "M": 1_000_000, "K": 1_000, "T": 1_000_000_000_000}
    for suffix, multiplier in suffixes.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * multiplier)
    return int(s)


# ---------------------------------------------------------------------------
# Tensor name mapping: HuggingFace safetensors → GGUF
# ---------------------------------------------------------------------------
#
# llama-quantize uses GGUF tensor names (blk.N.attn_q.weight …).
# Our profiler reads from HF safetensors which use different conventions.
# Each entry below is (compiled_regex, {hf_suffix -> gguf_suffix}) where the
# regex captures (block_id, suffix) from the full HF tensor name.  Patterns
# are tried in order; the first match wins.
#
# MTP / visual-encoder tensors are silently dropped by convert_hf_to_gguf
# and do not appear in the GGUF, so they need no entry here.

# ── Block suffix maps (used by multiple architecture patterns) ────────────

# Standard per-block suffixes shared across LLaMA/Mistral/Qwen/Gemma/Phi-3 …
_LLAMA_BLOCK: dict[str, str] = {
    # Layer norms
    "input_layernorm.weight":                   "attn_norm.weight",
    "post_attention_layernorm.weight":          "ffn_norm.weight",
    "pre_feedforward_layernorm.weight":         "ffn_norm.weight",    # Gemma 2
    "post_feedforward_layernorm.weight":        "ffn_norm.weight",    # Gemma 2
    # Separate Q / K / V (most models)
    "self_attn.q_proj.weight":                  "attn_q.weight",
    "self_attn.k_proj.weight":                  "attn_k.weight",
    "self_attn.v_proj.weight":                  "attn_v.weight",
    # Attention output
    "self_attn.o_proj.weight":                  "attn_output.weight",
    "self_attn.out_proj.weight":                "attn_output.weight",  # LFM2
    "self_attn.dense.weight":                   "attn_output.weight",  # Persimmon
    "self_attn.linear_attn.weight":             "attn_output.weight",  # Deci
    # Fused QKV (Persimmon, InternVL, Phi-3 …)
    "self_attn.query_key_value.weight":         "attn_qkv.weight",
    "self_attn.qkv_proj.weight":                "attn_qkv.weight",
    # Attention variants using .attention. prefix (InternLM2)
    "attention.wq.weight":                      "attn_q.weight",
    "attention.wk.weight":                      "attn_k.weight",
    "attention.wv.weight":                      "attn_v.weight",
    "attention.wo.weight":                      "attn_output.weight",
    "attention.dense.weight":                   "attn_output.weight",
    # Standard MLP (LLaMA / Mistral / Qwen / Gemma / …)
    "mlp.gate_proj.weight":                     "ffn_gate.weight",
    "mlp.up_proj.weight":                       "ffn_up.weight",
    "mlp.down_proj.weight":                     "ffn_down.weight",
    # Fused gate+up (Phi-3, GLM-4)
    "mlp.gate_up_proj.weight":                  "ffn_up.weight",
    # llama-pth / InternLM2 feed_forward naming
    "feed_forward.w1.weight":                   "ffn_gate.weight",
    "feed_forward.w3.weight":                   "ffn_up.weight",
    "feed_forward.w2.weight":                   "ffn_down.weight",
    "feed_forward.gate_proj.weight":            "ffn_gate.weight",    # LLaMA4 / Jamba
    "feed_forward.up_proj.weight":              "ffn_up.weight",
    "feed_forward.down_proj.weight":            "ffn_down.weight",
    # MoE shared expert (Qwen2-MoE, DeepSeek-V2/V3, …)
    "mlp.shared_expert.gate_proj.weight":       "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight":         "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight":       "ffn_down_shexp.weight",
    # Mamba SSM via .mamba sub-module (Jamba, Falcon-H1, Granite-Hybrid)
    "mamba.in_proj.weight":                     "ssm_in.weight",
    "mamba.conv1d.weight":                      "ssm_conv1d.weight",
    "mamba.out_proj.weight":                    "ssm_out.weight",
    # Qwen3.5 linear attention (SSM-style)
    "linear_attn.in_proj_qkv.weight":           "attn_qkv.weight",
    "linear_attn.in_proj_z.weight":             "attn_gate.weight",
    "linear_attn.in_proj_a.weight":             "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight":             "ssm_beta.weight",
    "linear_attn.conv1d.weight":                "ssm_conv1d.weight",
    "linear_attn.out_proj.weight":              "ssm_out.weight",
    # Kimi Linear attention extras
    "self_attn.b_proj.weight":                  "ssm_beta.weight",
    "self_attn.q_conv1d.weight":                "ssm_conv1d_q.weight",
    "self_attn.k_conv1d.weight":                "ssm_conv1d_k.weight",
    "self_attn.v_conv1d.weight":                "ssm_conv1d_v.weight",
    "self_attn.f_a_proj.weight":                "ssm_f_a.weight",
    "self_attn.f_b_proj.weight":                "ssm_f_b.weight",
    "self_attn.g_a_proj.weight":                "ssm_g_a.weight",
    "self_attn.g_b_proj.weight":                "ssm_g_b.weight",
}

# GPT-NeoX block suffixes (gpt_neox.layers.N.*)
_GPTNEOX_BLOCK: dict[str, str] = {
    "attention.query_key_value.weight":  "attn_qkv.weight",
    "attention.dense.weight":            "attn_output.weight",
    "mlp.dense_h_to_4h.weight":          "ffn_up.weight",
    "mlp.dense_4h_to_h.weight":          "ffn_down.weight",
    "input_layernorm.weight":            "attn_norm.weight",
    "post_attention_layernorm.weight":   "ffn_norm.weight",
}

# Falcon / GPT-2 / GPT-J block suffixes (transformer.h.N.*)
_FALCON_BLOCK: dict[str, str] = {
    # Falcon (7B / 40B)
    "self_attention.query_key_value.weight": "attn_qkv.weight",
    "self_attention.dense.weight":           "attn_output.weight",
    "mlp.dense_h_to_4h.weight":             "ffn_up.weight",
    "mlp.dense_4h_to_h.weight":             "ffn_down.weight",
    # GPT-2 / GPT-J fused (c_attn / c_proj)
    "attn.c_attn.weight":                   "attn_qkv.weight",
    "attn.c_proj.weight":                   "attn_output.weight",
    "attn.q_proj.weight":                   "attn_q.weight",    # GPT-J
    "attn.k_proj.weight":                   "attn_k.weight",
    "attn.v_proj.weight":                   "attn_v.weight",
    "attn.out_proj.weight":                 "attn_output.weight",
    "mlp.c_fc.weight":                      "ffn_up.weight",
    "mlp.c_proj.weight":                    "ffn_down.weight",
    "mlp.fc_in.weight":                     "ffn_up.weight",    # GPT-J
    "mlp.fc_out.weight":                    "ffn_down.weight",
    "ln_1.weight":                          "attn_norm.weight",
    "ln_2.weight":                          "ffn_norm.weight",
}

# Bloom block suffixes (h.N.*)
_BLOOM_BLOCK: dict[str, str] = {
    "self_attention.query_key_value.weight": "attn_qkv.weight",
    "self_attention.dense.weight":           "attn_output.weight",
    "mlp.dense_h_to_4h.weight":             "ffn_up.weight",
    "mlp.dense_4h_to_h.weight":             "ffn_down.weight",
    "input_layernorm.weight":               "attn_norm.weight",
    "post_attention_layernorm.weight":      "ffn_norm.weight",
}

# MPT block suffixes (transformer.blocks.N.*)
_MPT_BLOCK: dict[str, str] = {
    "attn.Wqkv.weight":   "attn_qkv.weight",
    "attn.out_proj.weight": "attn_output.weight",
    "ffn.up_proj.weight":  "ffn_up.weight",
    "ffn.down_proj.weight": "ffn_down.weight",
    "norm_1.weight":       "attn_norm.weight",
    "norm_2.weight":       "ffn_norm.weight",
}

# Mamba-HF block suffixes (backbone.layers.N.mixer.*)
_MAMBA_BLOCK: dict[str, str] = {
    "in_proj.weight":   "ssm_in.weight",
    "conv1d.weight":    "ssm_conv1d.weight",
    "out_proj.weight":  "ssm_out.weight",
}

# ── Architecture-specific patterns ───────────────────────────────────────────
# Each entry: (compiled_regex, block_suffix_map)
# The regex must capture (block_id, suffix) in groups 1 and 2.
_BLOCK_PATTERNS: list[tuple[re.Pattern, dict[str, str]]] = [
    # LLaMA-style (the vast majority: LLaMA, Mistral, Qwen, Gemma, Phi-3,
    # InternLM2, DeepSeek, Falcon-Mamba, Jamba, Granite-Hybrid, Kimi, …)
    # Matches both model.layers.N.* and model.language_model.layers.N.* (VLMs)
    (re.compile(r"model(?:\.language_model)?\.layers\.(\d+)\.(.+)"), _LLAMA_BLOCK),
    # GPT-NeoX
    (re.compile(r"gpt_neox\.layers\.(\d+)\.(.+)"),                   _GPTNEOX_BLOCK),
    # Falcon / GPT-2 / GPT-J
    (re.compile(r"transformer\.h\.(\d+)\.(.+)"),                      _FALCON_BLOCK),
    # Bloom
    (re.compile(r"h\.(\d+)\.(.+)"),                                   _BLOOM_BLOCK),
    # MPT
    (re.compile(r"transformer\.blocks\.(\d+)\.(.+)"),                 _MPT_BLOCK),
    # Mamba-HF (backbone.layers.N.mixer.*)
    (re.compile(r"backbone\.layers\.(\d+)\.mixer\.(.+)"),             _MAMBA_BLOCK),
]

# ── Non-block (model-level) tensor names ─────────────────────────────────────
_HF_NONBLOCK_MAP: dict[str, str] = {
    # LLaMA / Mistral / Qwen / Gemma / Phi
    "model.embed_tokens.weight":                  "token_embd.weight",
    "model.norm.weight":                          "output_norm.weight",
    "lm_head.weight":                             "output.weight",
    # VLMs with nested language_model (Qwen2-VL, Qwen3.5-VL, InternVL, …)
    "model.language_model.embed_tokens.weight":   "token_embd.weight",
    "model.language_model.norm.weight":           "output_norm.weight",
    # GPT-NeoX
    "gpt_neox.embed_in.weight":                   "token_embd.weight",
    "embed_out.weight":                           "output.weight",
    "gpt_neox.final_layer_norm.weight":           "output_norm.weight",
    # Bloom
    "word_embeddings.weight":                     "token_embd.weight",
    "word_embeddings_layernorm.weight":           "token_embd_norm.weight",
    "ln_f.weight":                                "output_norm.weight",
    # Falcon
    "transformer.word_embeddings.weight":         "token_embd.weight",
    "transformer.ln_f.weight":                    "output_norm.weight",
    # MPT
    "transformer.wte.weight":                     "token_embd.weight",
    "transformer.norm_f.weight":                  "output_norm.weight",
    # GPT-2
    "wte.weight":                                 "token_embd.weight",
    "wpe.weight":                                 "token_types.weight",
    "ln_f.weight":                                "output_norm.weight",
    # Mamba-HF
    "backbone.embeddings.weight":                 "token_embd.weight",
    "backbone.norm_f.weight":                     "output_norm.weight",
}


def hf_to_gguf_name(hf_name: str) -> str | None:
    """Map a HuggingFace safetensors tensor name to its GGUF equivalent.

    Returns None if the name is not recognised (tensors not in GGUF — such as
    MTP heads and VLM visual encoders — are silently skipped by the plan writer).

    Covers: LLaMA/Mistral/Qwen/Gemma/Phi-3, Qwen-VL (nested language_model),
    Qwen3.5 linear-attention SSM, Kimi Linear, Jamba/Falcon-H1/Granite-Hybrid
    (Mamba via .mamba module), Mamba-HF, GPT-NeoX, Falcon, GPT-2/J, Bloom, MPT.
    """
    if hf_name in _HF_NONBLOCK_MAP:
        return _HF_NONBLOCK_MAP[hf_name]
    for pattern, block_map in _BLOCK_PATTERNS:
        m = pattern.match(hf_name)
        if m:
            n, rest = m.group(1), m.group(2)
            gguf_suffix = block_map.get(rest)
            if gguf_suffix:
                return f"blk.{n}.{gguf_suffix}"
            return None   # matched architecture but unknown suffix
    return None


# Map our quant level names to the ggml type strings that llama-quantize accepts.
# K-quant _S/_M/_L variants all share the same ggml base type; the distinction
# is handled by llama-quantize's built-in heuristics for un-overridden tensors.
_GGML_TYPE: dict[str, str] = {
    "F16":    "f16",
    "Q2_K":   "q2_K",
    "Q3_K_S": "q3_K",
    "Q3_K_M": "q3_K",
    "Q3_K_L": "q3_K",
    "Q4_0":   "q4_0",
    "Q4_K_S": "q4_K",
    "Q4_K_M": "q4_K",
    "Q5_0":   "q5_0",
    "Q5_K_S": "q5_K",
    "Q5_K_M": "q5_K",
    "Q6_K":   "q6_K",
    "Q8_0":   "q8_0",
}


# ---------------------------------------------------------------------------
# Plan writing
# ---------------------------------------------------------------------------

def _write_plan(output_path: Path, base_name: str, assignment) -> None:
    """Write a llama-quantize compatible tensor-type-file and a JSON summary.

    Produces two files side-by-side:
      <name>.txt  — plain-text ``tensor_name=ggml_type`` for ``--tensor-type-file``
      <name>.json — human-readable JSON with HF names, size, and SNR metadata
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip .json or .txt if the caller passed an explicit extension, otherwise
    # use the path as-is.  Avoid pathlib's with_suffix() — model names like
    # "Foo_8.3GB_30dB" contain dots that pathlib treats as extensions.
    name_str = output_path.name
    if name_str.endswith(".json") or name_str.endswith(".txt"):
        name_str = name_str.rsplit(".", 1)[0]
    base_path = output_path.parent / name_str
    txt_path  = output_path.parent / (name_str + ".txt")
    json_path = output_path.parent / (name_str + ".json")

    # Tensor name prefixes that convert_hf_to_gguf routes to a separate file
    # (MMPROJ vision encoder) or skips entirely (MTP heads).  Unmapped tensors
    # under these prefixes are expected and should not trigger a warning.
    _SEPARATE_FILE_PREFIXES = ("mtp.", "model.visual.", "visual.")

    unknown: list[str] = []
    with open(txt_path, "w") as f:
        for hf_name, quant_level in sorted(assignment.levels.items()):
            gguf_name  = hf_to_gguf_name(hf_name)
            ggml_type  = _GGML_TYPE.get(quant_level)
            if gguf_name is None or ggml_type is None:
                # Suppress noise: always-F16 tensors (stay F16 by default in
                # GGUF), and tensors that go to separate GGUF files.
                if quant_level != "F16" and not any(
                    hf_name.startswith(p) for p in _SEPARATE_FILE_PREFIXES
                ):
                    unknown.append(f"{hf_name} ({quant_level})")
                continue
            f.write(f"{gguf_name}={ggml_type}\n")

    with open(json_path, "w") as f:
        json.dump({
            "model":        base_name,
            "total_gb":     round(assignment.total_bytes / 1e9, 3),
            "model_snr_db": round(assignment.model_snr_db, 2),
            "tensors":      assignment.levels,
        }, f, indent=2)

    print(f"\nPlan written:")
    print(f"  Tensor-type file : {txt_path}")
    print(f"  JSON summary     : {json_path}")
    if unknown:
        print(f"\n  Warning: {len(unknown)} optimizable tensor(s) not mapped to GGUF names.")
        print(f"  These will use the fallback quant type. First entries:")
        for u in unknown[:5]:
            print(f"    {u}")
        if len(unknown) > 5:
            print(f"    … and {len(unknown)-5} more")
    print(f"\nTo quantize with llama-quantize:")
    print(f"  # Step 1: convert HF model to F16 GGUF")
    print(f"  python llama.cpp/convert_hf_to_gguf.py /path/to/{base_name} \\")
    print(f"      --outtype f16 --outfile {base_name}_f16.gguf")
    print(f"  # Step 2: mixed-precision quantize")
    print(f"  llama-quantize --tensor-type-file {txt_path} \\")
    print(f"      {base_name}_f16.gguf {base_path.name}.gguf Q4_K_M")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SNR-aware quantization planner — survey, SNR-floor, or size-budget modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_path", type=Path, help="Path to model directory")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--survey", action="store_true",
        help="Show full size/SNR tradeoff curve (no output files)"
    )
    mode.add_argument(
        "--db", type=float, metavar="SNR_DB",
        help="Find smallest model with model_snr >= SNR_DB"
    )
    mode.add_argument(
        "--size", type=str, metavar="SIZE",
        help="Find best model_snr fitting in SIZE (e.g. 4G, 4.5GB)"
    )

    parser.add_argument(
        "--output-dir", type=Path, default=None, metavar="DIR",
        help="Write plan files to DIR/<ModelName_XGB_YdB>.{txt,json} (auto-named)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write tensor assignment plan to this path (writes .txt and .json alongside)"
    )
    parser.add_argument(
        "--no-survey", action="store_true",
        help="Skip the survey table when using --db or --size"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-profile even if snr_profile.json cache exists"
    )
    parser.add_argument(
        "--sample-size", type=int, default=100_000,
        help="Max elements to sample per tensor for SNR estimation (default: 100000)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        print(f"Error: {model_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    # ---- Phase 1: profile ----
    profiler = Profiler(
        model_path=model_path,
        sample_size=args.sample_size,
        verbose=not args.quiet,
    )
    profiles = profiler.run(force=args.force)

    # ---- Phase 2: mode dispatch ----
    if args.survey:
        print_survey(profiles, model_path)
        return

    # --db or --size: show survey first unless suppressed
    if not args.no_survey and not args.quiet:
        print_survey(profiles, model_path)

    if args.db is not None:
        assignment = optimize_for_snr(profiles, args.db)
        gb = assignment.total_bytes / 1e9
        name = model_name(model_path.name, assignment.total_bytes, assignment.model_snr_db)
        print(f"\n{'='*60}")
        print(f"  Target SNR  : >= {args.db} dB")
        print(f"  Achieved SNR: {assignment.model_snr_db:.1f} dB")
        print(f"  Total size  : {gb:.2f} GB")
        print(f"  Model name  : {name}")
        print(f"{'='*60}")
        print()
        counts = assignment.level_counts()
        for lvl, cnt in counts.items():
            print(f"  {lvl:<10} {cnt:4d} tensors")

        _write_plan_if_requested(args, name, assignment)

    elif args.size is not None:
        budget = parse_size(args.size)
        assignment = optimize_for_size(profiles, budget)
        gb = assignment.total_bytes / 1e9
        name = model_name(model_path.name, assignment.total_bytes, assignment.model_snr_db)
        print(f"\n{'='*60}")
        print(f"  Size budget : {budget/1e9:.2f} GB")
        print(f"  Achieved    : {gb:.2f} GB")
        print(f"  Model SNR   : {assignment.model_snr_db:.1f} dB")
        print(f"  Model name  : {name}")
        print(f"{'='*60}")
        print()
        counts = assignment.level_counts()
        for lvl, cnt in counts.items():
            print(f"  {lvl:<10} {cnt:4d} tensors")

        _write_plan_if_requested(args, name, assignment)


def _write_plan_if_requested(args, name: str, assignment) -> None:
    """Write plan files to --output or --output-dir/<name> if requested."""
    if args.output:
        _write_plan(args.output, name, assignment)
    if args.output_dir:
        # Pass without extension; _write_plan writes both .txt and .json
        out_path = args.output_dir / name
        _write_plan(out_path, name, assignment)


if __name__ == "__main__":
    main()
