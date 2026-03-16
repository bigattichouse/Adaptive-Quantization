# Adaptive Quantization

SNR-aware mixed-precision GGUF quantization planner.

Instead of picking a quantization level by name (`Q4_K_M`, `Q5_K_M`, …) and hoping for the best, this tool **measures the actual signal-to-noise ratio** of every tensor at every quantization level and finds the optimal per-tensor assignment for a given quality floor or size budget.

Output models are named by their measured size and quality: `MyModel_8.3GB_30dB.gguf` tells you exactly what you're getting. The files are standard GGUFs loadable by any llama.cpp build — no modifications needed.

---

## How it works

1. **Profile** — sample each tensor at every GGUF quantization level and record the SNR in dB. Results are cached in `snr_profile.json` next to the model so profiling only runs once.

2. **Optimize** — given a quality floor (`--db 30`) or size budget (`--size 8G`), assign each tensor to the cheapest quantization level that meets the constraint.

3. **Plan** — write a `name=type` tensor-type file for `llama-quantize --tensor-type-file`, plus a human-readable JSON summary.

4. **Verify** — after quantization, check that the output GGUF matches the plan and measure the achieved SNR.

**Model SNR** is the P5 weighted SNR: tensors are sorted by per-tensor SNR; the model SNR is the value at the 5th percentile of cumulative parameter weight. Conservative — catches the worst 5% of parameters without being dominated by tiny outlier tensors.

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `safetensors`, `gguf`.

The `gguf` package is only needed for `verify_gguf.py`. If you cannot install it via pip, pass `--gguf-py /path/to/llama.cpp/gguf-py` to that script instead.

**llama.cpp tools** are needed to build the actual GGUF:
- `convert_hf_to_gguf.py` — converts HF model to F16 GGUF
- `llama-quantize` — applies the per-tensor plan

These are not installed by pip; you need a [llama.cpp](https://github.com/ggerganov/llama.cpp) checkout or pre-built binaries.

---

## Step-by-step workflow

### Step 1 — Survey the model

See the full size/SNR tradeoff table before committing to a target:

```bash
python snr_quant.py /path/to/MyModel --survey
```

This profiles every tensor (cached to `snr_profile.json` next to the model) and prints a table showing both uniform-quantization baselines and mixed-precision rows at every SNR floor:

```
========================================================================================
  SNR SURVEY: MyModel
========================================================================================
  775 tensors  |  9.65B params  |  19.3 GB F16 baseline
  275 always-F16  |  5 always-Q8_0-min  |  495 optimizable

  Type        Label             Size       SNR     vs nearest standard  Level mix (% of params)
  ----------  ------------  --------  --------  ----------------------  ----------------------------------------
  standard    F16              19.3GB    96.0dB                            100%F16
  standard    Q8_0             10.3GB    44.5dB                            99%Q8_0
  mixed       ≥30dB             8.3GB    30.3dB             -3% vs Q6_K ◀  5%Q2_K  73%Q6_K  21%Q8_0
  standard    Q6_K              8.5GB    32.3dB                            78%Q6_K  21%Q8_0
  standard    Q4_K_M            6.6GB    19.4dB                            78%Q4_K_M  21%Q8_0
  ...

  ◀ = mixed is smaller than nearest same-quality standard quant
  † = SNR floor exceeds model ceiling; best achievable shown
```

**Standard rows** show uniform quantization — every tensor at that level (same as `llama-quantize q4_k_m`).

**Mixed rows** show the optimizer's per-tensor assignment at each SNR floor. `◀` means the mixed result is smaller than the nearest equivalent-quality standard quant — this is where mixed-precision wins.

### Step 2 — Generate a plan

Pick your target and write the plan files:

```bash
# Smallest model achieving >= 30 dB SNR
python snr_quant.py /path/to/MyModel --db 30 --output-dir ./plans

# Best quality fitting in 8 GB
python snr_quant.py /path/to/MyModel --size 8G --output-dir ./plans

# Skip the survey when you already know your target
python snr_quant.py /path/to/MyModel --db 30 --no-survey --output-dir ./plans
```

This writes two files (auto-named by actual predicted size and SNR):

```
plans/MyModel_8.3GB_30dB.txt    ← for llama-quantize --tensor-type-file
plans/MyModel_8.3GB_30dB.json   ← human-readable summary
```

The `.txt` file contains one `gguf_tensor_name=ggml_type` entry per line:

```
blk.0.attn_q.weight=q6_K
blk.0.ffn_down.weight=q6_K
blk.0.ffn_gate.weight=q2_K
output.weight=q8_0
token_embd.weight=q8_0
...
```

### Step 3 — Build the GGUF

```bash
# Convert HF model to F16 GGUF
python /path/to/llama.cpp/convert_hf_to_gguf.py /path/to/MyModel \
    --outtype f16 --outfile MyModel_f16.gguf

# Apply the per-tensor plan
# Q4_K_M is the fallback for any tensor not in the plan
/path/to/llama-quantize \
    --tensor-type-file plans/MyModel_8.3GB_30dB.txt \
    MyModel_f16.gguf \
    MyModel_8.3GB_30dB.gguf \
    Q4_K_M
```

The resulting GGUF loads in any llama.cpp build without modification.

### Step 4 — Verify

After quantization, check that the output matches the plan and measure the actual achieved SNR:

```bash
python verify_gguf.py \
    MyModel_8.3GB_30dB.gguf \
    /path/to/MyModel \
    --plan plans/MyModel_8.3GB_30dB.txt \
    --predicted-snr 30.3 \
    --predicted-gb 8.27
```

If the `gguf` pip package is not installed, point to your llama.cpp checkout:

```bash
python verify_gguf.py \
    MyModel_8.3GB_30dB.gguf \
    /path/to/MyModel \
    --plan plans/MyModel_8.3GB_30dB.txt \
    --predicted-snr 30.3 --predicted-gb 8.27 \
    --gguf-py /path/to/llama.cpp/gguf-py
```

The report shows:
- Actual file size vs predicted
- Tensors that used a different type than planned (llama.cpp always stores some tensors — layernorms, 1D convolutions — as F32 regardless of the plan)
- Actual achieved model P5 SNR vs predicted

---

## Additional commands

```bash
# Re-profile from scratch (stale cache or new quant levels added)
python snr_quant.py /path/to/MyModel --survey --force

# Standard-only survey (skip the mixed-precision optimizer, faster)
python gguf_survey.py /path/to/MyModel --no-mixed

# Compare k-means VQ codebook vs GGUF (requires snr_profile.json)
python compare_codebook.py /path/to/MyModel
```

---

## Example output — Qwen3.5-9B (9.65B params, 19.3 GB baseline)

```
========================================================================================
  SNR SURVEY: Qwen3.5-9B
========================================================================================
  775 tensors  |  9.65B params  |  19.3 GB F16 baseline
  275 always-F16 (layernorms/biases)  |  5 always-Q8_0-min (embeddings)  |  495 optimizable

  Type        Label             Size       SNR     vs nearest standard  Level mix (% of params)
  ----------  ------------  --------  --------  ----------------------  ----------------------------------------
  standard    F16              19.3GB    96.0dB                            100%F16
  mixed       ≥55dB            17.4GB    45.1dB             -10% vs F16 †  21%Q8_0  79%F16
  mixed       ≥45dB            12.6GB    45.0dB            +22% vs Q8_0    5%Q2_K  65%Q8_0  30%F16
  standard    Q8_0             10.3GB    44.5dB                            99%Q8_0
  mixed       ≥40dB            10.0GB    42.6dB             -3% vs Q8_0 ◀  5%Q2_K  94%Q8_0
  standard    Q6_K              8.5GB    32.3dB                            78%Q6_K  21%Q8_0
  mixed       ≥30dB             8.3GB    30.3dB             -3% vs Q6_K ◀  5%Q2_K  73%Q6_K  21%Q8_0
  standard    Q5_0              7.5GB    26.0dB                            78%Q5_0  21%Q8_0
  standard    Q5_K_M            7.7GB    26.0dB                            78%Q5_K_M  21%Q8_0
  mixed       ≥25dB             7.3GB    25.3dB             -2% vs Q5_0 ◀  5%Q2_K  73%Q5_0  21%Q8_0
  mixed       ≥20dB             7.1GB    20.0dB             -6% vs Q5_0 ◀  5%Q2_K  18%Q4_K_S  55%Q5_0  21%Q8_0
  standard    Q4_K_S            6.4GB    19.4dB                            78%Q4_K_S  21%Q8_0
  standard    Q4_K_M            6.6GB    19.4dB                            78%Q4_K_M  21%Q8_0
  mixed       ≥15dB             6.3GB    17.5dB           -2% vs Q4_K_S ◀  5%Q2_K  73%Q4_K_S  21%Q8_0
  standard    Q3_K_M            5.5GB    12.0dB                            78%Q3_K_M  21%Q8_0
  standard    Q2_K              4.8GB     4.6dB                            78%Q2_K  21%Q8_0

  ◀ = mixed is smaller than nearest same-quality standard quant
  † = SNR floor exceeds model ceiling; best achievable shown

  Suggested model names (mixed, at common SNR targets):

    Qwen3.5-9B_12.6GB_45dB                    [65%Q8_0  30%F16]
    Qwen3.5-9B_10.0GB_42dB                    [94%Q8_0]
    Qwen3.5-9B_8.3GB_30dB                     [73%Q6_K  21%Q8_0]
    Qwen3.5-9B_7.3GB_25dB                     [73%Q5_0  21%Q8_0]

========================================================================================
```

**`†` rows** exceed the model's achievable ceiling (Qwen3.5-9B's visual encoder limits to ~45 dB).

**`+%` rows** (≥45dB) cost more than the nearest standard quant — keeping the visual encoder at F16 pushes the mixed result larger than Q8_0.

**`◀` rows** are the wins: mixed-precision smaller than the equivalent-quality uniform quant. The ≥30 dB row saves 3% vs Q6_K (8.3 GB vs 8.5 GB) while guaranteeing 30 dB model SNR.

---

## Tensor policies

| Policy | Tensors | Notes |
|---|---|---|
| `always_f16` | layernorms, biases, < 1024 params | Tiny; compression gains negligible |
| `always_q8_min` | embeddings, lm_head | Quality floor — used for every token |
| `optimize` | attention, MLP, SSM weights | Optimizer assigns freely |

---

## Architecture support

The HF → GGUF name mapper covers all major architectures:

| Family | Examples |
|---|---|
| LLaMA-style | LLaMA, Mistral, Qwen, Gemma, Phi-3, InternLM2, DeepSeek |
| VLM variants | Qwen2-VL, Qwen3.5-VL, InternVL (nested `language_model`) |
| SSM / hybrid | Qwen3.5 linear-attention, Jamba, Falcon-H1, Granite-Hybrid (Mamba) |
| Other | GPT-NeoX, Falcon, GPT-2/J, Bloom, MPT, Mamba-HF |

Tensors that `convert_hf_to_gguf.py` routes to a separate MMPROJ file (VLM visual encoders) or skips entirely (MTP heads) are automatically excluded from the plan.

> **Note on size predictions:** llama.cpp stores some tensor types (layernorms, 1D SSM convolutions) as F32 regardless of the plan. `verify_gguf.py` reports these discrepancies explicitly. Actual output is typically within a few percent of predicted.

---

## Cache and rebuild

Profiling is cached in `<model_dir>/snr_profile.json`. A 9B model takes a few minutes on CPU; all subsequent queries use the cache.

```bash
# Force re-profile (stale cache, new quant levels, etc.)
python snr_quant.py /path/to/model --survey --force
```

---

## Project layout

```
snr_quant.py          Main entry point — survey, --db, --size, plan writing
gguf_survey.py        Standalone GGUF survey with SNR/size winner tables
verify_gguf.py        Post-quantization verification (size, SNR, type diff)
compare_codebook.py   K-means VQ vs GGUF side-by-side comparison
requirements.txt
src/
  quant_sim.py        GGUF quantization level definitions and numpy simulation
  codebook_sim.py     K-means VQ simulation (CB3–CB12)
  tensor_loader.py    Safetensors reader (single-shard and multi-shard)
  snr_profiler.py     Per-tensor SNR measurement and JSON cache
  optimizer.py        Mixed-precision assignment algorithms (SNR-floor, size-budget)
  reporter.py         Survey table formatting and model naming
tests/
  test_quant_sim.py
  test_optimizer.py
```
