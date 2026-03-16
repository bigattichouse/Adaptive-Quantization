"""
snr_profiler.py — Per-tensor SNR measurement at each quantization level.

For each tensor the profiler:
  1. Loads raw weights from safetensors
  2. Simulates quantization at every QUANT_LEVEL
  3. Computes SNR (dB) and estimated byte size
  4. Returns a TensorProfile

Results are cached to snr_profile.json in the model directory so the
expensive analysis runs only once.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

from .quant_sim import QUANT_LEVELS, QuantLevel, LEVELS_BY_NAME
from .tensor_loader import ModelLoader, TensorMeta


# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

# Tensor classes that are always stored at F16 (never quantized).
# These are tiny and precision-critical.
ALWAYS_F16_CLASSES = frozenset({"layernorm", "other"})

# Tensor classes with a minimum quality floor of Q8_0.
ALWAYS_Q8_MIN_CLASSES = frozenset({"embedding"})

# Tensors smaller than this are always F16 regardless of class.
MIN_PARAMS_FOR_QUANT = 1024


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LevelResult:
    """SNR and byte estimate for one tensor at one quant level."""
    snr_db: float
    estimated_bytes: int


@dataclass
class TensorProfile:
    """Full quantization profile for one tensor."""
    name: str
    shape: list[int]
    num_params: int
    tensor_class: str
    policy: str            # "always_f16", "always_q8_min", "optimize"
    levels: dict[str, LevelResult] = field(default_factory=dict)

    def snr_at(self, level_name: str) -> float:
        return self.levels[level_name].snr_db

    def bytes_at(self, level_name: str) -> int:
        return self.levels[level_name].estimated_bytes

    def best_level_meeting(self, snr_floor: float) -> str:
        """Return cheapest level with SNR ≥ snr_floor, or 'F16' if none qualify."""
        for level in QUANT_LEVELS:
            if level.name in self.levels and self.levels[level.name].snr_db >= snr_floor:
                return level.name
        return "F16"

    def min_eligible_level(self) -> str:
        """Lowest level this tensor is eligible for given its policy."""
        if self.policy == "always_f16":
            return "F16"
        if self.policy == "always_q8_min":
            return "Q8_0"
        return "Q2_K"


# ---------------------------------------------------------------------------
# SNR computation
# ---------------------------------------------------------------------------

def _compute_snr(original: np.ndarray, dequantized: np.ndarray) -> float:
    """Signal-to-noise ratio in dB (power SNR).

    Uses 10*log10(signal_power / noise_power), consistent with the rest of
    this codebase.  Clamped to [0, 96] dB.
    """
    orig_f = original.astype(np.float32).flatten()
    deq_f = dequantized.astype(np.float32).flatten()

    signal_power = float(np.mean(orig_f ** 2))
    noise_power = float(np.mean((orig_f - deq_f) ** 2))

    if noise_power < 1e-12:
        return 96.0
    if signal_power < 1e-12:
        return 0.0

    return float(np.clip(10.0 * np.log10(signal_power / noise_power), 0.0, 96.0))


# ---------------------------------------------------------------------------
# Policy assignment
# ---------------------------------------------------------------------------

def _assign_policy(meta: TensorMeta) -> str:
    if meta.num_params < MIN_PARAMS_FOR_QUANT:
        return "always_f16"
    if meta.tensor_class in ALWAYS_F16_CLASSES:
        return "always_f16"
    if meta.tensor_class in ALWAYS_Q8_MIN_CLASSES:
        return "always_q8_min"
    return "optimize"


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class Profiler:
    """Measure per-tensor SNR at every QUANT_LEVEL for a model.

    Args:
        model_path: Directory containing safetensors files.
        cache_path:  Where to write/read snr_profile.json.
                     Defaults to model_path/snr_profile.json.
        sample_size: Max elements to sample per tensor for SNR estimation.
                     Sampling keeps memory bounded; 100k is accurate to ~0.5 dB.
        verbose:     Print progress.
    """

    def __init__(
        self,
        model_path: Path,
        cache_path: Optional[Path] = None,
        sample_size: int = 100_000,
        verbose: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.cache_path = cache_path or (self.model_path / "snr_profile.json")
        self.sample_size = sample_size
        self.verbose = verbose
        self._loader = ModelLoader(self.model_path)

    def run(self, force: bool = False) -> list[TensorProfile]:
        """Return profiles for all tensors, loading from cache if available."""
        if not force and self.cache_path.exists():
            if self.verbose:
                print(f"Loading cached SNR profile from {self.cache_path}")
            return self._load_cache()

        return self._profile_all()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _profile_all(self) -> list[TensorProfile]:
        tensors = list(self._loader.tensors())
        n = len(tensors)
        profiles: list[TensorProfile] = []

        if self.verbose:
            print(f"Profiling {n} tensors across {len(QUANT_LEVELS)} quant levels...")
            print(f"  (sampling up to {self.sample_size:,} elements per tensor)")

        for i, meta in enumerate(tensors):
            if self.verbose:
                pct = (i + 1) / n * 100
                print(
                    f"  [{i+1:4d}/{n}  {pct:5.1f}%]  {meta.name[:60]:<60}",
                    end="\r",
                    flush=True,
                )

            profile = self._profile_tensor(meta)
            profiles.append(profile)

        if self.verbose:
            print()  # newline after \r progress

        self._save_cache(profiles)
        if self.verbose:
            print(f"Profile saved to {self.cache_path}")

        return profiles

    def _profile_tensor(self, meta: TensorMeta) -> TensorProfile:
        policy = _assign_policy(meta)
        data = self._loader.load_tensor(meta)

        # Sample if large, for speed
        flat = data.flatten().astype(np.float32)
        if len(flat) > self.sample_size:
            idx = np.random.default_rng(42).choice(len(flat), self.sample_size, replace=False)
            flat = flat[idx]
        # Reshape to 2D for block quant (treat as (1, N) so block boundaries are uniform)
        sample = flat.reshape(1, -1)

        levels: dict[str, LevelResult] = {}

        for level in QUANT_LEVELS:
            # Respect policy minimum
            if policy == "always_f16" and level.name != "F16":
                continue
            if policy == "always_q8_min":
                idx_q8 = next(
                    i for i, q in enumerate(QUANT_LEVELS) if q.name == "Q8_0"
                )
                idx_this = next(
                    i for i, q in enumerate(QUANT_LEVELS) if q.name == level.name
                )
                if idx_this < idx_q8:
                    continue

            dequant = level.simulate(sample)
            snr = _compute_snr(sample, dequant)
            est_bytes = level.estimated_bytes(meta.num_params)

            levels[level.name] = LevelResult(snr_db=snr, estimated_bytes=est_bytes)

        return TensorProfile(
            name=meta.name,
            shape=list(meta.shape),
            num_params=meta.num_params,
            tensor_class=meta.tensor_class,
            policy=policy,
            levels=levels,
        )

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _save_cache(self, profiles: list[TensorProfile]) -> None:
        data = []
        for p in profiles:
            d = asdict(p)
            data.append(d)
        with open(self.cache_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_cache(self) -> list[TensorProfile]:
        with open(self.cache_path) as f:
            raw = json.load(f)
        profiles = []
        for d in raw:
            levels = {
                k: LevelResult(**v) for k, v in d.pop("levels", {}).items()
            }
            profiles.append(TensorProfile(**d, levels=levels))
        return profiles
