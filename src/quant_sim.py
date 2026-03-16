"""
quant_sim.py — Numpy simulations of GGUF quantization levels.

Each level's simulate() function:
  - quantizes data to the specified bit depth using per-block symmetric absmax scaling
  - dequantizes back to float32
  - returns the dequantized array

Size estimates use the actual bits-per-weight ratios from the GGUF specification,
not the simulation block size, so byte counts match real llama-quantize output.

SNR notes:
  - K-quant variants (Q4_K_M etc.) also quantize their scale values, adding a small
    secondary noise floor (~0.5–2 dB).  This is not modeled here; the simulation
    gives a slight over-estimate of SNR — conservative for the optimizer.
  - S/M/L variants of the same bit-depth differ mainly in scale precision.  The
    simulation uses the same sim_bits for all; the bits_per_weight field carries
    the size difference.
  - Q2_K uses non-uniform (asymmetric) quantization in GGUF; symmetric simulation
    underestimates SNR by ~1–3 dB.  Conservative direction is safe for planning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Core simulation kernel
# ---------------------------------------------------------------------------

def _simulate_symmetric_block(
    data: np.ndarray,
    bits: int,
    block_size: int,
) -> np.ndarray:
    """Symmetric per-block absmax quantization, vectorized.

    Args:
        data:       float32 array, any shape.
        bits:       integer bit depth (1–8).
        block_size: number of elements per quantization block.

    Returns:
        Dequantized float32 array, same shape as data.
    """
    flat = data.flatten().astype(np.float32)
    n = len(flat)

    # Pad to an exact multiple of block_size
    pad = (-n) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])

    blocks = flat.reshape(-1, block_size)           # (B, block_size)
    max_val = float((1 << (bits - 1)) - 1)          # e.g. 7 for 3-bit, 127 for 8-bit

    scales = np.max(np.abs(blocks), axis=1, keepdims=True)    # (B, 1)
    safe_scales = np.where(scales == 0.0, 1.0, scales)

    q = np.round(blocks / safe_scales * max_val).clip(-max_val, max_val)
    dequant = (q / max_val) * safe_scales

    return dequant.flatten()[:n].reshape(data.shape)


# ---------------------------------------------------------------------------
# QuantLevel descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantLevel:
    """Descriptor for one GGUF quantization level.

    Attributes:
        name:             GGUF type string (e.g. "Q4_K_M").
        bits_per_weight:  Effective bits/weight including overhead (from GGUF spec).
                          Used for byte-size estimation only.
        sim_bits:         Integer bits used in the SNR simulation (0 = F16, no quant).
        block_size:       Elements per quantization block in the simulation.
    """
    name: str
    bits_per_weight: float
    sim_bits: int
    block_size: int

    def estimated_bytes(self, num_params: int) -> int:
        """Byte size estimate for a tensor with num_params weights."""
        if self.sim_bits == 0:
            return num_params * 2           # F16 / BF16: 2 bytes
        return int(num_params * self.bits_per_weight / 8)

    def simulate(self, data: np.ndarray) -> np.ndarray:
        """Return dequantized approximation of data at this quant level."""
        if self.sim_bits == 0:
            # F16: round-trip through float16
            return data.astype(np.float16).astype(np.float32)
        return _simulate_symmetric_block(data, self.sim_bits, self.block_size)


# ---------------------------------------------------------------------------
# Ordered level list — cheapest first
# ---------------------------------------------------------------------------
#
# bits_per_weight values from the GGUF specification / llama.cpp source.
# Variants (S/M/L/XS) of the same bit-depth differ in super-block scale
# precision:  S = Q4 scales, M = Q6 scales, L = Q8 scales.
# All S/M/L variants of the same bit-depth use the same sim_bits — their
# SNR difference (~0.5–1.5 dB) comes from scale quantization which is not
# modeled in the simulation.

QUANT_LEVELS: list[QuantLevel] = [
    # ---- 2-bit ----
    QuantLevel("Q2_K",     2.63,  2, 16),   # asymmetric in GGUF; sim is conservative

    # ---- 3-bit ----
    QuantLevel("Q3_K_S",   3.00,  3, 32),   # Q4 super-block scales
    QuantLevel("Q3_K_M",   3.35,  3, 32),   # Q6 super-block scales
    QuantLevel("Q3_K_L",   3.75,  3, 32),   # Q8 super-block scales

    # ---- 4-bit ----
    QuantLevel("Q4_K_S",   4.37,  4, 32),   # Q4 super-block scales
    QuantLevel("Q4_0",     4.50,  4, 32),   # uniform per-block, no super-block
    QuantLevel("Q4_K_M",   4.58,  4, 32),   # Q6 super-block scales

    # ---- 5-bit ----
    QuantLevel("Q5_0",     5.50,  5, 32),   # uniform per-block
    QuantLevel("Q5_K_S",   5.52,  5, 32),   # Q4 super-block scales
    QuantLevel("Q5_K_M",   5.69,  5, 32),   # Q6 super-block scales

    # ---- 6-bit ----
    QuantLevel("Q6_K",     6.57,  6, 32),   # only one Q6 variant in GGUF

    # ---- 8-bit ----
    QuantLevel("Q8_0",     8.50,  8, 32),   # uniform per-block

    # ---- float ----
    QuantLevel("F16",     16.00,  0,  0),
]

# Fast lookup by name
LEVELS_BY_NAME: dict[str, QuantLevel] = {q.name: q for q in QUANT_LEVELS}


def next_level(name: str) -> str | None:
    """Return the name of the next higher quality level, or None if already F16."""
    for i, q in enumerate(QUANT_LEVELS):
        if q.name == name and i + 1 < len(QUANT_LEVELS):
            return QUANT_LEVELS[i + 1].name
    return None


def prev_level(name: str) -> str | None:
    """Return the name of the next lower quality level, or None if already at minimum."""
    for i, q in enumerate(QUANT_LEVELS):
        if q.name == name and i > 0:
            return QUANT_LEVELS[i - 1].name
    return None
