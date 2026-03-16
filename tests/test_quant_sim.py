"""Tests for quant_sim.py — quantization level simulations."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quant_sim import (
    QUANT_LEVELS,
    LEVELS_BY_NAME,
    QuantLevel,
    next_level,
    prev_level,
    _simulate_symmetric_block,
)


class TestSymmetricBlockQuant:
    def test_output_shape_preserved(self):
        data = np.random.randn(64, 32).astype(np.float32)
        result = _simulate_symmetric_block(data, bits=4, block_size=32)
        assert result.shape == data.shape

    def test_non_multiple_shape(self):
        """Tensors whose size is not a multiple of block_size are handled."""
        data = np.random.randn(50).astype(np.float32)
        result = _simulate_symmetric_block(data, bits=4, block_size=32)
        assert result.shape == (50,)

    def test_8bit_near_identity(self):
        """Q8 should reconstruct with very small error."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal(1024).astype(np.float32)
        dq = _simulate_symmetric_block(data, bits=8, block_size=32)
        mse = np.mean((data - dq) ** 2)
        assert mse < 1e-4, f"Q8 MSE too high: {mse}"

    def test_2bit_higher_error_than_8bit(self):
        """Q2 should have more error than Q8."""
        rng = np.random.default_rng(1)
        data = rng.standard_normal(1024).astype(np.float32)
        dq8 = _simulate_symmetric_block(data, bits=8, block_size=32)
        dq2 = _simulate_symmetric_block(data, bits=2, block_size=16)
        mse8 = np.mean((data - dq8) ** 2)
        mse2 = np.mean((data - dq2) ** 2)
        assert mse2 > mse8 * 10, "Q2 should be substantially noisier than Q8"

    def test_all_zeros_no_nan(self):
        """Zero tensor should produce zero output, no division by zero."""
        data = np.zeros(128, dtype=np.float32)
        result = _simulate_symmetric_block(data, bits=4, block_size=32)
        assert not np.any(np.isnan(result))
        assert np.all(result == 0.0)

    def test_output_within_global_absmax(self):
        """Per-block quant: dequantized values are bounded by global absmax.

        Note: per-block scales mean dq can exceed data.min()/max() if a block's
        absmax is larger than the global signed extremes — but dq is always
        bounded by ±global_absmax.
        """
        rng = np.random.default_rng(2)
        data = rng.standard_normal(256).astype(np.float32)
        global_absmax = float(np.max(np.abs(data)))
        for bits in [2, 4, 8]:
            dq = _simulate_symmetric_block(data, bits=bits, block_size=32)
            assert float(np.max(np.abs(dq))) <= global_absmax * 1.01 + 1e-5


class TestQuantLevel:

    def test_levels_ordered_cheapest_first(self):
        """QUANT_LEVELS should be in ascending bits-per-weight order."""
        bpws = [q.bits_per_weight for q in QUANT_LEVELS]
        assert bpws == sorted(bpws)

    def test_estimated_bytes_f16(self):
        f16 = LEVELS_BY_NAME["F16"]
        assert f16.estimated_bytes(1024) == 1024 * 2

    def test_estimated_bytes_q4(self):
        q4 = LEVELS_BY_NAME["Q4_K_M"]
        result = q4.estimated_bytes(1024)
        expected = int(1024 * 4.58 / 8)
        assert result == expected

    def test_simulate_f16_roundtrip(self):
        """F16 simulate should be close to original (fp16 precision)."""
        data = np.array([1.5, -2.3, 0.0, 100.0], dtype=np.float32)
        f16 = LEVELS_BY_NAME["F16"]
        result = f16.simulate(data)
        assert np.allclose(data, result, rtol=1e-3), f"F16 roundtrip failed: {result}"

    def test_snr_increases_with_bits(self):
        """Higher bit levels should produce better SNR."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal(2048).astype(np.float32).reshape(1, -1)
        snrs = []
        for level in QUANT_LEVELS:
            dq = level.simulate(data)
            noise = np.mean((data - dq) ** 2)
            signal = np.mean(data ** 2)
            snr = 10 * np.log10(signal / noise) if noise > 1e-12 else 96.0
            snrs.append(snr)
        # Each successive level should have equal or better SNR
        for i in range(1, len(snrs)):
            assert snrs[i] >= snrs[i - 1] - 0.5, (
                f"{QUANT_LEVELS[i].name} SNR={snrs[i]:.1f} < "
                f"{QUANT_LEVELS[i-1].name} SNR={snrs[i-1]:.1f}"
            )


class TestNextLevel:
    def test_q2_next_is_q3_s(self):
        assert next_level("Q2_K") == "Q3_K_S"

    def test_q4_km_next_is_q5_0(self):
        assert next_level("Q4_K_M") == "Q5_0"

    def test_f16_has_no_next(self):
        assert next_level("F16") is None

    def test_q8_next_is_f16(self):
        assert next_level("Q8_0") == "F16"

    def test_prev_of_q3_ks_is_q2(self):
        assert prev_level("Q3_K_S") == "Q2_K"

    def test_prev_of_q2_is_none(self):
        assert prev_level("Q2_K") is None

    def test_all_levels_present(self):
        expected = {
            "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
            "Q4_0", "Q4_K_S", "Q4_K_M",
            "Q5_0", "Q5_K_S", "Q5_K_M",
            "Q6_K", "Q8_0", "F16",
        }
        assert set(LEVELS_BY_NAME.keys()) == expected
