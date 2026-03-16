"""
codebook_sim.py — K-means VQ (codebook) compression simulation.

K-means places centroids at the actual weight distribution rather than a
fixed grid, so it achieves better SNR than symmetric uniform quantization at
the same bits-per-weight.  The tradeoff is a fixed centroid-table overhead:

    size = (N * bits / 8) + (2^bits * 4)   bytes
          [  index storage  ]   [centroid table]

For large tensors the centroid table is negligible; for tiny tensors it can
exceed the cost of just keeping the tensor in F16.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# K-means kernel
# ---------------------------------------------------------------------------

def _kmeans_1d(data: np.ndarray, k: int, n_iter: int = 8) -> np.ndarray:
    """Fast 1D k-means returning exactly k centroid values.

    Uses percentile initialisation + sorted-centre / searchsorted assignment
    for O(N log K) per iteration instead of O(N * K).
    """
    flat = data.flatten().astype(np.float32)

    # Percentile initialisation — evenly spaced across the empirical CDF
    pct = np.linspace(0, 100, k)
    init = np.percentile(flat, pct).astype(np.float32)

    # Jitter duplicate centroids (happens on tensors with many repeated values)
    if len(np.unique(init)) < k:
        noise = np.random.default_rng(0).uniform(-1e-6, 1e-6, k).astype(np.float32)
        init = init + noise
    # Guarantee exactly k unique centroids (truncate/pad as needed)
    centers = np.sort(np.unique(init))
    if len(centers) < k:
        lo, hi = float(flat.min()), float(flat.max()) + 1e-6
        pad = np.linspace(lo, hi, k - len(centers) + 2, dtype=np.float32)[1:-1]
        centers = np.sort(np.unique(np.concatenate([centers, pad])))
    centers = centers[:k]                     # always exactly k entries
    if len(centers) < k:                      # safety: pad with last value
        centers = np.pad(centers, (0, k - len(centers)), mode='edge')

    for _ in range(n_iter):
        sorted_c = np.sort(centers)           # k elements
        mid = (sorted_c[:-1] + sorted_c[1:]) / 2.0  # k-1 elements
        labels = np.searchsorted(mid, flat)   # [0, k-1]
        labels = np.clip(labels, 0, k - 1)

        # Vectorised mean update
        sums   = np.bincount(labels, weights=flat, minlength=k).astype(np.float32)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        # Keep old position for empty clusters
        centers = np.where(counts > 0, sums / np.maximum(counts, 1.0), sorted_c)

    return centers


# ---------------------------------------------------------------------------
# CodebookLevel descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodebookLevel:
    """Descriptor for one k-means VQ level.

    Attributes:
        name:  Short label, e.g. "CB8".
        bits:  Bits per index (codebook has 2^bits entries).
    """
    name: str
    bits: int

    @property
    def n_centers(self) -> int:
        return 1 << self.bits  # 2^bits

    @property
    def bits_per_weight(self) -> float:
        """Nominal bits/weight — same as self.bits (centroid overhead negligible
        for large tensors; see estimated_bytes for the exact count)."""
        return float(self.bits)

    def estimated_bytes(self, num_params: int) -> int:
        """Packed indices + float32 centroid table."""
        index_bytes    = (num_params * self.bits + 7) // 8
        centroid_bytes = self.n_centers * 4   # float32 per centroid
        return index_bytes + centroid_bytes

    def simulate(self, data: np.ndarray) -> np.ndarray:
        """Quantise data via k-means and return the dequantised approximation."""
        flat = data.flatten().astype(np.float32)
        centers = _kmeans_1d(flat, self.n_centers)
        sorted_c = np.sort(centers)
        mid = (sorted_c[:-1] + sorted_c[1:]) / 2.0
        labels = np.searchsorted(mid, flat)
        labels = np.clip(labels, 0, self.n_centers - 1)
        return sorted_c[labels].reshape(data.shape)


# ---------------------------------------------------------------------------
# Level list
# ---------------------------------------------------------------------------

CODEBOOK_LEVELS: list[CodebookLevel] = [
    CodebookLevel("CB3",  3),   #   8 centers — very lossy
    CodebookLevel("CB4",  4),   #  16 centers
    CodebookLevel("CB5",  5),   #  32 centers
    CodebookLevel("CB6",  6),   #  64 centers
    CodebookLevel("CB7",  7),   # 128 centers
    CodebookLevel("CB8",  8),   # 256 centers
    CodebookLevel("CB10", 10),  # 1024 centers
    CodebookLevel("CB12", 12),  # 4096 centers
]

CODEBOOK_LEVELS_BY_NAME: dict[str, CodebookLevel] = {
    cb.name: cb for cb in CODEBOOK_LEVELS
}
