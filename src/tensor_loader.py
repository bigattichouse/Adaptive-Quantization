"""
tensor_loader.py — Load tensor metadata and raw data from safetensors model files.

Handles both single-shard and multi-shard models (model.safetensors.index.json).
Tensors are loaded on demand via direct file I/O — the full model is never
held in memory at once.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Tensor classification
# ---------------------------------------------------------------------------

def classify_tensor(name: str) -> str:
    """Classify a tensor by its role in the model.

    Returns one of: 'layernorm', 'embedding', 'attention', 'mlp_ffn',
                    'router', 'ssm_core', 'other'.

    Supports Llama, Mistral, Gemma, Qwen naming conventions.
    """
    n = name.lower()

    # Norms — check first; 'norm' appears in many names
    if any(k in n for k in ('layernorm', 'layer_norm', 'ln_', 'rms_norm',
                             'pre_feedforward_layernorm',
                             'post_feedforward_layernorm')):
        return 'layernorm'
    if 'norm' in n:
        return 'layernorm'

    # Routers / gates (MoE)
    if 'router' in n:
        return 'router'
    if 'gate' in n and 'expert' not in n and 'mlp' not in n:
        return 'router'

    # SSM / linear-attention scalars
    if any(k in n for k in ('a_log', 'dt_bias', 'conv1d', 'o_norm',
                             'f_a_proj', 'f_b_proj', 'g_a_proj', 'g_b_proj')):
        return 'ssm_core'

    # Embeddings and output head
    if any(k in n for k in ('embed', 'lm_head', 'wte')):
        return 'embedding'

    # MoE experts
    if any(k in n for k in ('.experts.', 'block_sparse', 'expert')):
        return 'moe_expert'

    # Attention projections
    if any(k in n for k in ('self_attn', 'attn', 'linear_attn')):
        return 'attention'

    # MLP / FFN
    if any(k in n for k in ('mlp', 'ffn', 'feed_forward')):
        return 'mlp_ffn'

    return 'other'


# ---------------------------------------------------------------------------
# TensorMeta — lightweight descriptor, no weight data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorMeta:
    """All metadata needed to load a tensor on demand."""
    name: str
    shape: tuple[int, ...]
    dtype: str          # "BF16", "F16", "F32"
    file: Path
    data_offset: int    # byte offset from the start of the file
    data_size: int      # byte length
    tensor_class: str   # from classify_tensor()
    num_params: int


# ---------------------------------------------------------------------------
# ModelLoader
# ---------------------------------------------------------------------------

class ModelLoader:
    """Index a safetensors model and load individual tensors on demand.

    Supports:
      - Single-shard:  one or more *.safetensors files without an index
      - Multi-shard:   model.safetensors.index.json pointing to shards
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._index: dict[str, TensorMeta] = self._build_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def tensor_names(self) -> list[str]:
        return list(self._index.keys())

    def tensors(self) -> Iterator[TensorMeta]:
        """Yield TensorMeta for every tensor in the model."""
        yield from self._index.values()

    def get(self, name: str) -> TensorMeta | None:
        return self._index.get(name)

    def load_tensor(self, meta: TensorMeta) -> np.ndarray:
        """Load and decode one tensor to float32."""
        with open(meta.file, "rb") as f:
            f.seek(meta.data_offset)
            raw = f.read(meta.data_size)
        return _decode_raw(raw, meta.dtype, meta.shape)

    def baseline_bytes(self) -> int:
        """Total byte size of the model in its stored dtype (BF16 baseline)."""
        return sum(m.data_size for m in self._index.values())

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> dict[str, TensorMeta]:
        index_file = self.model_path / "model.safetensors.index.json"

        if index_file.exists():
            with open(index_file) as f:
                idx = json.load(f)
            shard_map: dict[str, str] = idx["weight_map"]
        else:
            st_files = sorted(self.model_path.glob("*.safetensors"))
            if not st_files:
                raise FileNotFoundError(
                    f"No .safetensors files found in {self.model_path}"
                )
            # Map every tensor in every shard to its file
            shard_map = {}
            for sf in st_files:
                for name in _read_shard_tensor_names(sf):
                    shard_map[name] = sf.name

        # Group by shard so we read each header once
        by_shard: dict[str, list[str]] = {}
        for tname, shard in shard_map.items():
            by_shard.setdefault(shard, []).append(tname)

        result: dict[str, TensorMeta] = {}
        for shard_filename, tensor_names in by_shard.items():
            shard_path = self.model_path / shard_filename
            header, header_size = _read_safetensors_header(shard_path)
            # Tensor data starts immediately after the 8-byte length prefix + header
            data_base = 8 + header_size

            for tname in tensor_names:
                if tname == "__metadata__" or tname not in header:
                    continue
                info = header[tname]
                dtype = info["dtype"]
                shape = tuple(info["shape"])
                start, end = info["data_offsets"]
                num_params = int(np.prod(shape)) if shape else 0

                result[tname] = TensorMeta(
                    name=tname,
                    shape=shape,
                    dtype=dtype,
                    file=shard_path,
                    data_offset=data_base + start,
                    data_size=end - start,
                    tensor_class=classify_tensor(tname),
                    num_params=num_params,
                )

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_safetensors_header(path: Path) -> tuple[dict, int]:
    """Return (header_dict, header_size_bytes) for a safetensors file."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
    return json.loads(header_bytes), header_size


def _read_shard_tensor_names(path: Path) -> list[str]:
    header, _ = _read_safetensors_header(path)
    return [k for k in header if k != "__metadata__"]


def _decode_raw(raw: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    """Decode raw bytes from a safetensors file to a float32 numpy array."""
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        data = u32.view(np.float32)
    elif dtype == "F16":
        data = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype == "F32":
        data = np.frombuffer(raw, dtype=np.float32)
    else:
        # Unknown — attempt BF16
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        data = u32.view(np.float32)

    return data.reshape(shape)
