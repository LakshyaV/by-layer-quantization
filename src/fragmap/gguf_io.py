"""Read GGUF files, dequantize tensors, and map them into HF/MLX naming and layout.

The two correctness traps this module owns:
1. GGUF stores shapes in ggml order (innermost dim first) — we reshape to HF (rows, cols).
2. For family="llama", convert_hf_to_gguf.py permutes attn_q/attn_k rows into ggml's rope
   layout; splicing back into an HF-layout model requires the inverse permutation.
   Qwen2/Qwen3 converters do not permute. Guarded by the F16 exact-match and Q8_0 gates.
"""

from __future__ import annotations

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as QT
from gguf.quants import dequantize

FLOAT_TYPES = {QT.F32, QT.F16, QT.BF16}

# GGUF suffix -> HF template. {i} = layer index.
_NAME_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
    "blk.{i}.attn_q.weight": "model.layers.{i}.self_attn.q_proj.weight",
    "blk.{i}.attn_k.weight": "model.layers.{i}.self_attn.k_proj.weight",
    "blk.{i}.attn_v.weight": "model.layers.{i}.self_attn.v_proj.weight",
    "blk.{i}.attn_output.weight": "model.layers.{i}.self_attn.o_proj.weight",
    "blk.{i}.attn_q.bias": "model.layers.{i}.self_attn.q_proj.bias",
    "blk.{i}.attn_k.bias": "model.layers.{i}.self_attn.k_proj.bias",
    "blk.{i}.attn_v.bias": "model.layers.{i}.self_attn.v_proj.bias",
    "blk.{i}.attn_q_norm.weight": "model.layers.{i}.self_attn.q_norm.weight",
    "blk.{i}.attn_k_norm.weight": "model.layers.{i}.self_attn.k_norm.weight",
    "blk.{i}.attn_norm.weight": "model.layers.{i}.input_layernorm.weight",
    "blk.{i}.ffn_norm.weight": "model.layers.{i}.post_attention_layernorm.weight",
    "blk.{i}.ffn_gate.weight": "model.layers.{i}.mlp.gate_proj.weight",
    "blk.{i}.ffn_up.weight": "model.layers.{i}.mlp.up_proj.weight",
    "blk.{i}.ffn_down.weight": "model.layers.{i}.mlp.down_proj.weight",
}


def gguf_to_hf_name(gguf_name: str) -> str | None:
    """Map a GGUF tensor name to its HF name; None for tensors with no HF twin."""
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".", 2)
        layer, rest = parts[1], parts[2]
        key = f"blk.{{i}}.{rest}"
        tmpl = _NAME_MAP.get(key)
        return tmpl.format(i=layer) if tmpl else None
    return _NAME_MAP.get(gguf_name)


def unpermute_rope_rows(w: np.ndarray, n_head: int) -> np.ndarray:
    """Inverse of convert_hf_to_gguf.py's LlamaModel.permute (ggml rope layout -> HF)."""
    out_dim = w.shape[0]
    return (
        w.reshape(n_head, out_dim // n_head // 2, 2, *w.shape[1:])
        .swapaxes(1, 2)
        .reshape(w.shape)
    )


class GGUFTensorSource:
    """Lazy per-tensor dequantized access to one GGUF file, in HF layout and naming."""

    def __init__(self, path: str, family: str, n_head: int | None = None,
                 n_head_kv: int | None = None):
        self.reader = GGUFReader(path)
        self.family = family
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        if family == "llama" and (n_head is None or n_head_kv is None):
            raise ValueError("family='llama' needs n_head/n_head_kv for un-permutation")
        self._by_name = {t.name: t for t in self.reader.tensors}

    def gguf_names(self) -> list[str]:
        return list(self._by_name.keys())

    def hf_names(self) -> list[str]:
        return [n for n in (gguf_to_hf_name(g) for g in self._by_name) if n is not None]

    def tensor_type(self, gguf_name: str) -> QT:
        return QT(self._by_name[gguf_name].tensor_type)

    def get_hf(self, hf_name: str) -> np.ndarray:
        """Dequantized tensor in HF layout (float32)."""
        gname = self._hf_to_gguf(hf_name)
        t = self._by_name[gname]
        hf_shape = tuple(int(d) for d in reversed(list(t.shape)))
        ttype = QT(t.tensor_type)
        raw = np.asarray(t.data)
        if ttype in FLOAT_TYPES:
            if ttype == QT.BF16:
                arr = raw.view(np.uint16).astype(np.uint32) << 16
                arr = arr.view(np.float32)
            else:
                arr = raw.astype(np.float32)
            arr = arr.reshape(hf_shape)
        else:
            flat = raw.reshape(-1).view(np.uint8)
            arr = dequantize(flat, ttype).reshape(hf_shape)
        if self.family == "llama" and hf_name.endswith((".self_attn.q_proj.weight",
                                                        ".self_attn.k_proj.weight")):
            n_head = self.n_head if "q_proj" in hf_name else self.n_head_kv
            arr = unpermute_rope_rows(arr, n_head)
        return arr

    def _hf_to_gguf(self, hf_name: str) -> str:
        for gname in self._by_name:
            if gguf_to_hf_name(gname) == hf_name:
                return gname
        raise KeyError(f"no GGUF tensor maps to {hf_name}")


# The tensor classes we sweep (weights only; norms/biases stay high precision as in k-quants).
SWEEPABLE_SUFFIXES = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def sweepable_hf_names(source: GGUFTensorSource, include_embed: bool = True) -> list[str]:
    names = []
    for hf in source.hf_names():
        if hf.endswith(SWEEPABLE_SUFFIXES):
            names.append(hf)
        elif include_embed and hf in ("model.embed_tokens.weight", "lm_head.weight"):
            names.append(hf)
    return sorted(names, key=_sort_key)


def _sort_key(hf_name: str):
    if "layers." in hf_name:
        layer = int(hf_name.split("layers.")[1].split(".")[0])
        return (1, layer, hf_name)
    return (0 if "embed" in hf_name else 2, 0, hf_name)
