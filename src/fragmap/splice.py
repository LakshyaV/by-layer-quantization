"""Load an MLX model in fp16 and swap individual weight tensors in and out."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load as mlx_load


class SplicedModel:
    def __init__(self, hf_dir: str):
        self.model, self.tokenizer = mlx_load(hf_dir)
        # fp16 baseline everywhere: matches the f16 GGUF the pure-quant files derive from.
        self.model.update(
            tree_unflatten([(k, v.astype(mx.float16) if v.dtype in (mx.bfloat16, mx.float32) else v)
                            for k, v in tree_flatten(self.model.parameters())])
        )
        mx.eval(self.model.parameters())
        self._params = dict(tree_flatten(self.model.parameters()))
        self._originals: dict[str, mx.array] = {}

    def param_names(self) -> list[str]:
        return list(self._params.keys())

    def get(self, name: str) -> mx.array:
        return self._params[name]

    def splice(self, name: str, value: np.ndarray) -> None:
        """Replace parameter `name`; keeps the original for restore()."""
        if name not in self._params:
            raise KeyError(f"model has no parameter {name}; known e.g. {self.param_names()[:3]}")
        cur = self._params[name]
        if tuple(cur.shape) != tuple(value.shape):
            raise ValueError(f"{name}: shape {tuple(value.shape)} != model {tuple(cur.shape)}")
        if name not in self._originals:
            self._originals[name] = cur
        new = mx.array(value.astype(np.float16))
        self.model.update(tree_unflatten([(name, new)]))
        self._params[name] = new

    def restore(self, name: str | None = None) -> None:
        names = [name] if name else list(self._originals.keys())
        updates = []
        for n in names:
            if n in self._originals:
                orig = self._originals.pop(n)
                updates.append((n, orig))
                self._params[n] = orig
        if updates:
            self.model.update(tree_unflatten(updates))

    def spliced_names(self) -> list[str]:
        return list(self._originals.keys())

    def logits(self, tokens: mx.array) -> mx.array:
        """Teacher-forced logits for a (B, L) batch."""
        return self.model(tokens)
