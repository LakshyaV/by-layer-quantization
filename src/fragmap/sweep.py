"""Resumable sweep loop: splice variant -> score -> restore -> append one JSONL row.

A variant is a tuple of (hf_tensor_name, qtype) pairs: one pair for Sweep A, two for the
pairs mini-sweep, all-tensors for the Sweep-B baseline. Rows are keyed by a content hash;
rerunning a sweep skips completed keys, so a crash costs at most one variant.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .gguf_io import GGUFTensorSource
from .kl import BaselineCache, score_variant, score_variant_exact
from .splice import SplicedModel

HARNESS_VERSION = "0.1"


def variant_key(model_id: str, tensors: tuple[tuple[str, str], ...], probe_sha: str) -> str:
    payload = json.dumps([model_id, sorted(tensors), probe_sha, HARNESS_VERSION])
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def load_done_keys(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_row(out_path: Path, row: dict) -> None:
    tmp = out_path.with_suffix(".tmp")
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    if tmp.exists():
        tmp.unlink()


def run_sweep(
    model: SplicedModel,
    model_id: str,
    sources: dict[str, GGUFTensorSource],   # qtype -> pure-quant GGUF source
    variants: list[tuple[tuple[str, str], ...]],
    tokens,                                  # (N, L) np.int32
    cache: BaselineCache,
    probe_sha: str,
    out_path: Path,
    seq_domains: list[str] | None = None,
    batch_size: int = 8,
    log_every: int = 10,
    baseline_model: SplicedModel | None = None,   # set -> exact full-vocab KL mode
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(out_path)
    todo = [v for v in variants
            if variant_key(model_id, v, probe_sha) not in done]
    print(f"[sweep] {model_id}: {len(todo)} to run, {len(variants) - len(todo)} cached")

    for i, tensors in enumerate(todo):
        key = variant_key(model_id, tensors, probe_sha)
        t0 = time.time()
        try:
            for hf_name, qtype in tensors:
                model.splice(hf_name, sources[qtype].get_hf(hf_name))
            if baseline_model is not None:
                metrics = score_variant_exact(baseline_model, model, tokens,
                                              batch_size=max(2, batch_size // 2))
            else:
                metrics = score_variant(model, tokens, cache, batch_size=batch_size)
        finally:
            model.restore()
        row = {
            "key": key,
            "model": model_id,
            "tensors": list(map(list, tensors)),
            "probe_sha": probe_sha,
            "harness": HARNESS_VERSION,
            "seconds": round(time.time() - t0, 2),
            **metrics.summary(),
        }
        if seq_domains is not None:
            row["kl_by_domain"] = _domain_means(metrics, seq_domains)
        append_row(out_path, row)
        if (i + 1) % log_every == 0 or i == len(todo) - 1:
            print(f"[sweep] {i+1}/{len(todo)} last={tensors[0][0] if len(tensors)==1 else f'{len(tensors)} tensors'}"
                  f" kl={row['kl_mean']:.3e} {row['seconds']}s")


def _domain_means(metrics, seq_domains: list[str]) -> dict[str, float]:
    per_seq = metrics.kl_sum / metrics.n_pos
    by = {}
    for d in sorted(set(seq_domains)):
        sel = [i for i, dd in enumerate(seq_domains) if dd == d]
        by[d] = float(per_seq[sel].mean())
    return by


def interleave_variants(variants: list) -> list:
    """Round-robin order so a partial overnight run yields an unbiased partial map."""
    import random
    rng = random.Random(1337)
    shuffled = variants[:]
    rng.shuffle(shuffled)
    return shuffled
