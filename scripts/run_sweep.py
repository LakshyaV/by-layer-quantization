"""Run a fragility sweep.

Examples:
  uv run python scripts/run_sweep.py qwen3-1.7b --mode block --types Q3_K,Q4_K   # M1 first-look
  uv run python scripts/run_sweep.py qwen3-1.7b --mode tensor --types Q2_K,Q3_K,Q4_K,Q6_K
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fragmap.gguf_io import GGUFTensorSource, sweepable_hf_names  # noqa: E402
from fragmap.kl import BaselineCache, build_baseline_cache  # noqa: E402
from fragmap.probes import tokenize_probe  # noqa: E402
from fragmap.splice import SplicedModel  # noqa: E402
from fragmap.sweep import interleave_variants, run_sweep  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def get_source(name: str, qtype: str, family: str, n_head: int, n_kv: int) -> GGUFTensorSource:
    short = qtype.lower().replace("_m", "")
    path = ROOT / f"models/gguf/{name}-pure-{short}.gguf"
    return GGUFTensorSource(str(path), family, n_head=n_head, n_head_kv=n_kv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mode", choices=["tensor", "block"], default="tensor")
    ap.add_argument("--types", default="Q3_K,Q4_K")
    ap.add_argument("--seqs-per-domain", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--exact", action="store_true",
                    help="exact full-vocab KL (two models resident; 1.5-1.7B only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name = args.model
    cfg = json.loads((ROOT / f"models/hf/{name}/config.json").read_text())
    family = "llama" if "LlamaForCausalLM" in cfg.get("architectures", [""])[0] else "qwen"
    n_head, n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    L = cfg["num_hidden_layers"]

    model = SplicedModel(str(ROOT / f"models/hf/{name}"))
    tokens, domains, tok_sha = tokenize_probe(
        model.tokenizer, ROOT / "data/probes/probes.parquet",
        seqs_per_domain=args.seqs_per_domain)
    print(f"[run] probe: {tokens.shape} sha={tok_sha} domains={sorted(set(domains))}")

    baseline_model = None
    cache = None
    if args.exact:
        baseline_model = SplicedModel(str(ROOT / f"models/hf/{name}"))
        print("[run] exact full-vocab KL mode (two models resident)")
    else:
        cache_path = ROOT / f"caches/topk/{name}-{tok_sha}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            cache = BaselineCache.load(cache_path)
            print(f"[run] baseline cache loaded: {cache_path.name}")
        else:
            t0 = time.time()
            cache = build_baseline_cache(model, tokens, batch_size=args.batch_size)
            cache.save(cache_path)
            n_tok = tokens.shape[0] * tokens.shape[1]
            print(f"[run] baseline cache built in {time.time()-t0:.0f}s "
                  f"({n_tok/(time.time()-t0):.0f} tok/s) -> {cache_path.name}")

    qtypes = args.types.split(",")
    sources = {q: get_source(name, q, family, n_head, n_kv) for q in qtypes}
    tensor_names = sweepable_hf_names(sources[qtypes[0]])
    # tied models: GGUF may carry output.weight but MLX has no separate lm_head —
    # splicing embed_tokens covers both roles (same array in the tied forward).
    tensor_names = [t for t in tensor_names if t in model._params]
    print(f"[run] {len(tensor_names)} sweepable tensors, {L} layers")

    variants = []
    if args.mode == "tensor":
        for t in tensor_names:
            for q in qtypes:
                variants.append(((t, q),))
    else:  # block mode: all 7 weight tensors of one layer at once
        for i in range(L):
            block = [t for t in tensor_names if f".layers.{i}." in t]
            for q in qtypes:
                variants.append(tuple((t, q) for t in block))
        for q in qtypes:  # embeddings as their own "block"
            emb = [t for t in tensor_names if "embed" in t or t == "lm_head.weight"]
            variants.append(tuple((t, q) for t in emb))

    out = Path(args.out) if args.out else \
        ROOT / f"runs/sweepA-{args.mode}-{name}/results.jsonl"
    run_sweep(model, name, sources, interleave_variants(variants), tokens, cache,
              tok_sha, out, seq_domains=domains, batch_size=args.batch_size,
              baseline_model=baseline_model)
    print(f"[run] complete -> {out}")


if __name__ == "__main__":
    main()
