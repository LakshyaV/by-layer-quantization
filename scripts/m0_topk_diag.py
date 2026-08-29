"""Diagnose top-K+tail KL error vs K, split by natural vs random-token sequences,
and measure steady-state throughput. Decides K (or exact-KL fallback) for the sweeps."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fragmap.gguf_io import GGUFTensorSource  # noqa: E402
from fragmap.kl import build_baseline_cache, score_variant, exact_kl_check  # noqa: E402
from fragmap.splice import SplicedModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(name="qwen3-1.7b"):
    sys.path.insert(0, str(Path(__file__).parent))
    from m0_gates import make_gate_probe
    cfg = json.loads((ROOT / f"models/hf/{name}/config.json").read_text())
    model = SplicedModel(str(ROOT / f"models/hf/{name}"))
    baseline = SplicedModel(str(ROOT / f"models/hf/{name}"))
    src_q3 = GGUFTensorSource(str(ROOT / f"models/gguf/{name}-pure-q3_k.gguf"), "qwen")
    victim = f"model.layers.{cfg['num_hidden_layers'] // 2}.mlp.down_proj.weight"

    probe = make_gate_probe(model.tokenizer, n_seq=12, seq_len=256)
    natural = probe[[i for i in range(12) if i % 3 != 2]]
    randoms = probe[[i for i in range(12) if i % 3 == 2]]

    model.splice(victim, src_q3.get_hf(victim))
    for label, toks in [("natural", natural), ("random", randoms)]:
        for K in (256, 1024, 4096):
            cache = build_baseline_cache(baseline, toks, batch_size=4, k=K)
            r = exact_kl_check(baseline, model, toks, cache,
                               n_positions=toks.shape[0] * 255, batch_size=2)
            tail = float(np.exp(cache.tail_logp).mean())
            print(f"{label:8s} K={K:5d} exact={r['exact_kl_mean']:.3e} "
                  f"approx={r['approx_kl_mean']:.3e} rel_err={r['rel_err']:.2%} "
                  f"base_tail_mass={tail:.4f}")
    model.restore()

    # steady-state throughput: 3 timed passes over 24 x 512 natural-ish tokens
    big = np.tile(natural, (6, 2))[:, :512].astype(np.int32)
    import mlx.core as mx
    logits = model.logits(mx.array(big[:4])); mx.eval(logits)  # warmup
    t0 = time.time()
    for s in range(0, big.shape[0], 8):
        logits = model.logits(mx.array(big[s:s+8])); mx.eval(logits)
    dt = time.time() - t0
    print(f"steady-state prefill: {big.size} tokens in {dt:.1f}s = {big.size/dt:.0f} tok/s")


if __name__ == "__main__":
    main(*sys.argv[1:])
