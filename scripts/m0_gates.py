"""M0 correctness gates. Run: uv run python scripts/m0_gates.py <model-name>

G-A  F16 GGUF tensors exactly match the MLX fp16 model (catches name-map/layout/permute bugs
     with zero forward passes).
G-B  Whole-model Q8_0 splice -> tiny KL (catches global bugs; Q8_0 is near-lossless).
G-C  Per-tensor-class Q8_0 splices -> each tiny KL.
G-D  Top-K+tail KL vs exact full-vocab KL on a damaged variant -> rel err < 2%.
G-E  Prefill throughput (tok/s) -> feeds the real cost table.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fragmap.gguf_io import GGUFTensorSource, sweepable_hf_names  # noqa: E402
from fragmap.kl import build_baseline_cache, score_variant, exact_kl_check  # noqa: E402
from fragmap.splice import SplicedModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Deterministic gate probe: natural-ish text through the model's own tokenizer + random ids.
GATE_TEXTS = [
    "The history of computing begins long before the first electronic machines. Mechanical "
    "aids to calculation, from the abacus to the slide rule, shaped how people thought about "
    "number and procedure for centuries before electricity entered the picture.",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[0]\n    "
    "left = [x for x in xs[1:] if x < pivot]\n    right = [x for x in xs[1:] if x >= pivot]\n"
    "    return quicksort(left) + [pivot] + quicksort(right)",
    "To solve 47 * 23, first compute 47 * 20 = 940, then 47 * 3 = 141, and add them: "
    "940 + 141 = 1081. Multiplication distributes over addition, which is why this works.",
    "La capital de Francia es París, una ciudad conocida por su arquitectura, sus museos y "
    "su papel central en la historia europea desde la Edad Media hasta la actualidad.",
]


def make_gate_probe(tokenizer, n_seq=12, seq_len=256, seed=7):
    # Natural text only: flat (random-token) distributions break top-K KL at any K
    # (measured: 53% rel err at K=256 vs 1.7% on natural text — m0_topk_diag).
    rows = []
    text_ids = sum((tokenizer.encode(t) for t in GATE_TEXTS), [])
    for i in range(n_seq):
        start = (i * 61) % max(1, len(text_ids) - seq_len)
        row = np.array((text_ids * 40)[start:start + seq_len])
        rows.append(row[:seq_len])
    return np.stack(rows).astype(np.int32)


def main(name: str):
    cfg = json.loads((ROOT / f"models/hf/{name}/config.json").read_text())
    family = ("llama" if "LlamaForCausalLM" in cfg.get("architectures", [""])[0]
              else "qwen")
    n_head, n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    print(f"== {name}: family={family} layers={cfg['num_hidden_layers']} "
          f"hidden={cfg['hidden_size']} heads={n_head}/{n_kv} "
          f"tied={cfg.get('tie_word_embeddings')} vocab={cfg['vocab_size']}")

    model = SplicedModel(str(ROOT / f"models/hf/{name}"))
    gguf_dir = ROOT / "models/gguf"
    src_f16 = GGUFTensorSource(str(gguf_dir / f"{name}-f16.gguf"),
                               "llama" if family == "llama" else family,
                               n_head=n_head, n_head_kv=n_kv)
    src_q8 = GGUFTensorSource(str(gguf_dir / f"{name}-pure-q8_0.gguf"),
                              "llama" if family == "llama" else family,
                              n_head=n_head, n_head_kv=n_kv)
    results = {}

    # G-A: exact match of every mapped F16 tensor against the MLX fp16 parameters.
    mism, checked = [], 0
    for hf in src_f16.hf_names():
        if hf not in model._params:
            if hf == "lm_head.weight" and cfg.get("tie_word_embeddings"):
                continue
            mism.append((hf, "missing-in-mlx"))
            continue
        g = src_f16.get_hf(hf).astype(np.float16)
        m = np.array(model.get(hf), copy=False).astype(np.float16)
        checked += 1
        if g.shape != m.shape or not np.array_equal(g, m):
            frac = float(np.mean(g != m)) if g.shape == m.shape else 1.0
            mism.append((hf, f"mismatch frac={frac:.4f}"))
    results["G-A"] = {"checked": checked, "mismatches": mism[:10], "pass": not mism}
    print(f"G-A exact-match: {checked} tensors, {len(mism)} mismatches "
          f"{'PASS' if not mism else 'FAIL ' + str(mism[:3])}")
    if mism:
        print("!! G-A failed - fix layout/permutation before trusting anything downstream")

    probe = make_gate_probe(model.tokenizer)
    t0 = time.time()
    cache = build_baseline_cache(model, probe, batch_size=4)
    base_secs = time.time() - t0
    n_tok = probe.shape[0] * probe.shape[1]
    results["G-E"] = {"tok_per_s": round(n_tok / base_secs, 1)}
    print(f"G-E throughput: {n_tok} tokens in {base_secs:.1f}s = {n_tok/base_secs:.0f} tok/s")

    # G-B: whole-model Q8_0 splice.
    for hf in src_q8.hf_names():
        if hf in model._params and src_q8.tensor_type(
                src_q8._hf_to_gguf(hf)).name not in ("F32", "F16"):
            model.splice(hf, src_q8.get_hf(hf))
    m = score_variant(model, probe, cache, batch_size=4).summary()
    model.restore()
    results["G-B"] = {"kl_mean": m["kl_mean"], "pass": m["kl_mean"] < 5e-3}
    print(f"G-B whole-model Q8_0: KL={m['kl_mean']:.3e} "
          f"{'PASS' if m['kl_mean'] < 5e-3 else 'FAIL'}")

    # G-C: one tensor per class x {first, middle} layers.
    classes = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    L = cfg["num_hidden_layers"]
    picks = [f"model.layers.{i}.{c}.weight" for c in classes for i in (0, L // 2)]
    picks.append("model.embed_tokens.weight")
    gc = {}
    for hf in picks:
        if hf not in model._params:
            continue
        model.splice(hf, src_q8.get_hf(hf))
        s = score_variant(model, probe, cache, batch_size=4).summary()
        model.restore()
        gc[hf] = s["kl_mean"]
    worst = max(gc.values())
    results["G-C"] = {"kl_by_tensor": gc, "worst": worst, "pass": worst < 5e-4}
    print(f"G-C per-tensor Q8_0: worst={worst:.3e} "
          f"{'PASS' if worst < 5e-4 else 'FAIL: ' + max(gc, key=gc.get)}")

    # G-D: top-K vs exact on a genuinely damaged variant (mid ffn_down at Q3_K).
    src_q3 = GGUFTensorSource(str(gguf_dir / f"{name}-pure-q3_k.gguf"),
                              "llama" if family == "llama" else family,
                              n_head=n_head, n_head_kv=n_kv)
    victim = f"model.layers.{L // 2}.mlp.down_proj.weight"
    baseline_model = SplicedModel(str(ROOT / f"models/hf/{name}"))
    model.splice(victim, src_q3.get_hf(victim))
    gd = exact_kl_check(baseline_model, model, probe, cache, batch_size=2)
    model.restore()
    results["G-D"] = {**gd, "pass": gd["rel_err"] < 0.02}
    print(f"G-D topK vs exact: exact={gd['exact_kl_mean']:.3e} "
          f"approx={gd['approx_kl_mean']:.3e} rel_err={gd['rel_err']:.3%} "
          f"{'PASS' if gd['rel_err'] < 0.02 else 'FAIL'}")

    out = ROOT / "runs" / f"m0-gates-{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
