import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from fragmap.splice import SplicedModel
from fragmap.gguf_io import GGUFTensorSource
from fragmap.kl import score_variant_exact
from fragmap.probes import tokenize_probe

model = SplicedModel("models/hf/qwen3-1.7b")
base = SplicedModel("models/hf/qwen3-1.7b")
srcs = {q: GGUFTensorSource(f"models/gguf/qwen3-1.7b-pure-{s}.gguf", "qwen")
        for q, s in [("Q3_K", "q3_k"), ("Q4_K", "q4_k")]}
toks, doms, sha = tokenize_probe(model.tokenizer, Path("data/probes/probes.parquet"))
toks = toks[::3]
classes = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
for c in classes:
    nm = f"model.layers.2.{c}.weight"
    w = np.array(base.get(nm))
    line = f"{c:18s} max|w|={float(np.abs(w).max()):8.2f}"
    for q in ("Q3_K", "Q4_K"):
        model.splice(nm, srcs[q].get_hf(nm))
        m = score_variant_exact(base, model, toks, batch_size=8).summary()
        model.restore()
        line += f"  {q}={m['kl_mean']:.3e}"
    print(line, flush=True)
