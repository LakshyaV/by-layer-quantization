"""G1 first-look analysis of a block-mode sweep.

Reads runs/sweepA-block-<model>/results.jsonl and reports:
- the block map (KL per layer per type; embeddings separate)
- split-half stability (even vs odd sequences, Spearman)
- per-domain profiles + cross-domain rank correlations vs the split-half ceiling
- saves a heatmap + domain-profile figure

Usage: uv run python scripts/analyze_firstlook.py qwen3-1.7b
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def block_of(row) -> str:
    t = row["tensors"][0][0]
    if "layers." in t:
        return f"L{int(t.split('layers.')[1].split('.')[0]):02d}"
    return "EMB"


def main(name: str):
    path = ROOT / f"runs/sweepA-block-{name}/results.jsonl"
    rows = [json.loads(l) for l in open(path)]
    print(f"{len(rows)} variants loaded from {path.name}\n")

    types = sorted({r["tensors"][0][1] for r in rows})
    blocks = sorted({block_of(r) for r in rows})
    layer_blocks = [b for b in blocks if b != "EMB"]

    grid = {}
    for r in rows:
        grid[(block_of(r), r["tensors"][0][1])] = r

    print(f"{'block':>6} " + " ".join(f"{t:>10}" for t in types))
    for b in blocks:
        vals = [grid.get((b, t)) for t in types]
        print(f"{b:>6} " + " ".join(
            f"{v['kl_mean']:.3e}" if v else "     -    " for v in vals))

    for t in types:
        seq_mat = np.array([grid[(b, t)]["kl_seq_mean"] for b in layer_blocks])
        even, odd = seq_mat[:, 0::2].mean(1), seq_mat[:, 1::2].mean(1)
        rho = spearmanr(even, odd).statistic
        print(f"\nsplit-half Spearman ({t}, layer blocks): {rho:.3f}")

    t0 = types[0]
    domains = sorted(grid[(layer_blocks[0], t0)]["kl_by_domain"].keys())
    prof = {d: np.array([grid[(b, t0)]["kl_by_domain"][d] for b in layer_blocks])
            for d in domains}
    print(f"\ncross-domain Spearman of {t0} block profiles:")
    print(" " * 16 + " ".join(f"{d[:7]:>8}" for d in domains))
    for d1 in domains:
        cells = []
        for d2 in domains:
            rho = spearmanr(prof[d1], prof[d2]).statistic
            cells.append(f"{rho:>8.2f}")
        print(f"{d1[:15]:>15} " + " ".join(cells))

    # split-half ceiling per domain (12 seqs -> 6/6)
    print("\nper-domain split-half ceiling (same-domain, even/odd seqs):")
    dom_of_seq = None
    for d in domains:
        idx = None
        # per-domain per-seq values aren't stored per block; approximate ceiling with
        # all-domain split-half restricted later in the tensor-mode analysis.
        pass
    print("  (tensor-mode sweep stores what's needed; block-level uses aggregate rho above)")

    top = sorted(layer_blocks, key=lambda b: -grid[(b, t0)]["kl_mean"])[:5]
    bot = sorted(layer_blocks, key=lambda b: grid[(b, t0)]["kl_mean"])[:5]
    print(f"\nmost fragile blocks @{t0}: {top}")
    print(f"most robust blocks @{t0}: {bot}")
    emb = grid.get(("EMB", t0))
    if emb:
        print(f"embeddings @{t0}: kl={emb['kl_mean']:.3e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        mat = np.array([[grid[(b, t)]["kl_mean"] for b in layer_blocks] for t in types])
        im = axes[0].imshow(np.log10(mat), aspect="auto", cmap="magma")
        axes[0].set_yticks(range(len(types)), types)
        axes[0].set_xticks(range(0, len(layer_blocks), 2),
                           [b[1:] for b in layer_blocks][::2])
        axes[0].set_title(f"{name} block fragility (log10 KL vs fp16)")
        plt.colorbar(im, ax=axes[0])
        for d in domains:
            axes[1].plot(prof[d] / prof[d].mean(), label=d, alpha=0.8)
        axes[1].set_title(f"per-domain profiles @{t0} (normalized by domain mean)")
        axes[1].set_xlabel("layer"); axes[1].legend(fontsize=7, ncol=3)
        fig.tight_layout()
        out = ROOT / f"runs/sweepA-block-{name}/firstlook.png"
        fig.savefig(out, dpi=140)
        print(f"\nfigure -> {out}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main(sys.argv[1])
