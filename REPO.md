# REPO.md — Structure, dependencies, caching, reproducibility

Guiding rule: the novel code is ~600–900 lines (splice + sweep loop + KLD + allocator + plots). Everything else is reused. Anywhere you find yourself writing more than that, check the reuse table first.

## Layout

```
by-layer-quantization/
├── RESEARCH.md  EXPERIMENT.md  PLAN.md  REPO.md
├── pyproject.toml            # uv-managed; Python 3.12 (mlx wheel support — not 3.14)
├── uv.lock
├── Makefile                  # one target per milestone job; `make figures`
├── configs/
│   ├── models.yaml           # HF repo + pinned revision + arch facts (verified M0)
│   ├── probes.yaml           # domain sources, counts, seq len, seed
│   └── sweeps/*.yaml         # sweep specs: model, tensors, types, probe hash
├── src/fragmap/
│   ├── gguf_io.py            # gguf-py load/dequant, HF↔GGUF name map, un-permute, Q8_0 sanity
│   ├── splice.py             # swap tensor into MLX model / restore
│   ├── probes.py             # build + freeze probe parquet, chat templating
│   ├── kl.py                 # top-K logprob cache, KLD/flips/ΔNLL per position
│   ├── sweep.py              # the loop: single-tensor and pair variants → results.jsonl (resumable)
│   ├── analyze.py            # maps, split-half, null model, domain/lineage divergence, A↔B, pair typology
│   ├── allocate.py           # greedy ΔKLD/Δbyte + swaps; emits tensor_types.txt
│   ├── superweights.py       # one-pass super-weight localization per checkpoint (C6)
│   ├── proxies.py            # ecosystem proxy scores (imatrix stats, wMSE, SNR, noise-KL, ΔW) vs measured maps (C8)
│   ├── llamacpp.py           # subprocess wrappers: quantize/dry-run/imatrix/perplexity/cli
│   ├── fakequant.py          # your sketch, 32/256-faithful — cross-check only
│   └── cli.py                # typer entry: fragmap sweep|analyze|allocate|validate
├── scripts/                  # thin: download_models.sh, build_pure_quants.sh, gsm8k_eval.py
├── data/probes/              # frozen parquet + manifest.json (sha256, sources, seed)  [gitignored, manifest committed]
├── models/                   # fp16 GGUFs, pure-Q* files, imatrix per model            [gitignored]
├── caches/topk/              # baseline top-K logprobs per (model, probe_sha)           [gitignored]
├── runs/<run_id>/            # results.jsonl + meta.json + log                          [results committed]
├── results/maps/*.parquet    # final per-tensor maps                                    [committed]
├── analysis/*.ipynb          # or plain scripts; figures out to paper/figs
├── paper/                    # draft.md → LaTeX later; figs/
└── demo/                     # M6 CLI polish + static HTML heatmap
```

## Dependencies

- **Python (uv):** `mlx`, `mlx-lm` (model load + forward on Apple silicon — the M5 Pro's neural accelerators make this the fastest local stack), `gguf` (gguf-py: read + dequantize k-quant tensors), `transformers` (tokenizers/chat templates only), `datasets` (probe sources), `numpy`, `pyarrow`, `scipy`, `matplotlib`, `typer`, `pyyaml`. No PyTorch in the main path (MLX covers it; keep a torch-MPS cross-check script optional).
- **System (brew):** `llama.cpp` (installed today: formula 0.3.0). **Pin it:** record `brew list --versions llama.cpp` + `llama-quantize --help | head -1` output in every run's `meta.json`; don't `brew upgrade` mid-week. All quantize/imatrix/perplexity/generation work goes through these binaries via `llamacpp.py` wrappers with full command lines logged.

## Reuse vs build (flagging where you'd otherwise rebuild standard things)

| Need | Use (don't build) |
|---|---|
| Quantization kernels/fitting | `llama-quantize` (+ `--pure`, `--tensor-type-file`, `--dry-run`) |
| Dequantization | gguf-py; fallback `llama-quantize --allow-requantize → F16` |
| Importance-weighted rounding | `llama-imatrix` (one file per model, shared across all arms) |
| KLD/PPL/benchmark eval of final GGUFs | `llama-perplexity` (`--kl-divergence`, `--hellaswag`, `--multiple-choice`) |
| Model forward passes for sweeps | `mlx-lm` |
| Probe corpora | HF `datasets` (FineWeb-Edu, the-stack, GSM8K/MATH, TriviaQA, mC4, UltraChat) |
| Generative GSM8K scoring | thin script over `llama-server`; steal extraction regexes from lm-eval-harness rather than importing the whole harness |
| Heuristic baseline arm | `llama-imatrix --show-statistics` output (eaddario's exact ingredients) |

What you legitimately build: `splice.py`, `kl.py`, `sweep.py`, `allocate.py`, `analyze.py`. That's the whole novel surface.

## Result caching / crash-safety (the "never re-run a 6-hour sweep" design)

- **Content-addressed variant keys:** `key = sha1(model_id @ revision, tensor_name, quant_type, sweep_mode[A|B], probe_sha, harness_version)`. One JSON line per completed variant appended atomically (write temp, `os.replace`) to `runs/<run_id>/results.jsonl` with per-position summaries (mean/median/p95 KLD overall + per domain, flips, ΔNLL) — never raw logits.
- **Resume is the default:** `fragmap sweep` scans existing keys and skips them; a crash costs at most one variant (~30 s). Killing and restarting is always safe.
- **Expensive one-time artifacts cached on disk, keyed by hash:** baseline top-K caches (`caches/topk/<model>@<rev>-<probe_sha>.npz`, ~100 MB each), pure-quant GGUFs, imatrix files. Deleting a cache is always safe (recomputed on demand).
- **Ordering for early signal:** the sweep queue is interleaved (round-robin across depth and module type, both sweeps' duplicates first), so a partially completed overnight run still yields an unbiased partial map in the morning rather than "layers 0–9 only."
- llama.cpp logits files (`--kl-divergence-base`) are scratch: generate per model at validation time (~8–9 GB), evaluate all arms, delete. Never committed, never cached.

## Reproducibility (what makes a run a run)

Every `runs/<run_id>/meta.json` records: git SHA of this repo (+ dirty flag), config hashes, model HF revisions, probe SHA, `mlx`/`mlx-lm`/`gguf` versions, llama.cpp version string, macOS version, and the exact command line. Additional rules:

- **Determinism:** probes frozen (parquet + SHA committed as manifest); teacher-forcing everywhere in the map (no sampling anywhere in Sweeps A/B); generative evals greedy (temp 0) with logged commands; seeds fixed in configs for anything that samples (probe construction, bootstrap). MLX float nondeterminism across runs is negligible relative to the measured noise floor — the duplicate-variant repeats in M0/M2 quantify actual run-to-run noise rather than assuming it.
- **Figures are functions of committed data:** `make figures` regenerates every paper figure from `results/*.parquet` — no hand-edited plots.
- **Artifacts for release (M6/paper):** the three maps (parquet + HTML), each arm's `tensor_types.txt` + exact `llama-quantize` command, imatrix files, and probe manifests — enough for anyone to rebuild every GGUF byte-identically from the public fp16 weights.
- **.gitignore:** `models/`, `caches/`, `data/probes/*.parquet` (manifests committed), `runs/**/log`, logits scratch. Committed: all code, configs, manifests, `results/`, run metadata, paper.
