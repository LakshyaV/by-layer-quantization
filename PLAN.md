# PLAN.md — Milestones

**Assumptions:** ~5–6 focused days (≈ 35–45 human-hours). If you're actually evenings-only, use the cut list at the bottom — the plan degrades gracefully. Compute estimates are for the M5 Pro 24 GB and are **planning ranges until M0 measures real throughput**; every heavy job is designed to run overnight and resume from cache after a crash (see REPO.md). Estimated totals: **~10–14 h human-attended work + ~20–30 h unattended compute across 3 overnights.**

**Already done (today, during planning):** the load-bearing mechanical risk is retired. Verified on this machine with brew llama.cpp 0.3.0: `--tensor-type` regex overrides apply and log correctly on a real model (Llama-3.2-1B, 20 overrides), `--tensor-type-file` and `--pure` exist, `--dry-run` prints exact per-tensor allocation + final size in seconds, a custom-mix GGUF quantizes/loads/generates, and `llama-perplexity` has `--kl-divergence`, `--hellaswag`, `--multiple-choice`. Also observed: stock Q4_K_M on the tied-1B is **5.18 effective bpw** (budget in bytes, never nominal bits), and the folklore bump pattern is exactly `use_more_bits` layers {0,1,4,7,10,13,14,15}.

---

## M0 — Rig + calibration (Day 1 morning; ~4 h human, ~1 h compute)

Build the minimal harness and measure everything the cost model assumed.

**Tasks:** uv project skeleton (REPO.md layout); download models — Qwen3-1.7B/8B, Llama-3.2-3B-Instruct, **and the lineage: Qwen2.5-1.5B, Qwen2.5-Math-1.5B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen2.5-Coder-1.5B** (pin revisions); verify `config.json` architecture numbers from EXPERIMENT.md §1 incl. lineage chassis identity + distill ancestry; build the probe set (§4, now 9 domains incl. the three synthetic micro-probes) and freeze to parquet + SHA; generate imatrix per model; produce `--pure` Q2_K/Q3_K/Q4_K/Q6_K/Q8_0 files per small model (lineage: Q3/Q4/Q6/Q8_0 suffice); **verify gguf-py dequantizes k-quants** (fallback: requantize-to-F16 path, EXPERIMENT.md §2); MLX weight-splice round-trip; **two-level Q8_0-splice sanity gate** — full-model `--pure` Q8_0 splice ≈ noise floor, then every tensor class individually × Qwen3, Llama-3.2, and Qwen2.5 chassis (standing rule: attention looking spectacularly fragile = suspect the Q/K permutation first); **cross-stack arithmetic control** — full `--pure` Q4_K evaluated both as MLX-spliced weights and as real llama.cpp inference, discrepancy must be small vs per-tensor effects (this bounds the runtime-arithmetic gap the splice method does not eliminate); top-K=256 KLD vs exact-KLD agreement check (< 2% rel. error on 2k positions); baseline top-K logit caches (1.7B + the four lineage checkpoints — they share probe text, so this is five cheap passes); **super-weight localization script** (spike detection on `down_proj` activations, one forward pass — run on all four lineage checkpoints, table goes straight into the paper); **measure prefill throughput** for the 1.5–1.7B class, 3B, and 8B and rewrite the §5 cost table with real numbers.

**Artifact:** `runs/m0-calibration.md` — measured tok/s, Q8_0 sanity table, top-K error, revised sweep ETAs.
**Gate G0 (mechanics):** all checks green. A Q8_0-splice violation or gguf-py failure is a *bug/fallback* decision, not a project decision.

## M1 — First-look map on Qwen3-1.7B (Day 1 afternoon; ~2 h human, ~1–2 h compute)

Per-block Sweep A ({Q3_K, Q4_K} × 28 blocks + embedding = 58 runs) with per-domain KLD splits, plus the split-half stability numbers. Plot the first heatmap and the six domain profiles.

**Artifact:** first-look heatmap + stability memo.
**Gate G1 — earliest kill point (end of Day 1):**
- *Continue* if the block map shows stable structure beyond size (split-half ρ ≥ ~0.9 at block level) **and** at least a hint of either (a) residual structure after size/depth normalization or (b) cross-domain divergence below the split-half ceiling.
- *Kill/pivot* if the map is flat-plus-first/last-layers only **and** all six domain profiles are rank-identical within noise. That would mean C1/C2 are dead at block level; tensor level rarely rescues a dead block level. The pivot options at this point: (i) pure C3 engineering week (measured-map GGUF tool + blog post, no paper), or (ii) stop, publish the negative note, spend the week elsewhere. Decide deliberately, don't drift.

## M2 — Full fine-grained sweeps: deep-dive model + lineage (Days 2–3; ~5 h human, overnight compute ×2)

Overnight 1: Qwen3-1.7B — **pairs mini-sweep and Sweep-B first in the queue** (their result decides which direction feeds the allocator before the bulk of compute is spent), then full Sweep A (197 × 4) + Sweep B (1 + 197 × 2). Daytime 2: analysis — tensor-level maps, per-domain divergence vs ceilings (C2 numbers land here), capability-taxonomy first read (micro-probe splits), A↔B correlation + pair-interaction typology vs Hill's saturation null (C5), null-model residualization (C1), noise duplicates; Llama-3.2-3B per-block sanity sweep (~1 h, daytime). Overnight 2: **lineage Sweep A ×4 checkpoints** ({Q3, Q4} each) + Sweep B on base + distill (spillover to overnight 3 if throughput is at the low end). Day 3: cross-stage divergence analysis (C6) — map movement base↔Math↔Distill↔Coder vs the split-half ceiling, moved-tensors × arithmetic-micro-probe overlay, super-weight movement table; cross-family rank check vs the 3B blocks.

**Artifacts:** `results/maps/{qwen3-1.7b, qwen2.5-1.5b-base, -math, -r1distill, -coder}.parquet` + the C1/C2/C5/C6 analysis notebook + draft figures 1–3 (task divergence, lineage movement, pair typology).
**Gate G2 (end of Day 3): the headline decision, now two-axis.** Data answers both: does fragility move with *task* (C2) and with *training history* (C6)? Lead with whichever axis shows the stronger effect vs its ceiling; "fragility is conditional" covers both if both land. If **neither** moves beyond noise: the paper thins to C3+C5+C8 with two first-of-their-kind invariance negatives — decide deliberately whether that (plus the tool) is worth Days 4–6, or stop.

## M3 — Allocation + validation (Day 4; ~4 h human, ~4–6 h compute)

Build the allocator (greedy ΔKLD/Δbyte + swaps, `--dry-run` byte oracle). Construct arms per EXPERIMENT.md §7 at both budgets for **Qwen3-1.7B** (C3: stock / uniform / folklore+ / measured / **shuffled-map / trivial-proxy / domain-imatrix**, each reported with embeddings in and out of the allocable set) and for **the R1-distill** (C7: stock / base-map / own-map / folklore+ — quantize runs are minutes each). Evaluation at two levels: matched-workload held-out and **leave-one-domain-out**. Evaluate: held-out KLD via `llama-perplexity --kl-divergence` (mind the ~6–9 GB logits file per model — generate, evaluate all arms, delete), Wikitext-2 PPL, HellaSwag/ARC, GSM8K-200 greedy per arm (math held-out probes weighted for the distill). Bootstrap CIs on differences.

**Artifacts:** the C3 validation table and the **C7 transfer table** (own-map vs base-map vs stock on the distill) — the two core result tables of the paper, and the pitch slide.
**Gate G3 (end of Day 4): C3 + C7 verdicts.** Measured beats stock and folklore+ per the pre-registered criteria → C3 real; own-map beats base-map on the distill → C7 real and the startup wedge has its existence proof. Tie at Q4_K_M budget but win at Q3_K_M → reframe honestly around the low-bit regime. Loss to folklore+ everywhere → the map doesn't pay; the paper becomes C2/C5/C6/C8 mapping-and-methods, and the startup leg is gone. **This is the go/no-go you asked to pull early — it lands on Day 4, and G1 on Day 1 is the cheap preview of it.**

## M4 — Scale, task-specialized validation, proxy benchmark (Day 5; ~4 h human, overnight ×1 + ~2–3 h)

Overnight 3: Qwen3-8B coarse sweeps (per-block A, per-tensor A at Q4, B bump-only) + any lineage spillover. Daytime: 8B validation arms at Q4_K_M-budget (the "does it survive scale" check, single budget); if C2 passed G2 — math-mix and code-mix for 1.7B at Q3_K_M budget and the C4 transfer matrix (held-out per-domain KLD + GSM8K + HumanEval-mini per cell); **C8 proxy benchmark** (~half day, no new sweeps: compute imatrix-stats / weighted-MSE / SNR / noise-injection / ΔW scores for models already mapped, rank-correlate against measured maps, plus one allocate-by-proxy arm on 1.7B for regret-at-equal-bytes).

**Artifacts:** 8B replication table; C4 transfer matrix; C8 proxy table.

## M5 — Writeup (Day 6; ~6 h human)

Paper draft in `paper/`: intro framed per RESEARCH.md §8 (cite TAQ/TASA/KL-Lens as layer-level/proxy prior art, EAddario/Thireus/EvoPress as ecosystem baselines — the related-work section is already 80% written in RESEARCH.md); methods from EXPERIMENT.md; the validation table; the map figures; the C5 methods result; honest limitations (Mac-only, ≤8B, six domains). Target shape: workshop-length paper (ENLSP-style efficiency workshop or arXiv preprint first) — pick the venue after seeing which claims survived.

**Artifact:** compilable draft + all figures reproducible from `make figures`.

## Fast-follow (post-week, pre-submission-of-final-version): controlled fine-tunes

The public lineage is ecologically valid but confounded (different orgs, data, intervention types — EXPERIMENT.md §9.14): the causal claim "training for X relocates fragility" needs 2–3 **matched full-parameter fine-tunes** from one base varying only domain (math vs code vs generic, same tokens/recipe/steps). Full-parameter, not LoRA — 2607.08733 shows LoRA leaves `down_proj` untouched, so a LoRA control would show no relocation by construction. Feasible at 1.5B on this hardware (frozen embeddings + memory-lean optimizer, a few hours per run); scheduled after the week's measurement pipeline exists, before the archival version of the paper.

## M6 — Demo tool (Day 7 or parallel slack; ~4 h human) [stretch]

`fragmap` CLI: point at a model → emits the fragility map (static HTML heatmap, per-domain toggle) → `fragmap allocate --budget <bytes|Q4_K_M>` emits a `--tensor-type-file` + exact `llama-quantize` command line → published maps + recipes for the three models on HF/GitHub. This is the startup demo and the r/LocalLLaMA distribution artifact. Cut first if time runs out — the paper artifacts matter more, and the tool is a mechanical wrapper around code that already exists by M3.

---

## Decision points, collected

| Gate | When | Question | Kill/pivot condition |
|---|---|---|---|
| G0 | Day 1 noon | Do the mechanics work? | Bug-fix or fallback, never a project kill |
| **G1** | **Day 1 evening** | **Any structure worth a week?** | Flat map + identical domain profiles → stop or pivot to engineering-only |
| G2 | Day 3 | Which conditioning axis leads: task (C2) and/or training history (C6)? | Neither moves beyond noise → paper thins to C3+C5+C8 with two first-of-their-kind invariance negatives; decide stop vs tool-week deliberately |
| **G3** | **Day 4** | **Does measurement beat folklore at equal bytes (C3), and does own-map beat base-map on the distill (C7)?** | Loss to folklore+ everywhere → paper thins to mapping+methods, startup leg gone |

## Compute ledger (revise after M0)

| Job | Est. wall-clock |
|---|---|
| M0 one-time (quantize-pure files ×7 models, imatrix, caches, super-weight passes) | ~1.5–2 h |
| M1 first-look | 1–2 h |
| Overnight 1 (1.7B full A+B + pairs) | 5–9 h |
| Overnight 2 (lineage A ×4 + B ×2; spillover allowed) | 9–16 h |
| 3B per-block sanity (daytime) | 0.5–1 h |
| M3 validation evals (1.7B C3 + distill C7) | 4–6 h |
| Overnight 3 (8B + lineage spillover) + M4 evals + C8 | 9–14 h |
| **Total** | **~30–50 h, ≥70% unattended** |

Disk: models ≈ 40 GB fp16 (incl. four ~3 GB lineage checkpoints) + pure-quant files ≈ 45 GB + logits scratch ≤ 9 GB at a time + caches < 2 GB → ~95 GB peak of your 842 GB free.

## Cut list (if evenings-only or throughput disappoints, in order)

1. Drop M6 (tool) and the C4 HumanEval cells (keep GSM8K).
2. Lineage: drop the Coder branch (keeps base → Math → Distill, the sequential spine); Sweep B on distill only.
3. 8B: per-block sweeps only + single-budget validation (keeps the scale claim, coarser).
4. Drop the pairs mini-sweep (C5 reverts to A↔B + joint check only) and drop Q2_K from all sweeps.
5. Drop the 3B sanity sweep (single-family caveat goes in limitations) and the multilingual + chat domains (C2 becomes English-only).
6. Never cut: G1 first-look, 1.7B full sweeps, the base + distill lineage pair with the C7 arms, M3 validation with folklore+ arm, the micro-probe domains (they're free), bootstrap CIs.
