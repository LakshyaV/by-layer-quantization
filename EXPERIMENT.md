# EXPERIMENT.md — Design

Everything here is scoped to the decisions you locked in: fine-grained maps on two small models from two families + coarse replication on an 8B; damage mechanics matched to real k-quants; validation against **stock Q4_K_M at equal file size**; Mac-only compute. Terms: **teacher-forcing** = running the model over fixed text and reading its next-token distribution at every position (no generation, fully deterministic); **KLD** = KL divergence of the damaged model's next-token distribution from the fp16 model's, averaged over positions.

## 0. Claims under test (pre-registered)

Write these down before any sweep runs; they are what the week confirms or kills.

**Paper spine (revised 2026-08-28 after external review + verification):** the headline is the **assumptions audit** — every live allocation method rests on assumptions nobody has ground-truthed at the granularity where allocation actually happens: **additivity** (C5: Hill's saturation model and CoopQ's interactions are layer-level; typed/signed tensor-level pairs are unmeasured), **proxy validity** (C8: ΔW is QuantLRM's entire signal, imatrix statistics are the ecosystem's, weighted-MSE is llama.cpp PR #15550's — none scored against measured damage), and **map stability across training history** (C6/C7: QuantLRM *assumes* relocation, nobody has measured it). C3's validated allocator with the skeptic-control arms is the proof any of it matters. C2 (task-conditioning) is a *section*, not a headline — TAQ and TASA own the layer-level versions. **Minimum publishable unit: measurement → pairs → allocator → the C3+C7 tables, end-to-end on one model pair, before any replication or expansion.**

- **C1 (structure):** Per-tensor quantization fragility has reproducible structure beyond "bigger tensors hurt more" and beyond the depth profile already published in the pruning literature — i.e., module type × depth interactions that survive size normalization.
- **C2 (task-conditioning):** Fragility maps conditioned on task domain (math CoT vs code vs factual recall vs multilingual vs general web) differ at **tensor** granularity by more than probe-resampling noise — quantified as rank-correlation between domain maps, compared against the split-half same-domain ceiling. Positioning (provenance verified 2026-08-28): TAQ's *headline* sells task-distinct layer profiles, but its own Appendix D.2 reports "87.5% of layer assignments remain the same across calibration tasks, with cross-task Spearman correlation ρ=0.757" (Fig. 8: "task-specific variation concentrates at boundary blocks"); TASA's Kendall τ≈0 is a *different* comparison (ppl-derived vs reasoning-derived rankings, i.e., metric-conditioning, not task-vs-task). The layer-level literature is internally in tension; C2 measures at the granularity where none of them looked. This is a section-level claim, not the headline.
- **C3 (validation, the part that makes it real):** A mix allocated from the measured map beats **stock Q4_K_M at ≤ its file size** on held-out data — primary: mean KLD vs fp16; guardrail: no degradation beyond CI on benchmarks — and beats it by more than a folklore+ arm and a proxy arm do (i.e., *measurement* is what's adding value, not just "any tuning beats defaults").
- **C4 (task-specialized validation, contingent on C2):** A math-allocated mix beats the code-allocated mix on math probes *and vice versa*, at equal bytes — the transfer asymmetry that makes workload-specific compression a product.
- **C5 (methods):** Isolation sensitivity (quantize one tensor, rest fp16) and marginal sensitivity (all at Q4_K, bump one) rank tensors differently/similarly by a measurable amount, and marginal scores predict joint-mix damage better (the additivity question ScaleBITS raised but didn't measure). Extension (refine-or-refute Hill, arXiv 2607.12266): at **tensor** granularity, do pairwise interactions show *typed* structure (same-layer attn↔FFN vs adjacent vs distant) or *signed* effects (error cancellation, not just uniform subadditive saturation)? Measured by the pairs mini-sweep (§5).
- **C6 (training history — co-headline):** Per-tensor fragility maps of the *same architecture* move across training stages by more than probe noise: Qwen2.5-1.5B base → Qwen2.5-Math-1.5B (domain SFT) → DeepSeek-R1-Distill-Qwen-1.5B (reasoning distillation, built on the Math checkpoint), + Qwen2.5-Coder-1.5B branch. Sub-questions: does movement concentrate in the tensors the micro-probes (§4) associate with the trained capability? Do super-weight locations move (locate per checkpoint via the Apple one-forward-pass method — no published study exists)? Either outcome is a first (verified 2026-08-26; RESEARCH.md §9).
- **C7 (training-history validation — the killer table):** On the R1-distill at equal bytes, a mix allocated from *its own* map beats a mix allocated from the *base model's* map (what current practice implicitly does) on math held-out KLD and GSM8K — and both are compared against stock Q4_K_M. If C6 shows movement but C7 shows no payoff, say so; C7 is what makes C6 matter.
- **C8 (proxy benchmark — secondary, artifact-grade):** With measured maps as ground truth, rank the sensitivity proxies the ecosystem actually uses — imatrix statistics (Σ(Act²)/z-score, eaddario's), per-tensor weighted-MSE (llama.cpp PR #15550's signal), SNR (Adaptive-Quantization), activation-noise-injection KL (TAQ's scorer), ΔW vs base (QuantLRM's signal, defined on the lineage checkpoints) — by rank correlation with measured KLD maps. No such comparison is published; documented proxy misfires (RESEARCH.md §7) prove it's needed.

## 1. Models

| Model | Layers / hidden / ffn | Attention | Embeddings | Role |
|---|---|---|---|---|
| **Qwen3-1.7B** | 28 / 2048 / 6144 | GQA 16Q:8KV | tied | Fine-grained everything; fastest iteration |
| **Llama-3.2-3B-Instruct** | 28 / 3072 / 8192 | GQA 24Q:8KV | tied | Second-family sanity check (per-block coarse only — demoted to fund the lineage) |
| **Qwen3-8B** | 36 / 4096 / 12288 | GQA 32Q:8KV | untied | Coarse replication at practical scale |

**The training lineage (C6/C7)** — four checkpoints on one chassis, the natural experiment for "does capability training move fragility":

| Checkpoint | Training treatment | Notes |
|---|---|---|
| **Qwen2.5-1.5B** | base | reference map |
| **Qwen2.5-Math-1.5B** | domain SFT on math | sequential ancestor of the distill |
| **DeepSeek-R1-Distill-Qwen-1.5B** | reasoning distillation (on the Math checkpoint) | the deployment-relevant endpoint |
| **Qwen2.5-Coder-1.5B** | domain SFT on code | orthogonal-capability branch |

All ≈1.5B, same architecture (28 layers, hidden 1536, intermediate 8960, GQA 12Q:2KV, tied — **verify in M0**, including k-quant divisibility: 1536 and 8960 are both divisible by 256 ✓, and verify the distill's ancestry claim from the R1 paper), same tokenizer family — so maps are comparable tensor-for-tensor. Each is ~3 GB fp16; a full Sweep A costs ~1.5–3 h per checkpoint.

Architecture numbers are from memory — **verify each against `config.json` in M0** before they parameterize anything. Why these: two families (different data, different tokenizers, one shared family across scale so scale effects aren't confounded with family); all fit 24 GB in fp16/bf16 (8B ≈ 16 GB weights — run with small batches, nothing else open; raise the GPU wired limit via `sysctl iogpu.wired_limit_mb` if needed); all have first-party GGUF ecosystems so the validation artifacts are directly comparable to what people actually download. Use **instruct** variants (that's what GGUF users run; startup relevance). Tensor inventory per model: 7 weight matrices per block (`attn_q/k/v/o`, `ffn_gate/up/down`) × layers, plus embedding (+ separate output head only on 8B) → ≈ 197 / 197 / 254 allocation units. Norms stay f32 (as in k-quants) and are excluded.

GQA note (confound to keep in mind): `attn_k`/`attn_v` are 2–3× narrower than `attn_q` (512–1024 columns), so bumping them is nearly free in bytes — this is *why* folklore bumps V. All per-tensor results must be reported alongside byte cost, or V-bumping conclusions are trivially confounded.

## 2. The damage mechanic — replace the fake-quant sketch

Your `fake_quant` sketch (asymmetric min/max RTN, group 128) is the right instinct but the wrong specifics: k-quants use 32-element sub-blocks inside 256-element superblocks, quantize the sub-block scales/mins themselves to 6 bits, use an iterative grid-search fit (not plain min/max), and in deployment the fit is imatrix-weighted. Rather than simulating all that and then defending the simulation, **don't simulate at all**:

**Primary mechanic — splice real dequantized weights.** For each model, run `llama-quantize --pure` once per type (Q2_K, Q3_K, Q4_K, Q6_K, Q8_0), *with the same imatrix used for the final artifacts*. Each output file contains every tensor quantized with llama.cpp's actual fitting code. Then, per sweep variant: load the fp16 model in MLX, dequantize tensor *t* from the pure-Q*k* file (via `gguf-py`), swap it in, run the probe, swap back. This eliminates the **weight-representation** gap by construction — the damaged values *are* llama.cpp's output, including its grid-search fitting and imatrix weighting. It does **not** eliminate the **runtime-arithmetic** gap: llama.cpp's k-quant kernels quantize activations (Q8_K) and accumulate in their own order, while our forward pass is an MLX fp16 matmul over dequantized weights. That gap is measured, not assumed away — M0 control: run the full `--pure` Q4_K model both ways (MLX-spliced vs. real llama.cpp inference on the same probe), compute each one's KLD to its own fp16 reference, and require the cross-stack discrepancy to be small relative to per-tensor effect sizes. If it isn't, per-tensor conclusions carry a stated caveat and the validation stage (which lives entirely inside llama.cpp) becomes the only cross-stack-free claim.

Costs: one-time ~5 quantize runs per model (minutes each) + storing ~5 files per model (≈ 2–7 GB per small model, ~25 GB for 8B — fine on 842 GB free). Two mechanical risks, both with M0 checks and fallbacks:

1. **gguf-py dequantize coverage** for k-quants — verify in M0 (5 min). Fallback: `llama-quantize --allow-requantize` from the pure-Q*k* file back to F16 produces a dequantized-GGUF whose tensors are plain f16 — read those.
2. **Layout/permutation mismatch** — llama.cpp's converter permutes Q/K projection rows for Llama-family rope; GGUF names differ from HF names (`blk.N.attn_q` vs `model.layers.N.self_attn.q_proj`). One mapping dict + un-permute function, guarded by a **two-level Q8_0 sanity gate** in M0: (a) splice the **entire** `--pure` Q8_0 model at once and require whole-model KLD ≈ noise floor (catches global layout/mapping bugs in one shot); (b) splice every tensor's Q8_0 version **individually**, per tensor class × each model family, each ≈ noise floor (~1e-4 or below; Q8_0 whole-model reference is 0.0014 on 8B). Standing rule for the whole project: **if attention tensors ever look spectacularly fragile, suspect the Q/K permutation before believing the science** — a layout bug on exactly those tensors masquerades as a discovery.

Keep your `fake_quant` (upgraded to 32/256 + 6-bit scale quantization) in the repo only as a cross-check appendix — agreement between it and spliced-real damage is itself a nice supplementary figure, and it's the escape hatch if gguf-py surprises us.

## 3. Metric

**Primary: mean token-level KLD(fp16 ‖ damaged)** over all positions of the probe set, teacher-forced. This direction penalizes the damaged model for failing to reproduce the fp16 model's distribution (llama.cpp's `--kl-divergence` uses the same orientation, so our numbers stay comparable to ecosystem practice). Precedent for KL-over-accuracy/perplexity: arXiv 2407.09141; for KL-over-MSE/SQNR as a *sensitivity* signal: arXiv 2604.13440; for why perplexity alone misleads on reasoning: TASA's "perplexity illusion" (arXiv 2607.00908).

**Computation without storing full logits.** Full-vocab logits for the baseline are enormous (151k vocab × 2 B × tokens ≈ 300 KB/token). Instead cache, per position, the fp16 model's **top-K=256 log-probs + log-sum-exp of the tail**; compute KLD over top-K exactly plus a single tail bucket. Storage ≈ 100 MB per model/probe — trivial. Known bias direction: the tail bucket underestimates KLD worst exactly where damage pushes mass outside the reference top-K, i.e., on the most-damaged variants that drive the allocator. Mitigations: **report tail mass per condition** (a rising tail mass is itself a damage signal and flags where the approximation degrades); **M0 check** — on ~2,000 positions, compare against exact full-vocab KLD computed on the fly, require < 2% relative error and re-check specifically on the highest-damage variants; and for the **final allocator inputs and validation rankings on the 1.5–1.7B models, compute exact full-vocab KLD** (at this scale the baseline distribution can be recomputed or streamed — ~17 GB per model if stored, tractable) so no allocation decision rests on the approximation.

**Secondary metrics, logged per variant at no extra cost:** top-1 flip rate (how often the argmax token changes — interpretable, cited in 2407.09141), ΔNLL on the true next tokens (the perplexity link), and the 50th/95th percentile of per-position KLD (mean-KLD can be dominated by a few catastrophic positions; report the distribution, not just the mean).

**Noise floor / effect anchors** (from llama.cpp's perplexity README, Llama-3-8B): whole-model Q4_K_M ≈ 0.031 mean KLD; Q6_K ≈ 0.0055; fp16-vs-bf16 ≈ 0.000025. Single-tensor damage will live in the 1e-5…1e-2 range; anything below the measured probe-resampling noise (see §4) is reported as "indistinguishable from zero," never ranked.

## 4. Probe set

**Composition — six domains, equal token counts, fixed seed, frozen to parquet with a SHA-256 manifest:**

| Domain | Source (train/validation splits only, never test) | Represents |
|---|---|---|
| general web | FineWeb-Edu sample | default/perplexity-style behavior |
| code | the-stack Python/JS samples | code completion |
| math CoT | GSM8K + MATH training *questions with worked solutions* | step-by-step reasoning (the TASA-sensitive axis) |
| factual QA | TriviaQA/NQ formatted Q→A | parametric knowledge recall (TAQ's "retrieval") |
| multilingual | mC4 zh/es/ar samples | non-Latin + Latin non-English (the Marchisio axis) |
| chat | UltraChat excerpts, **rendered through each model's own chat template** | instruct-mode distribution |
| **micro: induction** | synthetic — repeated random-token sequences (…A B C … A B → C) | in-context copying / induction circuitry (attention-carried) |
| **micro: factual cloze** | ParaRel/LAMA-style completions ("The capital of France is ___") | parametric recall (FFN key-value memory — Geva 2021/ROME line) |
| **micro: arithmetic** | multi-digit add/multiply with worked steps, synthetic | composite computation (the capability the Math/R1 lineage trains) |

The three micro-probe domains are what turn per-domain KLD splits into a **capability damage taxonomy** (which tensor classes carry which capability — the causal direction no one has published; nearest prior quantizes the whole model and observes where damage shows up, RESEARCH.md §9). They're synthetic, a few dozen lines to generate, and free at sweep time.

**Size:** sweep probe = **9 domains × 12 sequences × 512 tokens ≈ 55k tokens**. **Lineage comparability rule:** the cross-checkpoint comparison (C6) uses *identical raw-text probes* for all four checkpoints (same tokenizer, teacher-forced, no chat templates), so map movement can't be a formatting artifact; each checkpoint's own-template chat domain is reported separately. Held-out probe for validation = disjoint 64 sequences × 512 from fresh documents, same domains, plus Wikitext-2 for tradition. Chat-template rendering makes probes model-specific token-wise but content-identical.

**Is 48k tokens enough? Measure, don't assume (M0/M1):**
1. **Split-half stability:** Spearman rank correlation of per-tensor KLD between two disjoint 24k halves. Target ≥ 0.95 for the aggregate map; per-domain maps (8k tokens each) will be noisier — their split-half ρ defines the **noise ceiling that C2 must beat** (task differences count only if cross-domain ρ is *below* same-domain split-half ρ).
2. **Probe-growth curve:** for ~20 representative tensors, KLD at 8k/16k/32k/48k tokens; rankings should stabilize before 48k. If not, grow the probe and revise the cost table (§5) — it's linear.
3. **Position effects:** KLD vs token position sanity plot (early positions are noisier; confirm no pathology from sequence starts).

## 5. Sweep design

**Sweep A — isolation map ("quantize one, rest fp16").** For every tensor: splice its {Q2_K, Q3_K, Q4_K, Q6_K} dequant, measure probe KLD. What it gives: clean per-tensor attribution, the fragility *map* itself, per-domain splits (free: KLD subset by domain from the same forward pass), and bit-curve slopes (how fast each tensor degrades with fewer bits — some tensors will be cliff-like, some graceful; the cliff set is the interesting one).

**Sweep B — marginal map ("everything at Q4_K, move one").** Baseline: all tensors spliced to pure Q4_K (one extra "variant" measuring the joint damage). Variants per tensor: bump to Q6_K (what does *rescuing* this tensor buy, in KLD, per byte?) and drop to Q3_K (what does *sacrificing* it cost, per byte saved?). This is the decision-relevant currency — allocation operates in the quantized regime, not the clean one — and it directly feeds the allocator (§7).

**Your "inverse sweep" question:** "quantize everything except tensor *t*" is Sweep B's bump variant taken to fp16 instead of Q6_K — same information, more extreme lever. We run the Q6_K version because Q6→fp16 is nearly a no-op (whole-model Q6_K KLD ≈ 0.0055) while costing 2 bits/weight; the fp16 version adds nothing decision-relevant at deployment budgets. What A tells you that B doesn't: uncontaminated attribution (B's signal for tensor *t* rides on 196 other damaged tensors — interactions included, per-tensor credit blurred). What B tells you that A doesn't: the value of a bit *where allocation actually happens*. **The A↔B rank correlation is claim C5 and costs nothing extra.** ScaleBITS (arXiv 2602.17698) asserts B-direction is the right ground truth without measuring the gap; we measure it.

**Pairs mini-sweep (C5 extension, Qwen3-1.7B only) — run EARLY, not last.** ~200 structured (stratified, not uniform) tensor *pairs* spliced jointly at Q4_K: ~60 same-layer pairs (all within-block type combinations sampled across depth), ~50 adjacent-layer pairs (e.g., `ffn_down(i)` × `attn_v(i+1)`), ~50 distant pairs (Δlayer ≥ 8), ~40 same-type-across-depth. Interaction per pair: I(a,b) = KLD(a,b) − KLD(a) − KLD(b), normalized by KLD(a)+KLD(b); singles come free from Sweep A. Questions: is I typed (does same-layer coupling differ from distant)? Is it ever materially *negative* (cancellation)? Does Hill's uniform-saturation form fit at tensor granularity, or is there structure his layer-lattice design couldn't see? **Sequencing rationale:** the pairs result + the A↔B comparison decide *which sweep direction is the right allocator input* — running them in the first overnight (they're ~10% of it) means the answer shapes every later sweep instead of auditing them post-hoc.

**Lineage sweeps (C6).** Sweep A at {Q3_K, Q4_K} per-tensor on each of the four Qwen2.5-1.5B-chassis checkpoints (~197 × 2 ≈ 395 runs each), identical raw-text probes. Sweep B on base + R1-distill only (feeds C7's allocators). Plus, per checkpoint: super-weight localization (one forward pass, spike detection on `down_proj` inputs/outputs per the Apple method) — a table of positions across the lineage.

**Coverage per model:**
- Qwen3-1.7B: full A (197 × 4 ≈ 790 runs) + full B (1 + 197 × 2 ≈ 395 runs) + pairs (~200 runs).
- Lineage (×4 checkpoints): A at {Q3, Q4} ≈ 395 runs each; B on base + distill ≈ 395 each.
- Llama-3.2-3B: per-*block* A at {Q3, Q4} only (~58 runs — second-family sanity).
- Qwen3-8B: A per-*block* at {Q3, Q4} (72 runs) + A per-tensor at Q4 only (252 runs) + B bump-only (252 runs).

**Cost model** (calibrate the throughput number in M0 — these are planning ranges, not promises): runtime ≈ variants × 55k tokens ÷ prefill throughput + ~2 s/variant swap overhead. Assuming MLX bf16 prefill on M5 Pro of ~3–5k tok/s for the 1.5–1.7B class, ~2–3k (3B), ~1–1.5k (8B):

| Job | Runs | Est. wall-clock |
|---|---|---|
| 1.7B A+B full + pairs | ~1,385 | **5–9 h** (overnight 1) |
| Lineage A ×4 + B ×2 | ~2,370 | **9–16 h** (overnight 2, spillover to overnight 3) |
| 3B per-block sanity | ~58 | **0.5–1 h** (daytime) |
| 8B coarse | ~576 | **6–10 h** (overnight 3) |

If M0's measured throughput is > 2× worse than the low end, the cuts in PLAN.md apply (Coder branch first, then 8B per-tensor slice) — the headline claims survive every listed cut.

## 6. Confound controls (explicit, per your list)

| Confound | Why it fakes a result | Control |
|---|---|---|
| **Tensor size** | Bigger matrices accumulate more rounding error → "important" = "big" | Report KLD **and** KLD-per-parameter and KLD-per-byte-saved; fit a null model (KLD ~ log params + type nominal bpw) and show the map's structure in the *residuals*; the allocator's currency (ΔKLD/Δbyte) prices size in by construction |
| **Depth × module conflation** | "Layer 0 matters" may really be "ffn_down in layer 0 matters" | The unit *is* the tensor; report a two-way depth × module-type decomposition (heatmap rows = module, cols = layer) and marginal profiles for each |
| **GQA width asymmetry** | K/V tensors are 2–3× smaller → cheap to protect, conclusions about "V is special" may be pure economics | Byte-normalized reporting (above) + call it out explicitly when comparing to folklore's V-bump |
| **Tied embeddings** (1.7B, 3B) | Embedding damage hits both input lookup and output head at once → not comparable to untied 8B | Treat tied embedding as one allocation unit; analyze embeddings separately from block tensors in all figures; note llama.cpp bumps tied embeddings to Q6_K in Q4_K_M (observed in spike) |
| **Special-cased tensors** | k-quants already treat `output`/`token_embd` specially; our map inherits their folklore if we copy it | Embedding/output enter the sweeps like any tensor; the allocator decides their types from measurement, not inheritance |
| **Probe sampling noise** | Rankings from noise | Split-half stability (§4), duplicate-variant repeats (~20 variants run twice as a direct noise measurement), everything below noise reported as ties |
| **Additivity assumption** | Greedy allocation from per-tensor scores assumes damages add; interactions could invalidate the mix | C5: A↔B comparison + **joint check** — predicted KLD of the final mix (sum of marginals) vs its actual measured KLD, reported in the paper |
| **Simulation gap** | Fake-quant damage ≠ shipped-format damage | Eliminated by construction: spliced weights *are* llama.cpp's output (§2); Q8_0 splice sanity test catches mechanical bugs |
| **Chat-template / tokenizer artifacts in task maps** | "Task structure" that is really formatting or tokenizer differences between domains | Equal token counts per domain; headline C2 comparison uses the four English text domains (web/code/math/QA); chat and multilingual reported separately |
| **fp16 reference mismatch** | MLX-fp16 forward ≠ llama.cpp-f16 forward numerically | The map lives entirely in MLX-land (baseline and variants share the stack); validation (§7) lives entirely in llama.cpp-land (its own f16 GGUF as reference); no cross-stack KLD is ever computed |

## 7. Validation — the first-class milestone

A fragility map that doesn't allocate better than folklore is decoration. This is the go/no-go experiment.

**Allocator.** Decision variable: a k-quant type per tensor from {Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0}. Objective: minimize predicted KLD (Sweep-B marginal utilities, interpolated across types from Sweep A's bit-curves) subject to total bytes ≤ budget. Greedy by ΔKLD/Δbyte with local swap refinement — HAWQ-V3 solved the same shape with an ILP; greedy + swaps is fine at 197 variables. Exact byte accounting via `llama-quantize --dry-run` (verified today: prints per-tensor types and exact final size in seconds, no quantization performed). Q5_K's utility is interpolated (it's not in the sweeps) — acceptable; the allocator mostly trades Q3↔Q4↔Q6.

**Budgets, pre-registered (both, to prevent budget cherry-picking):**
1. **bytes(stock Q4_K_M)** — the headline comparison. Warning from today's spike: "Q4_K_M" on a tied-1B is actually 5.18 bpw effective; budgets are *measured file bytes per model*, never nominal bits.
2. **bytes(stock Q3_K_M)** — the low-bit regime, where whole-model KLD is ~10× larger and allocation has real room to matter. Expect the clearer win here; a Q4_K_M-budget win within noise but a Q3_K_M-budget win outside it is a publishable shape (state this expectation now so it doesn't become post-hoc).

**Arms (all built with the *same imatrix*, all within −1%/+0% of budget):**
- **stock**: `Q4_K_M` / `Q3_K_M` as shipped (the folklore).
- **uniform**: `--pure Q4_K` / `Q3_K` (the strawman — included only to show mixing matters at all; beating it proves nothing, per your correction).
- **folklore+**: the eaddario-style heuristic — imatrix statistics + z-scores guide the bumps (cheap to build with `llama-imatrix --show-statistics`). This arm answers "does *measurement* beat *heuristics*," which is the actual contribution.
- **measured (ours)**: map-allocated mix via `--tensor-type-file`.
- **measured-task** (C4, contingent on C2): math-allocated and code-allocated mixes at the same budget.
- **Lineage arms (C7), on DeepSeek-R1-Distill-Qwen-1.5B at both budgets:** (i) stock Q4_K_M / Q3_K_M; (ii) **base-map mix** — allocated from Qwen2.5-1.5B-base's map, applied to the distill (this is what current practice implicitly does: every public recipe is base-model folklore); (iii) **own-map mix** — allocated from the distill's own map; (iv) folklore+. Same imatrix (built on the distill), same bytes.
- **Skeptic-control arms (all at the same bytes, on the primary model at minimum):** (v) **shuffled-map** — the measured sensitivities randomly permuted across tensors, then allocated. If shuffled also beats stock, the finding is "llama.cpp's 2023 mix is bad," not "measurement helps" — this arm separates the two. (vi) **trivial-proxy** — allocate from tensor size + layer index alone (the null model as an allocator). Measurement must beat what a size/depth heuristic gets for free. (vii) **domain-imatrix** — stock Q4_K_M type allocation but with the imatrix computed on target-domain data. This is the first thing a skeptic proposes (and TASA implies calibration choice captures real gains); our allocation win must survive against it, or the honest conclusion is "smarter calibration suffices" — still a useful result, a different paper.
- **Allocable-set toggle:** at ~1.5B with a 151k vocab, `token_embd` is ~15–20% of all bytes and can dominate budget decisions — every allocation is reported **twice**: embeddings+output inside the allocable set, and frozen to their stock types with only block tensors allocable. Structural conclusions must hold in the frozen-embed view.

**Evaluation (all inside llama.cpp, on artifacts a user could download):**
- Primary: mean KLD vs the model's own f16 GGUF on the **held-out** probe via `llama-perplexity --kl-divergence` (logits file ~250–300 KB/token → keep held-out probe ≈ 30k tokens ≈ 8–9 GB, delete after each model).
- **Two levels of held-out, reported separately:** (a) *matched-workload* — fresh documents from the same nine-domain mixture the allocator saw (this is the product promise: allocate for your workload, evaluate on your workload); (b) **leave-one-domain-out (LODO)** — allocate on eight domains, evaluate on the ninth, rotating. LODO is simultaneously the generalization test and the actual product question (does a customer need their own map, or does any decent map transfer?). Probe hygiene for the guardrails: math-CoT probe text comes from MATH-train only, so **GSM8K test remains a genuinely external generative eval**, not an in-distribution echo of the probe mix.
- Secondary: Wikitext-2 perplexity; HellaSwag (`--hellaswag`) and an ARC-style set (`--multiple-choice`).
- Generative guardrail (because teacher-forced metrics can miss reasoning damage — TASA's illusion, and the COLM 2025 result that low-bit breaks math first): **GSM8K test subset, 200 problems, greedy decoding, exact-match**, via `llama-cli`/server script per arm. For C4: + 100 HumanEval-style completions for the code axis.
- **Uncertainty:** paired bootstrap over held-out sequences (resample sequences, recompute mean KLD per arm, 10k resamples) → 95% CI on the *difference* measured-vs-stock. Same for GSM8K (bootstrap over problems).

**Pre-registered success criteria for C3:** measured beats stock on held-out KLD with the bootstrap CI excluding zero **and** ≥ 10% relative KLD reduction at the same bytes (or, equivalently stated, ≥ 3% fewer bytes at equal KLD via a small budget sweep), **and** no benchmark/GSM8K regression beyond CI, **and** measured > folklore+ (else the honest conclusion is "heuristics suffice"). For C4: the transfer matrix must show the diagonal (task-matched mixes) beating off-diagonal by CI-excluded margins on the task metrics, not just on KLD. **For C7:** own-map beats base-map on the distill's math held-out KLD and GSM8K-200 with CI excluding zero — that's the sentence the paper and the pitch both hang on.

## 8. Conditional-fragility analyses (C2/C4/C6/C7/C8)

From Sweep A's per-domain KLD splits (no extra compute):
1. **Map divergence:** Spearman/Kendall correlation matrix between the six domain maps at tensor granularity; same at layer granularity (direct comparison to TAQ's 87.5%/ρ=0.757 — do we reproduce their invariance when we aggregate up, while seeing structure below it?).
2. **Where the differences live:** rank tensors by cross-domain variance (normalized); test whether high-variance tensors cluster (module type? depth band? the super-weight/massive-activation layers from the literature?). This is figure 2 of the paper and the mechanistic garnish.
3. **Effect-size honesty:** every divergence statistic reported against the same-domain split-half ceiling (§4). "Maps differ" claims only where cross-domain ρ < split-half ρ by CI.
4. **C4 build + transfer matrix** (if C2 passes): math-mix and code-mix at the Q3_K_M budget (bigger headroom), evaluated on both domains' held-out probes + GSM8K/HumanEval-mini → 2×2 (or 3×3 with QA) transfer table. This is the startup's existence proof or its obituary.
5. **Cross-stage divergence (C6):** the same divergence machinery as (1), applied across the four lineage checkpoints instead of across probe domains — rank correlation of maps base↔Math↔Distill↔Coder against the same split-half ceiling, at tensor and layer granularity. Then the interpretive cross: do the tensors that *moved most* between base and Math/Distill coincide with the tensors the **arithmetic micro-probe** flags as arithmetic-carrying? (That's the "training relocates fragility toward capability-carrying tensors" figure — the paper's centerpiece if it holds.) Report the two conditioning axes against each other: does training history move maps more or less than probe task does?
6. **Super-weight tracking:** the located super-weight coordinates per checkpoint, one table. Scope honestly (verified 2026-08-28): arXiv 2607.08733 already measured persistence under **vanilla LoRA** — where `down_proj`, the tensor holding the super weights, is *left completely untrained*, so persistence is near-tautological (their Table 9: −0.03% magnitude change, rank retained). Persistence under **full-parameter** interventions — the Math SFT, R1 distillation, and Coder continued-pretraining in our lineage — remains unmeasured. This is a footnote-sized supporting observation, not a claim; cite 2607.08733 when reporting it.
7. **Capability damage taxonomy:** per tensor class (module type × depth band), which micro-probe's KLD inflates most when it's quantized — the causal-direction complement to arXiv 2604.19884/2505.13963 (who quantize everything and watch where damage appears).
8. **Proxy benchmark (C8):** Spearman of each ecosystem proxy (imatrix Σ(Act²)/z-score, weighted-MSE, SNR, TAQ-style noise-injection KL, ΔW-vs-base on lineage checkpoints) against the measured Sweep-A map, per model; plus each proxy's regret when its ranking drives the §7 allocator (allocate-by-proxy vs allocate-by-measurement at equal bytes, evaluated on held-out KLD). One table, directly citable by every tool author in RESEARCH.md §7.

## 9. How this could be wrong — failure modes and the control that catches each

1. **The map is noise.** → Split-half ρ, duplicate-run noise estimates, below-noise = tie (§4, §6). Kill signal: aggregate split-half ρ < 0.8 even at 48k tokens.
2. **The map is a size artifact.** → Null-model residualization; per-byte currency (§6). Kill signal: log-params alone explains > ~90% of map variance.
3. **The map is folklore rediscovered.** → folklore+ arm; also test how well the `use_more_bits` rule alone predicts our map. If Spearman(map, folklore rule) > split-half ceiling, we've measured what ikawrakow eyeballed — say so and lean on C2/C5.
4. **One-at-a-time doesn't predict joint damage (non-additivity).** → C5 joint check: predicted-vs-actual KLD of every built mix; Sweep B currency insulates the allocator. Kill signal: predicted and actual mix KLD rank-disagree across arms.
5. **The allocation overfits the probe distribution.** → Held-out probe from fresh documents; benchmarks + generative GSM8K as distribution-shifted guardrails; two pre-registered budgets.
6. **Mechanical splice bugs masquerade as fragility.** → Q8_0-splice ≈ noise-floor test per tensor class per family (§2); any violation is a bug hunt, not a result.
7. **Task differences are formatting artifacts.** → English-text-only headline comparison; template/multilingual reported separately (§6).
8. **The win is within measurement noise.** → Paired bootstrap CIs on differences; the 0.000025 fp16-vs-bf16 floor as an absolute sanity anchor; pre-registered ≥10% relative threshold so "statistically significant but pointless" can't sneak through.
9. **Q4_K_M is near-optimal and everything ties at that budget.** → The Q3_K_M budget is pre-registered as the regime where differences have room; a both-budgets tie is a real (negative) finding about how much folklore left on the table — publishable honesty, startup-lethal, and exactly what Day-4's decision point is for.
10. **Instruct-model thinking modes contaminate probes (Qwen3).** → Probes are teacher-forced raw text (no generation, no thinking tokens); chat domain uses the template with thinking disabled; generative GSM8K runs use the model's default mode, same for every arm.
11. **Cross-checkpoint KLD *levels* aren't comparable (C6).** Each checkpoint's KLD is measured against *its own* fp16 reference on a probe distribution the checkpoints fit differently well — so absolute damage levels confound training-stage effects with distribution fit. → C6 claims are made only on **within-checkpoint structure** (rankings, normalized maps) compared *across* checkpoints via rank correlation — never on raw KLD magnitudes across checkpoints.
12. **Lineage maps don't move.** The chat-tuning precedent (massive activations static) says this is a live possibility; distillation is a much larger intervention, but it's a bet. → Pre-registered as informative either way: near-invariance is the first *localized* training-stage invariance result (pairs with ICLR 2026's whole-model stage study), C7's base-map arm then simply ties the own-map arm, and the startup's re-mapping thesis dies for the price of one week. The Day-3 checkpoint (PLAN.md G2) reads this number before further lineage investment.
13. **Pair interactions look nonzero because KLD is noisy at small magnitudes.** → Interaction I(a,b) is only interpreted for pairs whose singles are individually above the noise floor; duplicate-pair repeats estimate interaction-scale noise; Hill's saturation model is the pre-registered null to beat, not "zero interaction."
14. **The lineage is confounded — differences are real but unattributable.** The public checkpoints differ in *kind and scale* of intervention (Math = SFT by Alibaba; R1-Distill = full-parameter distillation by DeepSeek, initialized from the Math checkpoint; Coder = continued pretraining on trillions of tokens), so "training for capability X moved fragility toward X-carrying tensors" cannot be claimed causally from them — only "real-world fine-tunes have materially different maps" (which is what C7's practical question needs). → Stated as a limitation in the paper; the causal version needs **matched controlled fine-tunes from one base varying only domain — and they must be full-parameter**, because 2607.08733 shows vanilla LoRA leaves the most fragility-relevant tensors untouched (a LoRA control would show "no relocation" by construction). Controlled full-FT runs at 1.5B are feasible on this hardware but are scheduled as the fast-follow (PLAN.md), not week one.
15. **The whole-model result is "stock is bad," not "measurement works."** → The shuffled-map and trivial-proxy arms (§7) are the direct controls; the claim "measurement adds value" is only made if measured > {shuffled, trivial, folklore+, domain-imatrix} at equal bytes.
