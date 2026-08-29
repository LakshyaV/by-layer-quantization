"""KL divergence of a damaged model from the fp16 baseline, via a top-K + tail-bucket cache.

Baseline cache per (model, probe): for every position, the fp16 model's top-K token ids,
their log-probs, the tail log-mass, and the NLL of the true next token. Variant evaluation
gathers the variant's log-probs at the baseline's top-K ids; the tail is one bucket.
KL here is KL(P_fp16 || Q_variant), matching llama.cpp's --kl-divergence orientation.
Tail-bucket KL is a lower bound; exact_kl_check() quantifies the gap (M0 gate G-D).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mlx.core as mx

# K=1024: rel err vs exact KL measured at 0.73% on natural text (m0_topk_diag, 2026-08-29);
# random/flat-distribution sequences are unsuitable for top-K KL at any K — probe domains
# must keep baseline tail mass small (logged as q_tail_mean per variant).
TOPK = 1024
EPS = 1e-10


@dataclass
class BaselineCache:
    idx: np.ndarray        # (N, L-1, K) int32 — baseline top-K token ids per position
    logp: np.ndarray       # (N, L-1, K) float32 — baseline log-probs at idx (sorted desc)
    tail_logp: np.ndarray  # (N, L-1) float32 — log of baseline tail mass
    nll: np.ndarray        # (N, L-1) float32 — baseline NLL of true next token

    def save(self, path: str) -> None:
        np.savez_compressed(path, idx=self.idx, logp=self.logp,
                            tail_logp=self.tail_logp, nll=self.nll)

    @staticmethod
    def load(path: str) -> "BaselineCache":
        z = np.load(path)
        return BaselineCache(z["idx"], z["logp"], z["tail_logp"], z["nll"])


def _batch_logprob_parts(logits: mx.array, tokens: mx.array, idx: mx.array | None, k: int):
    """Per-position pieces used by both cache building and variant scoring.

    logits: (B, L, V). Positions scored: 0..L-2 predicting tokens 1..L-1.
    Returns numpy (idx, logp_at_idx, tail_logp, nll). If idx is None, top-K is computed.
    """
    lg = logits[:, :-1, :].astype(mx.float32)
    lse = mx.logsumexp(lg, axis=-1, keepdims=True)
    logp_all = lg - lse                                     # (B, L-1, V)
    if idx is None:
        part = mx.argpartition(logp_all, kth=-k, axis=-1)[..., -k:]
        vals = mx.take_along_axis(logp_all, part, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        idx = mx.take_along_axis(part, order, axis=-1)
    logp_at = mx.take_along_axis(logp_all, idx, axis=-1)     # (B, L-1, K)
    covered = mx.sum(mx.exp(logp_at), axis=-1)
    tail = mx.log(mx.maximum(1.0 - covered, EPS))
    nxt = tokens[:, 1:][..., None]
    nll = -mx.take_along_axis(logp_all, nxt, axis=-1)[..., 0]
    mx.eval(idx, logp_at, tail, nll)
    return (np.array(idx, dtype=np.int32), np.array(logp_at, dtype=np.float32),
            np.array(tail, dtype=np.float32), np.array(nll, dtype=np.float32))


def build_baseline_cache(model, tokens: np.ndarray, batch_size: int = 8,
                         k: int = TOPK) -> BaselineCache:
    outs = ([], [], [], [])
    for s in range(0, tokens.shape[0], batch_size):
        batch = mx.array(tokens[s:s + batch_size])
        parts = _batch_logprob_parts(model.logits(batch), batch, None, k)
        for acc, p in zip(outs, parts):
            acc.append(p)
    return BaselineCache(*[np.concatenate(a, axis=0) for a in outs])


@dataclass
class VariantMetrics:
    """Per-sequence aggregates (for split-half / bootstrap downstream)."""
    kl_sum: np.ndarray      # (N,) sum of per-position KL
    kl_max: np.ndarray      # (N,)
    n_pos: int
    flip_sum: np.ndarray    # (N,) count of top-1 changes
    dnll_sum: np.ndarray    # (N,) sum of (variant NLL - baseline NLL)
    q_tail_sum: np.ndarray  # (N,) sum of variant tail mass (approximation-quality signal)

    def summary(self) -> dict:
        n = self.n_pos
        return {
            "kl_mean": float(self.kl_sum.sum() / (len(self.kl_sum) * n)),
            "kl_seq_mean": (self.kl_sum / n).tolist(),
            "kl_max": float(self.kl_max.max()),
            "flip_rate": float(self.flip_sum.sum() / (len(self.flip_sum) * n)),
            "dnll_mean": float(self.dnll_sum.sum() / (len(self.dnll_sum) * n)),
            "q_tail_mean": float(self.q_tail_sum.sum() / (len(self.q_tail_sum) * n)),
        }


def score_variant(model, tokens: np.ndarray, cache: BaselineCache,
                  batch_size: int = 8) -> VariantMetrics:
    N = tokens.shape[0]
    kl_sum = np.zeros(N); kl_max = np.zeros(N); flips = np.zeros(N)
    dnll = np.zeros(N); qtail = np.zeros(N)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        batch = mx.array(tokens[s:e])
        idx = mx.array(cache.idx[s:e])
        _, logq_at, q_tail_logp, nll_q = _batch_logprob_parts(
            model.logits(batch), batch, idx, cache.idx.shape[-1])
        p = np.exp(cache.logp[s:e])                     # (b, L-1, K)
        p_tail = np.exp(cache.tail_logp[s:e])
        kl_pos = (p * (cache.logp[s:e] - logq_at)).sum(-1) \
            + p_tail * (cache.tail_logp[s:e] - q_tail_logp)
        kl_pos = np.maximum(kl_pos, 0.0)
        kl_sum[s:e] = kl_pos.sum(-1)
        kl_max[s:e] = kl_pos.max(-1)
        flips[s:e] = (np.argmax(logq_at, -1) != 0).sum(-1)  # baseline argmax is idx[...,0]
        dnll[s:e] = (nll_q - cache.nll[s:e]).sum(-1)
        qtail[s:e] = np.exp(q_tail_logp).sum(-1)
    return VariantMetrics(kl_sum, kl_max, tokens.shape[1] - 1, flips, dnll, qtail)


def exact_kl_check(model_p, model_q, tokens: np.ndarray, cache: BaselineCache,
                   n_positions: int = 2000, batch_size: int = 2) -> dict:
    """Gate G-D: exact full-vocab KL vs the top-K+tail estimate on a position subset."""
    approx = score_variant(model_q, tokens, cache, batch_size=batch_size)
    n_seq = max(1, n_positions // (tokens.shape[1] - 1))
    sub = tokens[:n_seq]
    exact_sum = 0.0
    for s in range(0, n_seq, batch_size):
        batch = mx.array(sub[s:s + batch_size])
        lp = model_p.logits(batch)[:, :-1, :].astype(mx.float32)
        lq = model_q.logits(batch)[:, :-1, :].astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        lq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
        kl = mx.sum(mx.exp(lp) * (lp - lq), axis=-1)
        mx.eval(kl)
        exact_sum += float(np.array(kl).sum())
    n_pos = n_seq * (tokens.shape[1] - 1)
    exact_mean = exact_sum / n_pos
    approx_mean = float(approx.kl_sum[:n_seq].sum() / n_pos)
    rel_err = abs(approx_mean - exact_mean) / max(exact_mean, 1e-9)
    return {"exact_kl_mean": exact_mean, "approx_kl_mean": approx_mean, "rel_err": rel_err}
