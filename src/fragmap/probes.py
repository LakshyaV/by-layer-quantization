"""Probe corpus: 9 domains, stored as text (parquet), tokenized per model.

Six natural domains stream from public HF datasets (with fallbacks); three synthetic
micro-probes (induction, factual cloze, arithmetic) generate deterministically in-code.
The frozen parquet + its sha256 manifest are the source of truth for reproducibility;
regeneration from streams is best-effort (stream order can drift across dataset revisions).
Cross-checkpoint rule (EXPERIMENT.md §4): identical text for every model; tokenization
differs per tokenizer, so per-model token SHAs are recorded separately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SEQ_LEN = 512
SEED = 20260829

# (repo, load_dataset kwargs, split, text_field) — field None means domain-specific formatting.
NATURAL_SOURCES = {
    "web": [("HuggingFaceFW/fineweb-edu", {"name": "sample-10BT"}, "train", "text")],
    "code": [("codeparrot/codeparrot-clean-valid", {}, "train", "content")],
    "math": [("AI-MO/NuminaMath-CoT", {}, "train", None),            # problem+solution
             ("meta-math/MetaMathQA", {}, "train", None)],            # query+response
    "qa": [("mandarjoshi/trivia_qa", {"name": "rc.nocontext"}, "train", None)],
    "multilingual": [("wikimedia/wikipedia", {"name": "20231101.zh"}, "train", "text"),
                     ("wikimedia/wikipedia", {"name": "20231101.es"}, "train", "text"),
                     ("wikimedia/wikipedia", {"name": "20231101.ar"}, "train", "text")],
    "chat": [("HuggingFaceH4/ultrachat_200k", {}, "train_sft", None)],
}


def _row_text(domain: str, row: dict, field: str | None) -> str | None:
    if field:
        return row.get(field)
    if domain == "math":
        if "problem" in row:
            return f"Problem: {row['problem']}\nSolution: {row['solution']}"
        return f"Problem: {row['query']}\nSolution: {row['response']}"
    if domain == "qa":
        ans = row["answer"]["value"] if isinstance(row.get("answer"), dict) else row.get("answer")
        return f"Question: {row['question']}\nAnswer: {ans}"
    if domain == "chat":
        msgs = row.get("messages", [])
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs) or None
    return None


def _stream_one(domain, repo, kwargs, split, field, n_chars, skip) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=True, **kwargs)
    texts, total = [], 0
    for i, row in enumerate(ds):
        if i < skip:
            continue
        t = _row_text(domain, row, field)
        if not t or len(t) < 200:
            continue
        texts.append(t)
        total += len(t)
        if total >= n_chars:
            break
    return texts


def _stream_domain(domain: str, n_chars: int, skip: int = 0) -> list[str]:
    sources = NATURAL_SOURCES[domain]
    if domain == "multilingual":  # even mix across languages, not first-success
        texts = []
        for repo, kwargs, split, field in sources:
            texts.extend(_stream_one(domain, repo, kwargs, split, field,
                                     n_chars // len(sources), skip))
        if not texts:
            raise RuntimeError("no multilingual source worked")
        return texts
    last_err = None
    for repo, kwargs, split, field in sources:
        try:
            texts = _stream_one(domain, repo, kwargs, split, field, n_chars, skip)
            if texts:
                return texts
        except Exception as e:  # fall through to the next source
            last_err = e
    raise RuntimeError(f"no source worked for {domain}: {last_err}")


def _induction_texts(rng: np.random.Generator, n: int) -> list[str]:
    # Repeated random word sequences: ...A B C ... A B -> model should predict C.
    # Short chunks x many reps so peaked (repeat) positions dominate — flat first-pass
    # positions break top-K KL (see kl.TOPK note).
    words = [f"w{i:03d}" for i in range(600)]
    out = []
    for _ in range(n):
        seq = rng.choice(words, size=rng.integers(10, 16), replace=False)
        chunk = " ".join(seq)
        out.append((chunk + " . ") * 30)
    return out


def _cloze_texts(n: int) -> list[str]:
    # Reformatted TriviaQA (disjoint slice from the qa domain) as declarative cloze recall.
    rows = _stream_domain("qa", n_chars=n * 300, skip=50_000)
    out = []
    for r in rows:
        q = r.split("Question: ")[1].split("\nAnswer: ")
        if len(q) == 2:
            out.append(f"Fact: the answer to \"{q[0].strip()}\" is {q[1].strip()}.")
    return out


def _arith_texts(rng: np.random.Generator, n: int) -> list[str]:
    out = []
    for _ in range(n):
        a, b = int(rng.integers(23, 987)), int(rng.integers(12, 97))
        c = int(rng.integers(100, 999))
        out.append(
            f"Compute {a} * {b} + {c}. First, {a} * {b//10*10} = {a*(b//10*10)}. "
            f"Then {a} * {b%10} = {a*(b%10)}. So {a} * {b} = {a*b}. "
            f"Finally {a*b} + {c} = {a*b+c}. The answer is {a*b+c}."
        )
    return out


def build_corpus(out_dir: Path, chars_per_domain: int = 400_000) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(SEED)
    rows = []
    for domain in NATURAL_SOURCES:
        for t in _stream_domain(domain, chars_per_domain):
            rows.append({"domain": domain, "text": t})
        print(f"[probes] {domain}: {sum(1 for r in rows if r['domain'] == domain)} docs")
    for domain, texts in [("micro_induction", _induction_texts(rng, 120)),
                          ("micro_cloze", _cloze_texts(400)),
                          ("micro_arith", _arith_texts(rng, 300))]:
        rows.extend({"domain": domain, "text": t} for t in texts)
        print(f"[probes] {domain}: {len(texts)} docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    path = out_dir / "probes.parquet"
    pq.write_table(table, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(
        {"sha256": sha, "n_docs": len(rows), "seed": SEED,
         "chars_per_domain": chars_per_domain,
         "domains": sorted({r["domain"] for r in rows})}, indent=2))
    print(f"[probes] wrote {path} sha256={sha[:16]}…")
    return path


def tokenize_probe(tokenizer, parquet_path: Path, seqs_per_domain: int = 12,
                   seq_len: int = SEQ_LEN, holdout: bool = False):
    """Pack each domain's docs into fixed-length sequences. holdout=True uses the
    back half of each domain's docs (disjoint from the sweep probe's front half)."""
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path).to_pylist()
    by_domain: dict[str, list[str]] = {}
    for r in table:
        by_domain.setdefault(r["domain"], []).append(r["text"])
    eos = tokenizer.eos_token_id or 0
    seqs, domains = [], []
    for domain in sorted(by_domain):
        docs = by_domain[domain]
        half = len(docs) // 2
        docs = docs[half:] if holdout else docs[:half]
        ids: list[int] = []
        for t in docs:
            ids.extend(tokenizer.encode(t))
            ids.append(eos)
            if len(ids) >= (seqs_per_domain + 1) * seq_len:
                break
        n = min(seqs_per_domain, len(ids) // seq_len)
        for k in range(n):
            seqs.append(ids[k * seq_len:(k + 1) * seq_len])
            domains.append(domain)
    tokens = np.array(seqs, dtype=np.int32)
    sha = hashlib.sha256(tokens.tobytes()).hexdigest()[:16]
    return tokens, domains, sha
