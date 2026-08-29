#!/bin/bash
# Usage: scripts/convert_gguf.sh <model-name>   (e.g. qwen3-1.7b)
# Produces models/gguf/<name>-f16.gguf + pure-quant files for each sweep type.
# NOTE: pure files here are WITHOUT imatrix (fine for gates); regenerate with
# --imatrix once the probe corpus exists, before the real sweeps (EXPERIMENT.md §2).
set -euo pipefail
cd "$(dirname "$0")/.."
name="$1"
src="models/hf/$name"
out="models/gguf"
mkdir -p "$out"

f16="$out/$name-f16.gguf"
if [ ! -f "$f16" ]; then
  uv run python vendor/llama.cpp/convert_hf_to_gguf.py "$src" --outtype f16 --outfile "$f16"
fi

for T in Q8_0 Q6_K Q4_K_M Q3_K_M Q2_K; do
  short=$(echo "$T" | tr '[:upper:]' '[:lower:]' | sed 's/_m$//')
  dst="$out/$name-pure-$short.gguf"
  if [ ! -f "$dst" ]; then
    llama-quantize --pure "$f16" "$dst" "$T" > "$out/$name-pure-$short.log" 2>&1
    echo "made $dst"
  fi
done
ls -la "$out" | grep "$name"
