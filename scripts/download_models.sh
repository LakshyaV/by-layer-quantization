#!/bin/bash
# Downloads HF safetensors checkpoints in priority order. Resumable (hf download resumes partials).
set -u
cd "$(dirname "$0")/.."
mkdir -p models/hf
LOG=models/download.log
echo "=== download run $(date) ===" >> "$LOG"

dl() {
  repo="$1"; name="$2"
  echo "--- $repo -> models/hf/$name $(date +%H:%M:%S)" >> "$LOG"
  uv run hf download "$repo" --local-dir "models/hf/$name" >> "$LOG" 2>&1
  echo "--- done $name rc=$? $(date +%H:%M:%S)" >> "$LOG"
}

dl Qwen/Qwen3-1.7B                             qwen3-1.7b
dl Qwen/Qwen2.5-1.5B                           qwen2.5-1.5b
dl deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B   r1-distill-qwen-1.5b
dl Qwen/Qwen2.5-Math-1.5B                      qwen2.5-math-1.5b
dl Qwen/Qwen2.5-Coder-1.5B                     qwen2.5-coder-1.5b
dl Qwen/Qwen3-8B                               qwen3-8b
# Llama-3.2 is license-gated on the official repo; try official, fall back to unsloth mirror.
dl meta-llama/Llama-3.2-3B-Instruct            llama-3.2-3b-instruct
if [ ! -f models/hf/llama-3.2-3b-instruct/config.json ]; then
  dl unsloth/Llama-3.2-3B-Instruct             llama-3.2-3b-instruct
fi
echo "=== all done $(date) ===" >> "$LOG"
