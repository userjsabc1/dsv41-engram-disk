#!/bin/bash
# DeepSeek-V4.1-Flash-1017 (official FP8/FP4 checkpoint) on 4x RTX PRO 6000,
# Engram tables served from the checkpoint on NVMe (VLLM_ENGRAM_DISK=1).
export HF_HUB_OFFLINE=1
# Official DeepSeek-V4.1-Flash-1017 checkpoint directory (local, readable).
MODEL_DIR=${MODEL_DIR:-$HOME/models/DeepSeek-V4.1-Flash-1017}
export VLLM_ENGRAM_DISK=1
export VLLM_ENGRAM_DISK_THREADS=${VLLM_ENGRAM_DISK_THREADS:-32}
export C_INCLUDE_PATH="$(cat "$HOME/py_include_path.txt")"
# nvcc >= 12.9 is needed by DeepGEMM/DeepJIT at runtime (private CUDA 13.2 redist, no root).
export CUDA_HOME="$HOME/cuda-13.2"
export PATH="$CUDA_HOME/bin:$HOME/venvs/vllm-pr/bin:$PATH"  # ninja lives in the venv
exec "$HOME/venvs/vllm-pr/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name dsv41 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 --language-model-only \
  --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --block-size 64 \
  --no-enable-prefix-caching --enforce-eager \
  --attention-config '{"indexer_kv_dtype": "fp8"}' \
  --host 127.0.0.1 --port 8000 "$@"
