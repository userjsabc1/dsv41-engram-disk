#!/bin/bash
# Same as run_vllm_disk.sh but skips the (very slow, one-time) kernel autotune/warmup.
# Results stay correct; kernels just use default tactics and any remaining JIT happens on first use.
export VLLM_DEEP_GEMM_WARMUP=skip
exec "$HOME/run_vllm_disk.sh" --kernel-config '{"enable_flashinfer_autotune": false}'
