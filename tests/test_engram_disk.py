"""Bit-exact check of ParallelEngramEmbedding._lookup_disk vs the stock Triton
kernel reading a GPU-resident copy of the same rows, plus a cold-read timing."""
import os
import sys
import time
import types
from unittest import mock

import numpy as np
import torch

import vllm.models.deepseek_v41.common.engram as E

DEV = torch.device("cuda:0")
NUM_EMB = 384_006_168
HEADS, N_HASH_COLS, DIM = 6, 24, 256


def make(model_dir, layer, packed, rows, vocab_start, head_start):
    t = E.ParallelEngramEmbedding.__new__(E.ParallelEngramEmbedding)
    torch.nn.Module.__init__(t)
    t.packed_fp4 = packed
    t.dim, t.block_size = DIM, 32
    t.row_bytes = DIM // 2 if packed else DIM
    t.num_embeddings = NUM_EMB
    t.part_num_embeddings = rows
    t.vocab_start_idx = vocab_start
    t.head_start = head_start
    t.part_n_hash_cols = HEADS
    t.n_hash_cols = N_HASH_COLS
    t._num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    t.disk_source = (model_dir, f"layers.{layer}.engram.embed")
    t.dp_size = 1
    t.cpu_offload = True
    cfg = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=8192)
    )
    with mock.patch.object(E, "get_current_vllm_config", lambda: cfg):
        t._init_disk_tables()
    return t


def reference(t, indices, out):
    """Stock kernel over a GPU copy of the rows (non-sorted path)."""
    w = torch.from_numpy(np.ascontiguousarray(t._disk_mm[0][:])).to(DEV)
    s = torch.from_numpy(np.ascontiguousarray(t._disk_mm[1][:])).to(DEV)
    weight = w if t.packed_fp4 else w.view(torch.float8_e4m3fn)
    rows = indices.shape[0] * HEADS
    grid = min((rows + 15) // 16, t._num_sms)
    E._engram_lookup_kernel[(grid,)](
        weight, s, indices, indices, out,
        t.vocab_start_idx, t.vocab_end_idx_ref, rows,
        indices.stride(0), indices.stride(1),
        HEAD_START=t.head_start, LOCAL_HEADS=HEADS, TOTAL_HEADS=N_HASH_COLS,
        DIM=DIM, QUANT_BLOCK=32, BLOCK_R=16, GRID=grid, SORTED=False,
        PACKED=out.dtype == torch.uint8, OUT_STRIDE=out.stride(0),
        PACKED_FP4=t.packed_fp4,
    )
    del w, s


def make_indices(t, T, dtype, gen):
    """Random ids: this rank's heads inside its row range (with duplicates),
    a few -1 / out-of-range entries, other heads random."""
    idx = torch.randint(0, NUM_EMB, (T, N_HASH_COLS), generator=gen, dtype=torch.int64)
    lo, hi = t.vocab_start_idx, t.vocab_start_idx + t.part_num_embeddings
    mine = torch.randint(lo, hi, (T, HEADS), generator=gen, dtype=torch.int64)
    if T > 4:
        mine[: T // 4] = mine[T // 2 : T // 2 + T // 4]  # duplicate rows
        mine[0, 0] = -1  # invalid
        mine[1, 1] = hi + 5  # out of this rank's range
    idx[:, t.head_start : t.head_start + HEADS] = mine
    return idx.to(dtype).to(DEV)


def check(model_dir, label, layer, packed):
    ok = True
    for vocab_start, head_start in ((0, 0), (96_001_176, 6)):
        R = 2_000_000
        t = make(model_dir, layer, packed, R, vocab_start, head_start)
        t.vocab_end_idx_ref = vocab_start + R
        gen = torch.Generator().manual_seed(1234 + head_start)
        for dtype in (torch.int32, torch.int64):
            for T in (1, 7, 64, 1000, 4096):
                idx = make_indices(t, T, dtype, gen)
                if packed or True:
                    for mode in ("bf16",):
                        if mode == "bf16":
                            a = torch.zeros(T, HEADS, DIM, dtype=torch.bfloat16, device=DEV)
                            b = torch.zeros_like(a)
                        else:
                            width = DIM + DIM // 32
                            a = torch.zeros(T, HEADS * width + 8, dtype=torch.uint8, device=DEV)
                            b = torch.zeros_like(a)
                        reference(t, idx, a)
                        t._lookup_disk(idx, b, False)
                        torch.cuda.synchronize()
                        same = torch.equal(a, b)
                        nz = int((a != 0).sum().item())
                        if not same or nz == 0:
                            ok = False
                            print(f"  MISMATCH {label} vs={vocab_start} hs={head_start} "
                                  f"{dtype} T={T} {mode} same={same} nonzero={nz}")
        print(f"{label}: vocab_start={vocab_start} head_start={head_start} done")
    return ok


def timing(model_dir, label, layer, packed, vocab_start, rows):
    t = make(model_dir, layer, packed, rows, vocab_start, 0)
    gen = torch.Generator().manual_seed(int(time.time() * 1000) % (2**31))
    print(f"[{label}] random reads over {rows} rows "
          f"({rows * (t.row_bytes + 8) / 2**30:.1f} GiB), {t._disk_threads} threads")
    for T in (1, 8, 64, 512, 4096):
        cold, warm = [], []
        for _ in range(5):
            idx = make_indices(t, T, torch.int32, gen)  # fresh rows: page-cache cold
            out = torch.zeros(T, HEADS, DIM, dtype=torch.bfloat16, device=DEV)
            for sink in (cold, warm):  # second call hits the same rows: warm
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                t._lookup_disk(idx, out, False)
                torch.cuda.synchronize()
                sink.append((time.perf_counter() - t0) * 1e3)
        print(f"  T={T:5d} tokens ({T * HEADS:6d} rows): "
              f"COLD median {sorted(cold)[2]:7.2f} ms (max {max(cold):7.2f})   "
              f"WARM median {sorted(warm)[2]:6.2f} ms")


if __name__ == "__main__":
    fp8_dir = os.environ["FP8_CKPT"]   # official checkpoint dir
    fp4_dir = os.environ["FP4_CKPT"]   # NVFP4/MXFP4-Engram checkpoint dir
    allok = check(fp8_dir, "official FP8 ", 1, False) and check(fp4_dir, "NVFP4 MXFP4", 1, True)
    print("ALL BIT-EXACT" if allok else "FAILURES")
    if len(sys.argv) > 1 and sys.argv[1] == "timing":
        timing(fp8_dir, "official FP8 ", 1, False, 96_001_176, 96_001_176)
        timing(fp4_dir, "NVFP4 MXFP4", 1, True, 96_001_176, 96_001_176)
