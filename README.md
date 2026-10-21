# DeepSeek-V4.1-Flash on 4× RTX PRO 6000 (disk-backed Engram)

在 **4× RTX PRO 6000 Blackwell(SM120,每卡 96GB)+ 125GiB 主机内存** 上跑通
DeepSeek-V4.1-Flash-1017(官方 FP8/FP4 检查点,TP=4 + 专家并行)的补丁与脚本。

核心思路:vLLM 默认把 **Engram 表(官方 FP8 版 189GiB,MXFP4 版 97GiB)放在锁页主机内存**,
在 125GiB 内存的机器上启动时会 OOM(实测锁页/共享内存合计 >115GiB,内核卡死约 20 分钟)。
这里改为 `VLLM_ENGRAM_DISK=1`:**直接 mmap 检查点分片里该 rank 负责的连续行区间**,
GPU 端对本批行号去重 → 线程池从页缓存/NVMe 读行 → 拷到显存里的一张紧凑小表 →
复用原有 Triton 取数内核。页缓存是文件映射,可回收,不会像锁页内存那样撑爆主机。

## 基线

- vLLM `main` @ `44f0ddfd7`(2026-10-12)+ [PR #61530](https://github.com/vllm-project/vllm/pull/61530)(MXFP4 Engram,head `1800929af`)
- FlashInfer **0.7.2**(`flashinfer-python` / `flashinfer-cubin` / `flashinfer-jit-cache+cu130`;基线钉的 0.7.0.post1 在 SM120 上页大小不匹配)
- PyTorch 2.13 + CUDA 13.2,驱动需支持 CUDA 13.x

## 补丁(`patches/`)

| 补丁 | 内容 |
|---|---|
| 0001 | 磁盘版 Engram(`common/engram.py`,约 200 行)+ 加载器跳过 `.engram.embed.` 张量(`model_loader/ep_weight_filter.py`) |
| 0002 | 小批量取行也走线程池(小批量冷读 1–5ms → 约 1ms) |
| 0003 | 锁页暂存缓冲显式建在 CPU(模型初始化处于 CUDA 默认设备上下文,`.pin_memory()` 会失败) |
| 0004 | DSV4.1 索引器在 SM120 用 64 行内核块(DeepGEMM 在 SM120 只支持 32/64 行页) |
| 0005 | DSV4.1 稀疏 MLA 后端在 SM120 用 64 token 内核块 |

应用:

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
git fetch origin pull/61530/head
git checkout -b dsv41-engram-disk FETCH_HEAD     # = main@44f0ddfd7 + PR #61530 (1800929af)
git am /path/to/patches/*.patch
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

## 环境要点(都不需要 root)

- **nvcc ≥ 12.9**:DeepGEMM / DeepJIT 运行时要现场编译。系统自带 12.8 不够,用
  `scripts/redist_fetch.py get ~/cuda-13.2 cuda_nvcc cuda_crt cuda_cudart cuda_cccl libnvvm libnvptxcompiler cuda_nvdisasm cuda_cuobjdump libnvfatbin libnvjitlink cuda_profiler_api cuda_nvrtc libcurand`
  解压 NVIDIA redist,并 `export CUDA_HOME=~/cuda-13.2`、`ln -s lib lib64`。
- **`Python.h`**:Triton 首次使用要用 gcc 编一个 C 辅助库。系统缺 `python3-dev` 时,
  `uv python install 3.12` 并把它的 `include/python3.12` 放进 `C_INCLUDE_PATH`。
- **`ninja`** 要在 `PATH` 里(装在 venv 里时 `PATH` 要包含 venv 的 `bin`)。
- 不要开 `enable_multithread_load`(会把整个分片读进内存,Engram 分片单个约 94GiB)。

## 启动

```bash
MODEL_DIR=/path/to/DeepSeek-V4.1-Flash-1017 scripts/run_fast.sh      # 跳过一次性调优,约 3 分钟起服务
python3 scripts/memguard2.py 6 &                                     # 可选:可用内存 <6GiB 时 SIGKILL vLLM 进程树
```

`run_vllm_disk.sh` 的关键参数:`--tensor-parallel-size 4 --enable-expert-parallel --block-size 64
--kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype":"fp8"}' --tokenizer-mode deepseek_v4
--language-model-only --enforce-eager`。`run_fast.sh` 额外设置 `VLLM_DEEP_GEMM_WARMUP=skip` 与
`--kernel-config '{"enable_flashinfer_autotune": false}'`。

> 不要开 FlashInfer 调优:同形状 GEMM 在每一层都重新调优(约 5 个/层 × 46s × 40 层 ≈ 3 小时),
> 且中途停止不会写缓存。

## 验证

- `tests/test_engram_disk.py`:磁盘版取数与原版 Triton 内核(显存里放完整子表)**逐位一致**;
  覆盖 FP8 与 MXFP4 两种表格式、两种行偏移、int32/int64 索引、1–4096 token、重复行与无效索引。
  需 `FP8_CKPT`、`FP4_CKPT` 环境变量指向两种检查点目录;加参数 `timing` 额外测冷/热读耗时。
- `tests/bench.py`:对本机服务的简单延迟/吞吐测试。

## 结果(eager 模式,未开 CUDA 图)

| 项目 | 结果 |
|---|---|
| 权重加载 | 约 92 秒,每卡约 71GiB |
| 单流解码 | 13.6 token/s |
| 4 / 8 并发合计 | 51.5 / 96.5 token/s(每路几乎不降) |
| 长提示词(约 3700 token) | 冷读 8.8s;热缓存下 2 并发共 7400 token 3.3s |
| 主机内存 | 稳定,可用约 110GiB |

端到端只做了少量人工问答检查(中文问答与算术正确),**不是系统性的精度评测**。

## 局限

- 仅验证了官方 FP8-Engram 检查点;NVFP4/MXFP4 版的磁盘路径只通过了单元测试,未跑整机。
- eager 模式,CUDA 图与调优缓存未启用;磁盘读取延迟直接加在每个解码步上(第 1 个 Engram 层要等刚采样出的 token)。
- 未配推理解析器(回答里会带思考过程);未做 systemd 常驻。
- 补丁基于上面的基线提交,与主线的最新改动可能有冲突。

## 许可

补丁是对 vLLM(Apache-2.0)的修改,沿用其 Apache-2.0 许可;`scripts/` 与 `tests/` 的许可由仓库所有者另行决定。
