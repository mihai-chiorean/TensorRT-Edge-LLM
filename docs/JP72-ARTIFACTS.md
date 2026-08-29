# JetPack 7.2 Artifact Pipeline: Gemma 4 E4B W4A16 AWQ + INT8 Sidecars

Companion to [`JP72-BUILD.md`](JP72-BUILD.md). That document covers the SM87
toolchain on the device (CMake, plugin library, `llm_build`, `visual_build`).
This one covers everything **upstream** of the engine build: the Hugging Face
checkpoint, ModelOpt W4A16 AWQ quantization, and the ONNX export with the
INT8 embedding/PLE sidecars added by `ec2d6cf`.

```text
google/gemma-4-E4B-it (HF, BF16, 14.92 GB)
  -> ModelOpt W4A16 AWQ g128, CPU calibration        [x86_64 host]
  -> quantized HF-style checkpoint                   [x86_64 host]
  -> ONNX export + tokenizer/config/runtime sidecars [x86_64 host]
  -> rsync to device
  -> llm_build (SM87) / visual_build (SM87)          [Jetson Orin NX]
```

## Checkpoint availability

`google/gemma-4-E4B-it` is **not gated**. Verified against the Hub API both
with and without a token:

```bash
curl -fsS 'https://huggingface.co/api/models/google/gemma-4-E4B-it?blobs=true' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("gated:",d.get("gated")); \
      print(sum(s.get("size",0) for s in d["siblings"])/2**30,"GiB")'
```

Nine files, 14.92 GB total, dominated by a single `model.safetensors`
(15,251.7 MiB, BF16). No credentials required.

## Where each stage runs, and why

| Stage | Host | Why |
|---|---|---|
| HF download | x86_64 build host | 14.92 GB; no reason to spend device disk |
| AWQ quantization | x86_64 build host, **CPU** | Needs the full BF16 model resident (~15 GB) plus calibration activations; peak RSS measured at 12.5 GB. The Orin has 15.6 GB of *unified* memory shared with a live voice assistant (3.7 GB available while it runs) and cannot hold the BF16 checkpoint at all, even with every service stopped. |
| ONNX export + sidecars | x86_64 build host, **CPU** | Same memory argument; pure PyTorch/ONNX, no CUDA path |
| `llm_build` / `visual_build` | Jetson Orin NX | A TensorRT plan is architecture-bound. SM87 only. |

Neither quantization nor export requires a GPU. This matters here: the build
host used (AMD Ryzen 7 8845HS, 8 cores / 16 threads, 90 GB RAM) has **no
NVIDIA GPU at all** — only an AMD Radeon 780M iGPU. ModelOpt 0.45.0's
`INT4_AWQ_CFG` path (`awq_lite`) and `export_hf_checkpoint` are both pure
PyTorch and run to completion on CPU; the single `torch.cuda.mem_get_info`
call in `modelopt/torch/export/model_config_utils.py` is guarded by
`if linear_layer.weight.is_cuda`.

## Hard constraint: do not calibrate in FP16 on CPU

The JP6.2 reference ran `--dtype fp16`. On an x86_64 CPU without
`avx512_fp16`, PyTorch FP16 matmul falls off a cliff. Measured on the build
host (`torch 2.13.0+cpu`, 16 threads, `256x2560 @ 2560x10240`):

| dtype | GFLOP/s |
|---|---:|
| float32 | 447 |
| **bfloat16** | **1085** |
| float16 | **0.4** |

FP16 is ~2,700x slower than BF16 here — days, not hours. Use BF16, which is
also the checkpoint's declared dtype (`config.json: "dtype": "bfloat16"`).

This is safe downstream:

- The engine builder documents its input as "a Hugging Face-style
  **FP16/BF16** checkpoint or a supported pre-quantized checkpoint"
  (`docs/source/user_guide/getting_started/direct-engine-builder.md`).
- `tensorrt_edgellm/checkpoint/checkpoint_utils.py` casts BF16/FP32 embedding
  and PLE tables to FP16 before INT8 sidecar quantization (the
  `if weight.dtype in (torch.float32, torch.bfloat16)` guards), so the sidecar
  contract is identical to the FP16 path.
- ModelOpt's AWQ scale search runs in FP32 internally regardless of model dtype.

`tensorrt-edgellm-quantize`'s `--dtype` flag restricts `choices=["fp16"]`, so
the driver below calls `quantize_and_export()` directly with `dtype="bf16"`.

## Versions

| Component | Version | Note |
|---|---|---|
| Python | 3.12.3 | Reference used 3.10; `pyproject.toml` declares `>=3.10` and a 3.12 classifier |
| torch | 2.13.0+cpu | `--index-url https://download.pytorch.org/whl/cpu` |
| nvidia-modelopt | 0.45.0 | as reference |
| transformers | 5.14.1 | as reference |
| onnx | 1.19.0 / onnxscript 0.7.1 / onnx-graphsurgeon 0.6.1 | as `pyproject.toml` |
| safetensors | 0.8.0 | |
| numpy | 2.2.6 | |
| datasets | 5.0.0 | calibration corpus loader |

All pins publish `cp312` or `py3-none-any` wheels; nothing had to be relaxed
for Python 3.12. ModelOpt prints a benign
`transformers 5.14.1 is not tested with current version of modelopt` warning —
the same pairing the JP6.2 reference used.

`ec2d6cf`'s sidecar code is version-clean on this stack:

```bash
cd <repo>
LLM_SDK_DIR=$PWD ONNX_DIR=/tmp/onnx-scratch \
  PYTHONPATH=$PWD ~/tensorrt-edgellm-workspace/quantvenv/bin/python \
  -m pytest tests/python-unittests/test_embedding_quantization.py -q
# 16 passed
```

## Environment

```bash
mkdir -p ~/tensorrt-edgellm-workspace/gemma-4-E4B/{hf,logs}
cd ~/tensorrt-edgellm-workspace

python3 -m venv quantvenv
./quantvenv/bin/pip install --upgrade pip setuptools wheel
./quantvenv/bin/pip install torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cpu
./quantvenv/bin/pip install \
  'nvidia-modelopt==0.45.0' 'transformers==5.14.1' 'onnx==1.19.0' \
  'onnxscript==0.7.1' 'safetensors==0.8.0' 'numpy==2.2.6' \
  'onnx-graphsurgeon==0.6.1' 'datasets==5.0.0' 'tqdm==4.69.0' \
  'einops==0.8.2' 'accelerate'
```

The repository is put on `PYTHONPATH` rather than installed, so no build
artifacts land in the checkout.

## Stage 1 — download

```bash
./quantvenv/bin/pip install huggingface_hub
./quantvenv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("google/gemma-4-E4B-it",
                  local_dir="/home/mihai/tensorrt-edgellm-workspace/gemma-4-E4B/hf",
                  max_workers=8)
EOF
```

## Stage 2 — ModelOpt W4A16 AWQ, group size 128

`INT4_AWQ_CFG` in ModelOpt 0.45.0 resolves to
`{"num_bits": 4, "block_sizes": {"-1": 128, "type": "static"}}` with
`*input_quantizer` and `*lm_head*` disabled — i.e. W4A16 with group size 128,
exactly the reference recipe. Nothing extra needs to be passed.

`~/tensorrt-edgellm-workspace/run_quantize.py`:

```python
#!/usr/bin/env python3
import os, sys, time
sys.path.insert(0, "/home/mihai/workspace/TensorRT-Edge-LLM-gemma-sidecars")
from tensorrt_edgellm.quantization.quantize import quantize_and_export

t0 = time.time()
out = quantize_and_export(
    model_dir=os.environ["MODEL_DIR"],
    output_dir=os.environ["OUT_DIR"],
    quantization="int4_awq",
    lm_head_quantization=None,
    visual_quantization=None,     # visual tower stays FP16, as in the reference
    audio_quantization=None,
    cp_quantization=None,
    kv_cache_quantization=None,
    dtype="bf16",                 # see "do not calibrate in FP16 on CPU"
    device="cpu",
    text_dataset="cnn_dailymail",
    num_samples=int(os.environ.get("NUM_SAMPLES", "128")),
)
print(f"TOTAL {time.time()-t0:.1f}s -> {out}", flush=True)
```

```bash
cd ~/tensorrt-edgellm-workspace
MODEL_DIR=$PWD/gemma-4-E4B/hf \
OUT_DIR=$PWD/gemma-4-E4B/quantized-w4a16-awq \
NUM_SAMPLES=128 OMP_NUM_THREADS=16 \
  ./quantvenv/bin/python -u run_quantize.py \
  > gemma-4-E4B/logs/quantize.log 2>&1
```

### Calibration sample count: 128, not ModelOpt's default 512

`awq_lite` runs `forward_loop(model)` **twice** — once with
`AWQLiteHelper.cache_mode = True` to collect per-channel activation scales,
then once more in search mode where every quantized `Linear` evaluates
`alpha_step=0.1` -> 11 candidate scalings. Total cost is roughly 13
forward-equivalents over the whole calibration set, not one.

Measured on this host: one batch (`batch_size=16`, `max_length=512`,
8,192 tokens) costs **77.5 s** in the caching pass. At 13x that is ~17 min per
batch, so ModelOpt's default `num_samples=512` (32 batches) projects to about
**9 hours** of CPU. 128 samples (8 batches) is ~2.3 hours and matches the
calibration-set size used by the original AWQ paper (128 sequences x 512
tokens). Record this if accuracy is ever compared against the JP6.2 engines,
which used 512.

The calibration corpus is unchanged: `cnn_dailymail` (`abisee/cnn_dailymail`
3.0.0, train split), ModelOpt's and this repo's default text dataset.

## Stage 3 — ONNX export with INT8 embedding/PLE sidecars

```bash
cd ~/tensorrt-edgellm-workspace
PYTHONPATH=/home/mihai/workspace/TensorRT-Edge-LLM-gemma-sidecars \
  ./quantvenv/bin/python -c \
  'from tensorrt_edgellm.scripts.export import main; main()' \
  $PWD/gemma-4-E4B/quantized-w4a16-awq \
  $PWD/gemma-4-E4B/onnx-int8emb \
  --dtype float16 \
  --int8-embedding \
  > gemma-4-E4B/logs/export.log 2>&1
```

`--int8-embedding` is the `ec2d6cf` flag. It is mutually exclusive with
`--fp8-embedding`, applies to runtime sidecars only, and does not change the
ONNX graph or the checkpoint's linear-weight quantization. For Gemma 4 it
quantizes both `embedding.safetensors` and `ple_embedding.safetensors` with
per-row scales (one scale per row for the input embedding; one scale per row
*and per layer input* for PLE), format
`int8_symmetric_column_groups_per_row`, version `1`.

Note the documented limitation in
`docs/source/user_guide/features/int8-embedding.md`: the INT8 sidecars are
implemented for **the standard ONNX exporter plus the C++ sidecar runtime**.
The experimental direct checkpoint builder (`tensorrt-edgellm-build
--model-dir <checkpoint>`) still binds FP16 embedding tables. The device build
must therefore consume this ONNX directory via `llm_build`/`visual_build`,
not the direct builder.

## Target capacity

Capacity is **not** baked into these artifacts. `--max-kv-cache-capacity` in
`tensorrt_edgellm/scripts/export.py` is used only by the Alpamayo action
exporter; the Gemma LLM ONNX export is capacity-agnostic. The choice is made
at build time:

```bash
llm_build --onnxDir <onnx>/llm --engineDir <engines>/llm \
  --maxInputLen 8192 --maxKVCacheCapacity 16384 --maxBatchSize 1
```

### Why 16384, and what it costs

Gemma 4 E4B's KV geometry is unusually cheap. From `config.json`
(`text_config`): 42 layers, `num_kv_shared_layers: 18` (so only the first 24
layers allocate KV), `num_key_value_heads: 2`, `head_dim: 256`,
`global_head_dim: 512`, `sliding_window: 512`, and full attention only at
layers 5/11/17/23/29/35/41 — 4 of which fall inside the 24 allocating layers.

Per token per layer, FP16, K+V: sliding 2,048 B, full 4,096 B.

| max KV | sliding capped + KV sharing | no sliding cap | no KV sharing either |
|---:|---:|---:|---:|
| 1,024 | 36 MiB | 56 MiB | 98 MiB |
| 8,192 | 148 MiB | 448 MiB | 784 MiB |
| 16,384 | 276 MiB | 896 MiB | 1,568 MiB |

Context-phase activations scale with `--maxInputLen` (hidden 2,560 +
gate/up 2x10,240, FP16): 45 MiB at 1,024, 360 MiB at 8,192, 720 MiB at 16,384.

The JP6.2 reference measured **7.3 GiB RSS** at 1024/1024. Projecting:

| Profile | Projected RSS |
|---|---|
| 8,192 KV / 8,192 input | 7.7 - 8.3 GiB |
| **16,384 KV / 8,192 input** | **7.8 - 9.1 GiB** |
| 16,384 KV / 16,384 input | 8.2 - 9.4 GiB |

Measured on `orin-nx-vqplnc` on 2026-08-29 with the live stack running
(`systemctl show -p MemoryCurrent`): `edge-llm` (llama.cpp, Gemma 4 E4B Q4_0,
16,384 ctx) **9.70 GiB**; `edge-stt` 0.93 GiB; `edge-orchestrator` 0.27 GiB;
`edge-tts` 0.25 GiB; `edge-motion` 0.13 GiB; `edge-vision` 0.03 GiB. Total
system 15.23 GiB.

The TensorRT runtime replaces `edge-llm`, so the budget is
15.23 - 1.6 (rest of the edge stack + reachy-daemon) - ~1.6 (kernel, container
runtimes, page cache) ~= **11.8 GiB**. Every profile above fits, and all of
them are at or below what llama.cpp uses today at the same 16,384 context.

Recommended: `--maxKVCacheCapacity 16384 --maxInputLen 8192 --maxBatchSize 1`.
16,384 preserves the context contract `edge-conversation` already runs;
8,192 max input halves the context-phase activation peak while still holding
the ~920-1,190-token system prompt, 280 image tokens and a long history in one
prefill. If the SM87 build runs out of memory, fall back to 8,192/8,192 —
that saves 0.4-0.8 GiB at runtime and materially more builder workspace.
Changing either number later forces an LLM engine rebuild.

Visual profile: Gemma 4 E4B declares `vision_soft_tokens_per_image: 280`.
The reference's validated 4-560 profile covers that with headroom; keep it.

**The engine build needs memory freed.** Stopping `edge-llm` for the build
window releases 9.70 GiB. That is the main session's call, not the build
agent's.
