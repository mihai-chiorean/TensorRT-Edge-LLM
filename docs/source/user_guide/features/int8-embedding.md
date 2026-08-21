# INT8 Runtime Embedding Sidecars

`--int8-embedding` reduces runtime embedding memory with symmetric INT8
quantization. Ordinary embeddings use one FP32 scale per vocabulary row;
Gemma 4 PLE uses one scale per vocabulary row and layer input. Unlike the FP8
embedding path, the INT8 CUDA gather and dequantization path does not depend
on `SUPPORTS_FP8` and is available on SM87 Jetson Orin devices.

For Gemma 4 models with per-layer embeddings (PLE), the option quantizes both
`embedding.safetensors` and `ple_embedding.safetensors`. Existing FP16/BF16
sidecars continue to load without changes. `--int8-embedding` and
`--fp8-embedding` are mutually exclusive.

## Export

```bash
tensorrt-edgellm-export \
  /path/to/checkpoint \
  /path/to/onnx \
  --int8-embedding
```

The option applies to runtime sidecars only. It does not change the ONNX graph
or the checkpoint's linear-weight quantization.

## Sidecar Contract

The safetensors payload is identified by the signed INT8 table dtype and its
named FP32 scale tensor. Metadata records format
`int8_symmetric_column_groups_per_row`, version `1`, and tensor role
`embedding` or `ple_embedding`.

| File | INT8 table | FP32 scales | Shapes |
|---|---|---|---|
| `embedding.safetensors` | `embedding` | `embedding_scale` | `[vocab, hidden]`, `[vocab]` |
| `ple_embedding.safetensors` | `weight` | `weight_scale` | `[vocab, layers * ple_hidden]`, `[vocab, layers]` |

For each contiguous column group $g$ within row $r$, export computes:

$$
s_{r,g} = \frac{\max_i |w_{r,g,i}|}{127}, \qquad
q_{r,g,i} = \operatorname{clamp}(\operatorname{round}(w_{r,g,i}/s_{r,g}), -127, 127)
$$

A zero group uses scale `1.0` and an all-zero payload. Runtime gathers only the
requested rows and writes FP16 values as $q_{r,g,i}s_{r,g}$; it never expands
the full table to FP16. Invalid token IDs and Gemma 4 image/audio placeholder
rows retain the existing zero-fill behavior.

Before the final FP16 rounding, nearest-integer quantization bounds each
element's absolute error by $s_{r,g}/2$, or
$\max_i |w_{r,g,i}|/254$ for its row group.

## Gemma 4 E4B Memory

For vocabulary size 262,144:

| Table | FP16 bytes | INT8 + scales bytes | Reduction |
|---|---:|---:|---:|
| `[262144, 2560]` input embedding | 1,342,177,280 | 672,137,216 | 670,040,064 |
| `[262144, 10752]` PLE | 5,637,144,576 | 2,862,612,480 | 2,774,532,096 |
| Combined | 6,979,321,856 | 3,534,749,696 | 3,444,572,160 (3.21 GiB) |

Safetensors headers add a small amount beyond these tensor payload sizes.

## Limitations

- The current implementation covers the standard ONNX exporter and C++
  sidecar runtime. The experimental direct checkpoint builder still emits or
  binds FP16 embedding tables.
- Qwen3-Omni models are not supported because their Talker runtimes share the
  Thinker embedding table but do not yet propagate quantization scales.
- DFlash mask-row patching does not support quantized embedding sidecars.
- INT4 sidecars are not part of this format. They require a separate packing,
  scale-granularity, and accuracy design.
