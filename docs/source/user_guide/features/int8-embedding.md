# INT8 Embedding Sidecar

`--int8-embedding` writes the runtime token-embedding table,
`embedding.safetensors`, as symmetric INT8 with one FP32 scale per vocabulary
row. The runtime gathers only the requested rows and dequantizes them into the
FP16 hidden states; the table is never expanded to FP16 in memory.

Unlike `--fp8-embedding`, which stores FP8 values with block scales, the INT8
format uses one FP32 scale per vocabulary row. The two options are mutually
exclusive. FP16 and FP8 sidecars load unchanged.

## Export

```bash
tensorrt-edgellm-export \
  /path/to/checkpoint \
  /path/to/onnx \
  --int8-embedding
```

The option applies to the runtime sidecar only. It does not change the ONNX
graph, the engine, or the checkpoint's linear-weight quantization.

## Sidecar Contract

The runtime identifies the format by the dtype of the `embedding` tensor. The
safetensors metadata records the format `int8_symmetric_per_row`, version `1`,
and tensor role `embedding`.

| Tensor | dtype | Shape |
|---|---|---|
| `embedding` | INT8 | `[vocab, hidden]` |
| `embedding_scale` | FP32 | `[vocab]` |

Each row is scaled by its own maximum absolute value so that the payload uses
the full `[-127, 127]` range; an all-zero row uses scale `1.0`. The runtime
reconstructs each element as `int8_value * row_scale` in FP16. Image and audio
placeholder positions receive their FP16 encoder features exactly as with an
FP16 table; token IDs outside the vocabulary, and placeholders without a mapped
feature row, are zero-filled.

## Memory Example

For a `[262144, 2560]` token-embedding table:

| Format | Bytes |
|---|---:|
| FP16 | 1,342,177,280 |
| INT8 + FP32 row scales | 672,137,216 |

The safetensors header adds a small amount beyond the tensor payloads.

## Packed INT4 Sidecar

`--int4-embedding` writes `embedding.safetensors` as packed symmetric INT4:
every 32 contiguous columns of a row share one FP16 scale, values are rounded
to `[-7, 7]` against that scale and stored as `value + 8` nibbles, two columns
per byte (even column in the low nibble). The runtime detects the format by the
UINT8 table dtype without an embedding descriptor and dequantizes the gathered
rows as `fp16((nibble - 8) * scale)`; placeholder and out-of-vocabulary
positions behave as for INT8. The option is mutually exclusive with
`--int8-embedding` and `--fp8-embedding` and needs a hidden size divisible by
32.

| Tensor | dtype | Shape | Metadata format |
|---|---|---|---|
| `embedding` | UINT8 | `[vocab, hidden / 2]` | `int4_symmetric_group32_packed` |
| `embedding_scale` | FP16 | `[vocab, hidden / 32]` | |

For a `[262144, 2560]` table the payload is 377,487,360 bytes (calculated:
4.5 bits per element). Both INT8 and INT4 round the embeddings that feed the
model, INT4 far more coarsely; validate the task accuracy for each model and
format.

`--int4-embedding-scale q4_0_grid` uses codes in `[-8, 7]` and a signed FP16
step per group instead of the default `amax / 7` and `[-7, 7]`. Q4_0 maps the
largest-magnitude value of a group to code -8, so its step is negative when that
value is positive. Each group therefore tries `max(max / 7, -min / 8)` and
`-max(max / 8, -min / 7)` and keeps the one with the smaller reconstruction
error. Checkpoints trained for Q4_0 (for example Gemma 4 QAT Q4_0 releases)
store their token table on that grid, so this rule reproduces it up to FP16
rounding of the scale. The sidecar format and the runtime decode are the same
for both rules.

## Limitations

- The experimental direct checkpoint builder has no INT8 embedding option; it
  writes FP16 tables (or FP8 with its own `--fp8-embedding`).
- Qwen3-Omni models are not supported: the Talker runtime shares the Thinker
  embedding table and reads it as dense FP16.
- A DFlash or DSpark draft that carries its own mask-token embedding cannot be
  folded into a quantized `embedding.safetensors`; export such a base without
  `--int8-embedding`.
