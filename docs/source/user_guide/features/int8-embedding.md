# INT8 Embedding Sidecars

`--int8-embedding` writes the runtime token-embedding table,
`embedding.safetensors`, as symmetric INT8 with one FP32 scale per vocabulary
row. For Gemma 4 E-models it also writes the per-layer embedding table,
`ple_embedding.safetensors`, as symmetric INT8 with one FP32 scale per
vocabulary row and layer. The runtime gathers only the requested rows and
dequantizes them into the FP16 inputs; neither table is expanded to FP16 in
memory.

Unlike `--fp8-embedding`, which stores FP8 values with block scales for the
token embedding only, the INT8 format uses one scale per row and also covers
the Gemma 4 per-layer table, which has no FP8 option. The two options are
mutually exclusive. FP16 and FP8 token-embedding sidecars and FP16/BF16 PLE
sidecars load unchanged.

## Export

```bash
tensorrt-edgellm-export \
  /path/to/checkpoint \
  /path/to/onnx \
  --int8-embedding
```

The option applies to the runtime sidecars only. It does not change the ONNX
graph, the engine, or the checkpoint's linear-weight quantization.

## Sidecar Contract

The runtime identifies the format by the dtype of the table tensor. The
safetensors metadata records the format, version `1`, and the tensor role.

| File | Tensor | dtype | Shape | Metadata format |
|---|---|---|---|---|
| `embedding.safetensors` | `embedding` | INT8 | `[vocab, hidden]` | `int8_symmetric_per_row` |
| `embedding.safetensors` | `embedding_scale` | FP32 | `[vocab]` | |
| `ple_embedding.safetensors` | `weight` | INT8 | `[vocab, layers * ple_hidden]` | `int8_symmetric_column_groups_per_row` |
| `ple_embedding.safetensors` | `weight_scale` | FP32 | `[vocab, layers]` | |

Each row, or each row's per-layer group for the PLE table, is scaled by its own
maximum absolute value so that the payload uses the full `[-127, 127]` range;
an all-zero row or group uses scale `1.0`. The PLE table gets one scale per
layer because its row is the concatenation of every layer's input: a single
row scale would let an outlier in one layer cost precision in all the others.
The runtime reconstructs each element as `int8_value * scale` in FP16. In the
token-embedding lookup, image and audio placeholder positions receive their
FP16 encoder features exactly as with an FP16 table, and token IDs outside the
vocabulary or placeholders without a mapped feature row are zero-filled. The
PLE gather writes zeros for every placeholder position and out-of-vocabulary
ID, as the FP16/BF16 PLE gather does.

## Memory Example

For Gemma 4 E4B (vocabulary 262,144):

| Table | FP16 bytes | INT8 + FP32 scales bytes |
|---|---:|---:|
| `[262144, 2560]` token embedding | 1,342,177,280 | 672,137,216 |
| `[262144, 10752]` PLE | 5,637,144,576 | 2,862,612,480 |
| Combined | 6,979,321,856 | 3,534,749,696 |

The safetensors headers add a small amount beyond the tensor payloads.

## Packed INT4 Sidecars

`--int4-embedding` writes both sidecars as packed symmetric INT4: every 32
contiguous columns of a row share one FP16 scale, values are rounded to
`[-7, 7]` against that scale and stored as `value + 8` nibbles, two columns per
byte (even column in the low nibble). The runtime detects the format by the
UINT8 table dtype and dequantizes the gathered rows as `fp16((nibble - 8) *
scale)`; placeholder and out-of-vocabulary positions behave as for INT8. The
option is mutually exclusive with `--int8-embedding` and `--fp8-embedding` and
needs a hidden size and `layers * ple_hidden` divisible by 32 (every Gemma 4
E-model qualifies). It combines with `--hostEmbeddingTables`.

| File | Tensor | dtype | Shape | Metadata format |
|---|---|---|---|---|
| `embedding.safetensors` | `embedding` | UINT8 | `[vocab, hidden / 2]` | `int4_symmetric_group32_packed` |
| `embedding.safetensors` | `embedding_scale` | FP16 | `[vocab, hidden / 32]` | |
| `ple_embedding.safetensors` | `weight` | UINT8 | `[vocab, layers * ple_hidden / 2]` | `int4_symmetric_group32_packed` |
| `ple_embedding.safetensors` | `weight_scale` | FP16 | `[vocab, layers * ple_hidden / 32]` | |

For Gemma 4 E4B the two tables shrink from 3,534,749,696 bytes (INT8) to
1,962,934,272 bytes (calculated: 4.5 bits per element). Both INT8 and INT4
round the embeddings that feed the model, INT4 far more coarsely; validate
the task accuracy for each model and format.

`--int4-embedding-scale q4_0_grid` uses codes in `[-8, 7]` and a signed FP16
step per group instead of the default `amax / 7` and `[-7, 7]`. Q4_0 maps the
largest-magnitude value of a group to code -8, so its step is negative when that
value is positive. Each group therefore tries `max(max / 7, -min / 8)` and
`-max(max / 8, -min / 7)` and keeps the one with the smaller reconstruction
error. Checkpoints trained for Q4_0 (for example Gemma 4 QAT Q4_0 releases)
store their token and PLE tables on that grid, so this rule reproduces them up
to FP16 rounding of the scale. The sidecar format and the runtime decode are the same
for both rules.

## Host-Mapped Tables

By default the runtime copies both sidecars into device memory at load. On a
board whose CPU and GPU share one memory pool, that memory is pinned for the
life of the process even though a step touches only a handful of rows.
`--hostEmbeddingTables` (`EmbeddingTableResidency::kHostMapped` in the API)
keeps the files mapped read-only in host memory instead. Each step gathers the
rows of its token ids on the CPU into a pinned staging buffer, copies only those
rows to the device, and runs the same dequantization on them, so the inputs are
bit-identical to the device-resident tables. The mapped pages are file-backed
and reclaimable: the kernel can drop them under memory pressure and fault them
back in on the next lookup.

The staging buffers keep a pinned and a device copy of the rows: the token
table stages the engine's maximum number of physical tokens, and the PLE table
`maxBatchSize` rows times its sequence length (the maximum input length, or the
verify-tree width when that is larger). The mode needs the safetensors sidecars
(not checkpoint-backed tables), single-rank execution, and vanilla or Gemma 4
MTP decoding. The Gemma 4 MTP decoder samples its draft tokens on the device, so
with host-mapped tables it copies those ids back to the host before each lookup:
once per assistant step that feeds a token to the next step, and once for the
verify tokens. Each copy waits for the stream, so a round of draft depth K
synchronizes K + 1 times instead of once. Other speculative decoders and diffusion
decoders are rejected. `llm_inference` rejects the flag together with Qwen3-Omni
audio output, because the Talker also reads the dense table.

## Limitations

- The experimental direct checkpoint builder has no INT8 embedding option; it
  writes FP16 tables (or FP8 with its own `--fp8-embedding`).
- A checkpoint-backed PLE table (direct checkpoint builder) cannot be INT8; the
  runtime rejects it at load.
- Qwen3-Omni models are not supported: the Talker runtime shares the Thinker
  embedding table and reads it as dense FP16.
- A DFlash or DSpark draft that carries its own mask-token embedding cannot be
  folded into a quantized `embedding.safetensors`; export such a base without
  `--int8-embedding`.
