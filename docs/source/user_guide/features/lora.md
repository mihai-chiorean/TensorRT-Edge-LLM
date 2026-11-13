# LoRA (Low-Rank Adaptation)

TensorRT Edge-LLM supports LoRA through the checkpoint-based `tensorrt_edgellm` workflow.

Use dynamic LoRA when you need to select adapters at runtime. Use static LoRA
merge when the adapter is always required, such as the Phi-4-Multimodal
`vision-lora` adapter.

Install the `tools` extra using the [wheel or source installation guide](../getting_started/installation.md#optional-python-dependencies).
The C++ build/inference commands below still require a source build.

## Dynamic Runtime LoRA

This workflow exports the base checkpoint, inserts LoRA inputs into the ONNX
graph, processes HuggingFace adapter weights, then builds an engine with a
maximum adapter rank.

```bash
cd /path/to/TensorRT-Edge-LLM

# Step 1: Export the base model with tensorrt_edgellm
trtedge export \
  /path/to/base_model \
  /tmp/onnx_output

# Step 2: Insert LoRA support into the exported LLM graph
trtedge insert-lora \
  --onnx-dir /tmp/onnx_output/llm

# Step 3: Convert each adapter to the runtime sidecar format
trtedge process-lora \
  --input-dir /path/to/adapter1 \
  --output-dir /tmp/onnx_output/llm/lora_weights/adapter1

trtedge process-lora \
  --input-dir /path/to/adapter2 \
  --output-dir /tmp/onnx_output/llm/lora_weights/adapter2

# Step 4: Build the engine with LoRA support
./build/examples/llm/llm_build \
  --onnx-dir /tmp/onnx_output/llm \
  --engine-dir engines \
  --max-batch-size 1 \
  --max-lora-rank 64

# Step 5: Run inference with adapter selection in input.json
./build/examples/llm/llm_inference \
  --engine-dir engines \
  --input-file input.json \
  --output-file output.json
```

## Adapter contract for engine-stable updates

An adapter can ship as a behaviour update to a deployed device without
rebuilding or replacing the engine. This section states what such an adapter
can change, what it cannot, and what qualifying it means. The rules come from
code; the function names below are where each one is enforced.

### What an adapter can change

- The output of the linear projections the engine was built with LoRA inputs
  for, through `lora_A` / `lora_B` factors. For Gemma 4 robot engines these
  are the attention and MLP projections (`q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, `down_proj`), selected at insertion with
  `--lora-targets self_attn,mlp`.
- Anything those projections influence: tool-call formatting, routing,
  phrasing, refusals, within the limits below.

### What it cannot change

- **Tokens, tokenizer or chat template.** Embeddings, per-layer embeddings and
  the LM head are never adaptable: `process-lora` refuses tensors for
  `embed_tokens`, `embed_tokens_per_layer`, `lm_head`, `vision_tower`,
  `audio_tower`, `visual` and `multi_modal_projector`
  (`_UNSUPPORTED_MODULE_PARTS` in `tensorrt_edgellm/lora/lora.py`). A new token
  needs a new export and engine. A different chat template or tokenizer file
  is a sidecar change that keeps the engine but changes the base identity, so
  every adapter must be re-packaged and re-qualified.
- **Modules the engine has no LoRA input for.** With `--lora-targets
  self_attn,mlp` the per-layer-embedding projections (`per_layer_input_gate`,
  `per_layer_projection`, `per_layer_model_projection`) have no inputs. When
  `process-lora` is given `--onnx-dir` (always, for a deployable adapter), an
  adapter that trains them is refused ("adapter targets modules the engine has
  but did not LoRA-insert"), and tensors for modules the engine does not have,
  such as `k_proj` / `v_proj` of Gemma 4's KV-shared layers, are dropped and
  reported as `unused_adapter_modules`. Without `--onnx-dir` neither check
  runs.
- **PEFT features the runtime does not implement.** `_check_adapter_config`
  refuses `use_rslora`, `use_dora`, a non-empty `rank_pattern` or
  `alpha_pattern`, `modules_to_save`, any `layers_to_transform` or
  `layers_pattern` (including layer `0`), any `bias` other than `"none"`, and
  `r <= 0`. Processing folds one global `lora_alpha / r` into B.
- **Rank above the engine's `max_lora_rank`** (16 for the robot engine). The
  runtime refuses an over-rank adapter at load, before any graph capture.
- **The speculative draft.** With Gemma 4 MTP the draft (assistant) engine has
  no LoRA inputs. It still reads the adapted target's hidden states and shared
  KV cache, but its own weights are unadapted. Gemma 4 MTP decoding is greedy
  (the server turns sampled requests into greedy ones), and every draft token
  is verified against the adapted target, so output is unchanged; acceptance
  can drop. Measured on the AGX on 2026-11-12 with the tool adapter: 65 to 71
  verify rounds for a 129-token story, about 5% slower on long replies.

### Training settings

Train against the same base checkpoint revision the engine was exported from:

```python
from peft import LoraConfig

LoraConfig(
    r=16,                      # <= the engine's max_lora_rank
    lora_alpha=32,             # folded into B; must keep the factors finite in FP16
    lora_dropout=0.05,
    target_modules=r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)",
    bias="none",
    use_rslora=False,
    use_dora=False,
)
```

Do not set `modules_to_save`, `rank_pattern`, `alpha_pattern`,
`layers_to_transform` or `layers_pattern`.

### Packaging and identity

1. `tensorrt-edgellm-process-lora --input-dir <peft adapter> --output-dir
   <engine>/llm/lora_weights/<name> --onnx-dir <llm onnx dir> --max-lora-rank
   16` converts the adapter against the engine's LoRA inputs. For the rank-16
   attention+MLP tool adapter on Gemma 4 E4B the processed file is 69,822,824
   bytes in FP16 (its training checkpoint is twice that, in FP32).
   `--weight-format int8` writes per-column INT8 factors with FP16 scales
   instead; for that adapter: 37.4 MB, at most 0.44% per-tensor error and
   identical greedy output on 9 of 9 test prompts (measured off-device,
   2026-11-12). FP16 engines dequantize it to FP16 at load, so device memory
   and speed are those of the FP16 adapter. INT8-resident engines keep the codes
   and scales on the device (see below).
2. `tensorrt-edgellm-lora-package base-identity --engine-dir <engine>/llm`
   whenever any file it covers changes, then `tensorrt-edgellm-lora-package adapter
   --engine-dir <engine>/llm --adapter-dir <engine>/llm/lora_weights/<name>
   --version <version>` per adapter, then `tensorrt-edgellm-lora-package verify
   --engine-dir <engine>/llm --full` before shipping.
3. The server refuses an adapter whose manifest is missing, names a different
   base identity, or whose weights, modules or rank differ
   (`_verify_lora_adapters` in `experimental/server/runtime/engine.py`), unless
   it runs with `--allow-unverified-lora-adapters`, which serves such adapters
   with a warning. The base identity is the size and SHA-256 of every engine,
   embedding table, tokenizer, chat template and config file in the engine
   directory; any change to their contents changes it (a byte-identical rebuild
   does not). The default `--lora-verify full` checks those hashes at startup;
   `sizes` checks only names and sizes. After a change, write the base identity
   first, then re-package and re-qualify every adapter.

### INT8-resident engines

Insert with `--lora-group-plugin --lora-factor-format int8` to keep
`int8_per_channel_v1` factors on the GPU. Only LoraGroupPlugin engines support
this format. Each target uses the transport file's tensor names:

| Binding | Dtype | Shape |
|---------|-------|-------|
| `<stem>.lora_A.weight` | INT8 | `[K, <stem>.rank]` |
| `<stem>.lora_A.weight_scale` | FP16 | `[<stem>.rank]` |
| `<stem>.lora_B.weight` | INT8 | `[<stem>.rank, N]` |
| `<stem>.lora_B.weight_scale` | FP16 | `[N]` |

Scales are one-dimensional: A's rank is dynamic and B's `N` is static.
Inactive targets have rank 0: A `[K, 0]`, B `[0, N]`, A scale `[0]`, and
B scale `[N]` filled with zeros. These engines refuse FP16 adapter files;
process adapters with `process-lora --weight-format int8`. Kernels reconstruct
each factor as `fp16(float(code) * float(scale))`, so output equals the same
INT8 file served by an FP16 engine, which dequantizes on the host.

### Qualification

- Qualify on the exact deployed engine (behaviour, tool-call accuracy, a
  numerical sanity check), not on the BF16 checkpoint: the engine runs W4A16
  weights, INT8 embedding tables and FP16 activations, so an adapter validated
  in BF16 or under another quantization is not qualified for it.
- **Baking was not a fallback for the tool adapter.** Merging the tool adapter
  into the Gemma 4 QAT checkpoint and re-quantizing to the Q4_0 grid erased its
  effect: the baked engine behaved like the base model (measured on the AGX,
  2026-11-12). A low-rank delta that is small next to the Q4_0 step size is
  rounded away; whether a larger adapter survives is untested, so ship
  adapters as the runtime FP16 LoRA branch. See also the note under
  [Static LoRA Merge](#static-lora-merge).

### Serving limits

- Adapters are selected per request (`model: "<served>:<adapter>"` or
  `lora_name`); requests without one run the base model on the same engine.
- The experimental server refuses adapter requests under
  `--enable-in-flight-batching`; adapters are served on the blocking path, one
  request at a time.
- The CUDA graph counters in `/health.cuda_graphs` come from the root rank's
  executors only; multi-rank (tensor-parallel) deployments do not aggregate
  other ranks. Edge deployments are single-rank.
- Adapter weights are immutable for a runtime's lifetime; an adapter is loaded
  at server start. Replacing one means writing the new version, re-packaging it
  and restarting the server (about 13 s end to end, measured on the AGX on
  2026-11-12). Swapping an adapter in a running server is a draft design (fork PR
  #64), not implemented.

## Static LoRA Merge

Static merge permanently applies a LoRA adapter to the base HuggingFace
checkpoint before optional quantization and ONNX export. Quantizing the merged
checkpoint can round the adapter's low-rank delta away: baking the Gemma 4 tool
adapter into the QAT Q4_0 engine left no measurable effect, so that adapter
ships as a runtime adapter (see
[Adapter contract](#adapter-contract-for-engine-stable-updates)). For Phi-4-Multimodal,
merge the required `vision-lora` adapter before quantization and export; dynamic
runtime LoRA is for adapters selected per request.

```bash
cd /path/to/TensorRT-Edge-LLM

export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Phi-4-multimodal-instruct
export MODEL_DIR=$WORKSPACE_DIR/$MODEL_NAME
export HF_MODEL_DIR=$MODEL_DIR/hf

# Assumes $HF_MODEL_DIR is a local HuggingFace clone with Git LFS files pulled.

# Step 1: Merge LoRA into the base checkpoint
trtedge merge-lora \
  --model-dir $HF_MODEL_DIR \
  --lora-dir $HF_MODEL_DIR/vision-lora \
  --output-dir $MODEL_DIR/merged

# Step 2: Optional quantization of the merged checkpoint
trtedge quantize llm \
  --model-dir $MODEL_DIR/merged \
  --output-dir $MODEL_DIR/quantized \
  --quantization nvfp4 \
  --lm-head-quantization nvfp4

# Step 3: Export the checkpoint with tensorrt_edgellm
trtedge export \
  $MODEL_DIR/quantized \
  $MODEL_DIR/onnx
```

If you do not need weight quantization, export `$MODEL_DIR/merged` directly in step 3.

## Input Format

Specify available adapters and select one adapter per request:

```json
{
    "available_lora_weights": {
        "french": "/path/to/lora_weights/french/processed_adapter_model.safetensors",
        "medical": "/path/to/lora_weights/medical/processed_adapter_model.safetensors"
    },
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "Translate to French: Hello world"}
            ],
            "lora_name": "french"
        },
        {
            "messages": [
                {"role": "user", "content": "What is aspirin?"}
            ],
            "lora_name": "medical"
        }
    ]
}
```

All requests in the same batch must use the same LoRA adapter. To disable LoRA,
omit `lora_name` or set it to an empty string.

## Script Reference

(tensorrt-edgellm-insert-lora)=
### `trtedge insert-lora`

Inserts LoRA patterns into an exported `tensorrt_edgellm` ONNX model and creates
`lora_model.onnx` in the same directory.

| Argument | Required | Description |
|----------|----------|-------------|
| `--onnx-dir` | Yes | Directory containing `model.onnx` and `config.json` |

(tensorrt-edgellm-process-lora)=
### `trtedge process-lora`

Processes HuggingFace LoRA adapter weights for runtime use.

| Argument | Required | Description |
|----------|----------|-------------|
| `--input-dir` | Yes | Directory with `adapter_config.json` and `adapter_model.safetensors` |
| `--output-dir` | Yes | Output directory for processed weights |
| `--onnx-dir` | No | Directory with `lora_model.onnx`; validates the adapter against the engine's LoRA bindings |
| `--max-lora-rank` | No | The engine's `--max-lora-rank`; processing fails if the adapter rank exceeds it |

The output contains `processed_adapter_model.safetensors` and `config.json`
(the adapter config plus an `edgellm_lora_processing` record with the source
adapter digest, the key mapping, the bound, dropped and unbound module lists
and, with `--onnx_dir`, the export and sidecar digests).

Adapter contract. Every tensor must be a `lora_A`/`lora_B` pair of a linear
module; names are mapped onto the engine's `model.layers.N...` stems,
including the `model.language_model.` wrapper prefix of multimodal
checkpoints such as Gemma 4 or Qwen3-VL. Adapters that set `use_rslora`,
`use_dora`, `rank_pattern`, `alpha_pattern`, `modules_to_save`,
`layers_to_transform`, `layers_pattern` or `bias != "none"`, and adapters that
update embeddings, `lm_head` or the vision/audio towers, are refused. With
`--onnx_dir`, each kept pair must match its binding's dimensions exactly,
values must be finite in FP16, every binding inside the adapter's
`target_modules` must be covered, and pairs for modules the engine does not
have (for example `k_proj`/`v_proj` on the KV-shared layers of Gemma 4) are
dropped and listed. Nothing is written when a check fails.

(tensorrt-edgellm-merge-lora)=
### `trtedge merge-lora`

Permanently merges LoRA weights into a base HuggingFace checkpoint.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-dir` | Yes | - | Base model directory |
| `--lora-dir` | Yes | - | LoRA adapter directory |
| `--output-dir` | Yes | - | Output directory for merged checkpoint |
| `--device` | No | `cuda` | Device used while merging |
| `--torch-dtype` | No | `float16` | Model dtype used while merging |

## Build Parameters

| Parameter | Description |
|-----------|-------------|
| `--max-lora-rank` | Maximum LoRA rank to support. Set to `0` to disable dynamic LoRA. |

## Notes

- Static merge produces a single checkpoint and does not require runtime LoRA flags.
- Dynamic LoRA enables adapter switching without rebuilding, but all adapters must have rank less than or equal to `--max-lora-rank`.
- CUDA graphs are captured separately for each LoRA configuration.
