# LoRA adapters for the Gemma 4 MTP draft: design

Status: draft for review, 2026-11-13. Fork `main` 23e37714. MIT-2120. The
implementation issues in section 8 are filed after this document is reviewed.

## 1. Problem

Gemma 4 MTP serving pairs the base engine (`spec_base.engine`, LoRA-capable
since MIT-1715) with the assistant draft engine (`spec_draft.engine`, about
0.16 GB), which has no LoRA inputs. The assistant already reads the adapted
target's hidden state and shared KV cache, so its predictions do move with the
adapter, but its own weights do not. Measured on the AGX on 2026-11-12 with
the tool adapter: a 129-token story takes 71 verify rounds with the adapter
against 65 without it, about 5% slower on long replies. Kernel work cannot
recover this; an adapted draft may. Section 7 separates the two effects before
training anything.

Output does not change: every draft token is verified against the adapted
target (`Gemma4MTPDecoder::acceptAndCommit`). This is a speed problem only.

## 2. Current code path (fork main)

| Fact | Where |
|---|---|
| One `LoRAManager`, created from the base engine's `max_lora_rank`; every adapter is loaded into it. The draft gets none. | `SharedResources::createForSpecDecode` (`cpp/runtime/state/sharedResources.cpp`) |
| The adapter is switched at prefill and only the base tensor map is refreshed. | `LLMRankRuntime::setUpForPrefillExecution` |
| The decoder owns the draft executor and its tensor map; draft steps call `mDraftExecutor->prepare(kDecodeProfile, draftDims, mDraftTensorMap, ...)` with no LoRA manager. | `Gemma4MTPDecoder` (`cpp/runtime/decoding/gemma4MTPDecoder.cpp`) |
| Draft graphs are captured once per batch size, with no adapter fanout (the base side uses `captureBaseGraphWithLoraFanout`). | `Gemma4MTPDecoder::captureCudaGraphs` |
| An engine input that is neither registry-bound nor in the tensor map fails `prepare`, so a draft rebuilt with LoRA inputs would fail today until the runtime fills them. | `TrtEngineExecutor::prepare` / `MapBindingCache::apply` |
| The direct builder now builds speculative drafts without LoRA inputs even when `--max-lora-rank` is set, because nothing binds them. | `_speculative_build_plan` (`experimental/builder/cli.py`, #67) |
| The assistant is a direct-builder model; every projection is a `layers.Linear`, so `Linear._uses_lora` and `--lora-targets` (MIT-2097) apply once the draft is allowed a rank. | `experimental/builder/models/gemma4/modeling_gemma4_assistant.py` |

## 3. Decisions

### 3.1 Draft engine: one-time rebuild with LoRA inputs on the called projections

Build the draft with `--max-lora-rank 16 --lora-targets
q_proj,o_proj,mlp,pre_projection,post_projection`. That selects exactly the 22
projections the assistant's forward pass calls (section 4) and leaves out the
ordered-embedding `masked_embedding.centroids` projection and the LM head (the
assistant's own head when its checkpoint has one, otherwise the target's; the
builder never adapts an `lm_head` prefix). The shared-KV attention constructs
`k_proj` / `v_proj` but never calls them, so they get no inputs: naming them
as targets fails the build (`_check_lora_targets` refuses a target that
selects no input), while `self_attn` would work and select only `q_proj` and
`o_proj`. The explicit names are used for clarity.

The builder change is small: `_speculative_build_plan` takes separate draft
LoRA settings (`--draft-max-lora-rank`, `--draft-lora-targets`) instead of
forcing rank 0. The base engine is unchanged. A deployment can keep its
current draft engine; then no draft adapter can be served and behaviour is
exactly today's.

Alternatives:
- (a) No draft adapter, accept the loss. Zero work; about 5% on long tool
  replies forever, and more for adapters that move the distribution further.
- (b) A fully fine-tuned draft engine per behaviour update. Best acceptance,
  but a 0.16 GB engine file per update defeats adapter-only OTA and needs an
  engine build per update.
- (c) This design: draft LoRA inputs once, small draft adapters per update.

### 3.2 Runtime: a second LoRAManager for the draft, one alias for both

- `SharedResources::draftLoraManager`, created in `createForSpecDecode` only
  when `bundle.draft->maxSupportedLoraRank > 0`.
- One alias names both halves: `tool` maps to base weights in the base manager
  and, optionally, to draft weights under the same name in the draft manager.
  An alias without draft weights resets the draft manager to its rank-1 zero
  dummies for that request, so the draft computes the unadapted assistant, as
  today.
- The draft weights are loaded from `processed_draft_adapter_model.safetensors`
  in the same adapter directory. Their content identity (MIT-2098) joins the
  base adapter's identity in the alias's cache identity, so a changed draft
  adapter never reuses draft-side cached state.

### 3.3 Switching: an explicit decoder hook at the request boundary

`DecodingStrategy` gains `onLoraSwitch(std::string const& alias)` (no-op by
default). `LLMRankRuntime::setUpForPrefillExecution` calls it right after the
base switch; `Gemma4MTPDecoder` switches the draft manager and calls
`refreshTensorMap(mDraftTensorMap)`. The binding cache (#65) then rebinds only
the changed draft inputs on the next `prepare`.

Rejected alternative: the decoder reads `context.loraWeightsName` at its own
seed step. That works too, but hides a second switch point inside the decoder
and makes the switch order depend on decoder internals.

### 3.4 Graphs: capture fanout per draft adapter

`Gemma4MTPDecoder::captureCudaGraphs` captures the draft once with no adapter
and once per alias that has draft weights, for each batch size, mirroring
`captureBaseGraphWithLoraFanout`. Capture results feed the same
`/health.cuda_graphs` counters (the draft executor is already reported).

### 3.5 Packaging: an optional `draft` object in the adapter manifest

The MIT-2100 manifest gains:

```json
"draft": {
  "file": "processed_draft_adapter_model.safetensors",
  "size": 1540096, "sha256": "...", "rank": 16,
  "targets": ["model.layers.0.mlp.down_proj", "..."],
  "distilled_from": "<sha256 of the base adapter weights>"
}
```

The server refuses a draft adapter when the deployed draft engine has
`max_lora_rank == 0`, and refuses one whose `distilled_from` is not the paired
base adapter's weights hash (a draft adapter is only useful for the target it
was distilled from). No `draft` object means today's behaviour. The base
identity already covers `spec_draft.engine`.

### 3.6 Losslessness

Greedy output must be byte-identical with and without a draft adapter, and
identical to non-speculative decoding with the same base adapter. Only speed
may change. This is the first acceptance gate.

### 3.7 Cost risk

A LoRA-capable draft runs its LoRA branches on every draft step, including
for requests without an adapter: rank-0 inputs are not an option (a K=0 GEMM
does not write zeros on this TensorRT version, measured on the AGX). The draft
is small (22 branches, 44 inputs; the base has 258 attention and MLP
branches), so the overhead should be small, but it must be measured: the
acceptance metric is end-to-end tok/s, not verify rounds alone.

## 4. Draft binding table (spike)

`tests/python-unittests/test_gemma4_assistant_lora_bindings.py` builds the
assistant module tree with the geometry of `google/gemma-4-E4B-it-assistant`
(`config.json`: `hidden_size` 256, `backbone_hidden_size` 2560, 4 layers,
4 query heads, `head_dim` 256, `global_head_dim` 512 on the full-attention
layer 3, `intermediate_size` 2048, `num_kv_shared_layers` 4, ordered
embeddings with 2048 centroids), runs `Gemma4AssistantForCausalLM.forward`
over mocked graph operations, and records every projection the forward pass
calls: the 22 below plus the LM head and `masked_embedding.centroids`, and no
`k_proj` / `v_proj`. With the targets above `Linear._uses_lora` selects exactly
the 22; with no targets it also selects `masked_embedding.centroids` (the LM
head is never selected).

| Binding stem | k (in) | n (out) |
|---|---:|---:|
| `pre_projection` | 5120 | 256 |
| `model.layers.{0,1,2}.self_attn.q_proj` | 256 | 1024 |
| `model.layers.3.self_attn.q_proj` | 256 | 2048 |
| `model.layers.{0,1,2}.self_attn.o_proj` | 1024 | 256 |
| `model.layers.3.self_attn.o_proj` | 2048 | 256 |
| `model.layers.{0..3}.mlp.gate_proj` | 256 | 2048 |
| `model.layers.{0..3}.mlp.up_proj` | 256 | 2048 |
| `model.layers.{0..3}.mlp.down_proj` | 2048 | 256 |
| `post_projection` | 256 | 2560 |

`pre_projection` reads the concatenation of the token embedding and the
target hidden state (2 × 2560). Each binding has `lora_A [k, r]` and
`lora_B [r, n]`. At rank 16 the FP16 draft adapter is
16 × 2 × Σ(k + n) = 1.54 MB, about 2% of the 69.8 MB base adapter.

The dimensions come from the checkpoint configuration and the module code;
they are not read from a built draft engine. Confirming them against a draft
engine built with LoRA inputs is the first step of issue (A) below.

The deployed draft engine's build command is not recorded in this repository
(the JP72 documents cover the base engine). The fork lists
`google/gemma-4-E4B-it-assistant` as the supported E4B assistant
(`docs/source/user_guide/getting_started/supported-models.md`); issue (A) must
confirm it is the checkpoint behind the deployed `spec_draft.engine`.

## 5. Training note (for the training owner, not code)

Distil the draft adapter from the adapted target:

1. On the base adapter's training prompts, run the target with the base
   adapter and record its greedy continuation tokens (optionally the top-k
   logits), the target hidden state that seeds each draft chain, and the
   target KV the assistant attends to.
2. Train the draft adapter (rank 16, the targets in 3.1) on the drafting chain
   as it runs at inference (`Gemma4MTPDecoder`): the first step reads the
   target hidden state; each later step reads the assistant's own
   `post_projection` feedback from the previous step, with the target KV fixed.
   Unroll the chain to the deployed drafting depth and supervise the token at
   every depth against the adapted target's greedy tokens; teacher-forcing
   each step on target hidden states would never train the feedback path
   (`post_projection`).
3. Target: acceptance with the adapter on equal to base-level acceptance with
   the adapter off (65 verify rounds per 129-token story).

Train against the same quantized target the engine runs if the retrain shows
that matters for the base adapter; the assistant itself is small and BF16.

## 6. Acceptance (measured by the AGX bench owner)

- Greedy output with and without the draft adapter is byte-identical, and
  identical to non-speculative decoding with the base adapter.
- With the tool adapter on: verify rounds per 129-token story within about 2%
  of adapter-off (65).
- End-to-end tok/s with the adapter on at least today's adapter-on number,
  and the measured draft-LoRA overhead on base requests reported.
- Every draft decode step a graph launch for both the base and adapter cases
  (`/health.cuda_graphs.counters.draft`).

## 7. Risks

- **Draft LoRA overhead exceeds the acceptance gain.** Measured in (D); if it
  does, keep (a).
- **The distilled draft adapter does not recover acceptance.** The assistant
  sees the adapted target's hidden states and KV already, so part of the
  adaptation reaches the draft without draft weights. First measure acceptance
  on the LoRA-capable draft with zero draft factors (it must equal today's
  draft), then with the distilled adapter; only the difference is the gain.
- **Memory.** A second set of adapter weights is about 1.5 MB per adapter;
  draft graphs double per adapter (small engine).
- **Naming.** The assistant's PEFT module names must map onto the builder's
  stems (`pre_projection`, `model.layers.N...`); `process-lora` needs an
  assistant mode and a test.

## 8. Implementation split (filed after review)

- (A) Build: `--draft-max-lora-rank` / `--draft-lora-targets` in the direct
  builder's speculative plan; confirm the deployed draft checkpoint; build a
  LoRA-capable draft on the AGX and check its binding names and shapes against
  section 4.
- (B) Runtime: draft `LoRAManager`, the `onLoraSwitch` hook, draft capture
  fanout, the alias's combined cache identity; GTests with
  `MockEngineExecutor`.
- (C) Packaging: `process-lora` for the assistant naming, the manifest `draft`
  object, server startup checks and `/health` reporting.
- (D) AGX acceptance (section 6).
