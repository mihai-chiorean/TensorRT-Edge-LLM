# Port of the fork line (86557fb, bundle tools13 = 252c047) onto upstream 0.11.0 (95515c2)

Branch: `feat/jp72-sm87-0.11.0`, created from `origin/main` (95515c2 = tag v0.11.0).
Source line: `fork/feat/jp72-sm87-0.10.1` at 86557fb, 20 commits over e8b2952 (0.10.1); see
`/home/mihai/workspace/TensorRT-Edge-LLM-port/PORT-0.10.1.md` for what each one did.
Upstream delta e8b2952..95515c2 is a single squashed release commit (e649dad + merge), 987 files.

## Phase 0 findings

### (a) Native chat-template rendering vs our tool-template work

0.11.0 renders the model's own Jinja template in C++ (`cpp/chatTemplate/`, vendored Inja in `3rdParty/inja`):

* `cpp/chatTemplate/chatTemplate.cpp:567-640` loads **`<engineDir>/chat_template.jinja`** (or a `chat_template.model`
  marker for three built-in renderers; both present, or an `additional_chat_templates/` directory, is fatal).
  `cpp/runtime/multiDevice/runtimeCoordinator.cpp:288-297` calls `load(mConfig.engineDir, bos, eos)`, so for our tree the
  file is `root/llm/chat_template.jinja`. The 0.11.0 `llm_build` copies it from the ONNX dir
  (`cpp/builder/llmBuilder.cpp:2154-2230`, build fails without one) and the exporter writes it from the checkpoint
  (`tensorrt_edgellm/chat_template.py:241-282`). `processed_chat_template.json` is no longer read anywhere.
* Template variables (`chatTemplate.cpp:695-722`): `messages`, `tools`, `tool_choice` (`"none"|"auto"|"required"|
  {"type":"function",...}`), `parallel_tool_calls`, `enable_thinking`, `reasoning_effort`, `bos/eos_token`.
* Server path: `api/serving_chat.py:201-214` validates the tool request, `:282` sets
  `skip_special_tokens = not (reasoning parser or tool_config.parse_output)`, `:416-421` builds the engine request with
  `tool_config`, `runtime/engine.py:1283-1342` sets `request.tools/tool_choice/skipSpecialTokens`, and
  `cpp/runtime/llmRankRuntime.cpp:1485,2314` now decode with the request's `skipSpecialTokens` (0.10.1 hard-coded
  `true`). `experimental/server/parsing/tool_chat_template.py` is deleted upstream.

Consequences for the fork's server features (file evidence in the table below):

| Fork feature | 0.11.0 | Decision |
|---|---|---|
| Python tool chat template (`parsing/tool_chat_template.py`) | replaced by the C++ renderer | **dropped** |
| template warm-up at startup | Inja template parsed once in `injaRenderer.cpp:104-115` when the coordinator is built; no transformers import | **dropped** |
| delimiter-preservation re-decode of `output_ids` | unnecessary: tool requests decode with `skipSpecialTokens=false` | **dropped** |
| Gemma 4 parser (`call:NAME{...}` bare or in `<\|tool_call>...<tool_call\|>`) | absent; `_MARKER_FAMILIES` (`tool_calling.py:283-298`) covers `<tool_call>`, `<function=`, `[TOOL_CALL]`; auto-selection (`:1087-1103`) maps qwen/nemotron/openai only, Gemma gets `generic` | **ported** |
| forced / required tool enforcement, fail-closed `ToolProtocolError` (502 / SSE error + `finish_reason:"error"`) | `tool_choice` is only handed to the template (`chatTemplate.cpp:446-458`); the only post-check is `_tool_name_allowed` (`tool_calling.py:1297-1301`) which demotes other names to text; `"required"` is never checked. Gemma 4's own `chat_template.jinja` (checkpoint) references `tools` but never `tool_choice`, so the variable is inert for this model | **ported**, with the instruction injected into the messages before they reach the C++ renderer |
| terminal-token stripping with split-safe hold-back | only `.replace("<\|im_end\|>", "")` (`serving_chat.py:45,330,~705`); `<end_of_turn>`/`<eos>` stay in tool output | **ported** |
| incremental tool streaming | `StreamingToolParser` (`tool_calling.py:755-1077`) | adopted (already in 0.10.1) |
| forced choice streamed without re-validation (a5173bf) | chat path streams the prepared request with `tool_config` (`serving_chat.py:416-421,688-695`) | adopted; port only what the Gemma parser needs |
| `seed` accepted (cc76708) | accepted and applied (`api/protocol.py:70`, `engine.py:1316`) | adopted |
| unknown request fields ignored (f532bd7) | still `extra="forbid"` (`api/protocol.py:33-34`) | **ported** |
| greedy normalisation `temperature < 1e-3`, AudioParams | only `temperature == 0` (`serving_chat.py:268-271`) | **ported** |
| image preflight, `--allow-remote-media` (default off) | absent; remote media fetched unconditionally (`media_source.py:80-114`) | **ported** |
| `/health.context_cache` + pybind `media_aware_sequences` | `/health` has `scheduling` only (`api/routes.py:44-70`); metrics logged only (`engine.py:1445-1470`) | **ported** |

### (b) Gemma 4 tokenizer fix vs fail-closed protocol / terminal stripping

`cpp/tokenizer/tokenEncoder.cpp:99-131`, `tokenizer.h:244-252,377-381`: `skipSpecialTokens` now drops only ids in
`mSkippableSpecialTokenIds` (added tokens flagged `special`, bos/eos/pad/unk, and the `config.json` extra EOS ids set via
`setAdditionalEosIds`); non-special added tokens survive; added-token definitions take priority over the base vocab.
The BPE fix (`tokenEncoder.cpp:283-321`, `getLeftNeighborRank`) corrects the left-neighbour rank after a merge and is
generic. Neither changes what the server sees for a tool request: with skipping off every token is kept, including
`<|tool_call>`, `<tool_call|>`, `<end_of_turn>`, `<eos>` and `<|tool_response>`. So the Gemma parser must still accept
both wrapped and bare calls, and terminal-token stripping (with hold-back for split pieces in streams) is still needed.
For non-tool requests the configured EOS ids are now skipped by the tokenizer itself.

Stop tokens: upstream now unions `config.json` and `generation_config.json` `eos_token_id`
(`tensorrt_edgellm/checkpoint/checkpoint_utils.py:1100-1121`, `experimental/builder/core/artifacts/runtime_artifacts.py:84-106`),
which yields `[1, 106, 50]` for Gemma 4 E4B (checkpoint `generation_config.json` carries all three). The fork's
dcd590c (generation_config precedence, strict validation) is superseded; only its input validation and tests would be
worth keeping, and the "copy `eos_token_id` forward into `llm/config.json`" step of the 0.10.1 port is no longer
needed once the ONNX is re-exported with 0.11.0.

### (c) INT8 embedding / PLE sidecars

Still no upstream equivalent. 0.11.0 supports FP16 and FP8 `embedding.safetensors` (`embeddingKernels.cu:692`
`#if SUPPORTS_FP8`) and FP16/BF16 `ple_embedding.safetensors` only (`gemma4EmbeddingPreprocessor.cpp:53-63` still
requires exactly one tensor named `weight`). `kernel::gemma4PleGather` (`embeddingKernels.cu:739-784`) is textually
unchanged. The PLE preprocessor output became token-major (`makeTokenMajorOutputViewForLayer`, `reshapeOutputsTokenMajor`),
so the fork's view hunk must be re-applied there. `tests/test_lists/l0_checkpoint_export.yml` was deleted; the
`test_checkpoint_export` cases now live in `tests/test_lists/l0_onnx_export_{1,2}.yml`. b5edc51 must be re-applied
(cherry-pick with conflicts in `embeddingKernels.cu`, `gemma4EmbeddingPreprocessor.{cpp,h}`, `checkpoint_utils.py`,
`scripts/export.py`, `tests/defs/config.py`, `l0_python_ut.yml`).

### (d) Pre-built engine bundle entry point

Still needed. `experimental/server/runtime/engine_build.py:115-128` still raises "engine bundles are cache artifacts"
for a root containing `llm.engine`; our `root/llm/llm.engine` + `root/visual/` tree is classified as a checkpoint and
fails in `load_model` on the missing `root/config.json` (`engine.py:2165-2195`). `engine_layout.py` is unchanged (c914b74
hunks apply); `engine.py` was rewritten (+640/-166) and gained the in-flight-batching `RequestEngine(...)` constructor at
`:1053-1058`, which must get the same bundle/media/model dirs. d2f79b8 (`chat_template.jinja` copied by `llm_build`) is
superseded by `LLMBuilder::copyTokenizerFiles` (`llmBuilder.cpp:2119-2240`).

### (e) Encoder embedding cache budget flag (86557fb)

Still applies. Pybind `ContextCacheConfig` binds only `enabled`, `max_records`, `recurrent_snapshot_pool_bytes`,
`partial_kv_snapshot_pool_bytes` (`experimental/pybind/edgellm_pybind.cpp:769-774`); no server flag; C++ default still
256 MiB with 0 disabling (`contextCacheConfig.h:57-58`). `EncoderEmbeddingCache` was reworked (entries carry
`features`, multi-entry `tryRestore`, `erase`; still per-entry `cudaMalloc`, LRU), budget semantics unchanged.

### (f) Engine compatibility: tools13 engine and the 0.10.0 ONNX under 0.11.0

**The tools13 engine cannot load, and the 0.10.0 ONNX cannot be rebuilt; a re-export is required.**

* Runtime refuses the engine at three independent points:
  1. `cpp/runtime/config/llmEngineConfig.cpp:583-586` requires `builder_config.ragged_backend ==
     "entry_padded_compatibility"` (`getRequired` throws when missing, `:85-89`); only the 0.11.0 builder writes it
     (`llmBuilder.h:248`). Hit first, in `createDeploymentConfig` (`modelArtifacts.cpp:213`).
  2. `validateAgainstEngine` (`llmEngineConfig.cpp:1481-1490,1563-1574`) requires an engine input named
     `context_sequence_count_carrier`; 0.10.1 engines have `max_seqlen_carrier` / `spec_verify_phase_marker`.
  3. The serialised XQA JIT blob is version 1 and `decoderXQAJitCompiler.cpp` (`kXQA_JIT_BLOB_VERSION{2}`,
     `contiguousQuerySwa` key) throws "Unsupported XQA JIT kernel blob version 1 (expected 2)";
     `AttentionPlugin::createPlugin` returns `nullptr` (`attentionPlugin.cpp:2510-2514`) so TensorRT deserialisation
     fails. Plugin names/versions are unchanged ("1"), so TensorRT itself cannot tell the engines apart. The
     AttentionPlugin I/O also changed (query lengths, four new inputs, rank-3 QKV). `Int4GroupwiseGemmPluginV2`,
     `QkvConcat`, `RmsNorm`, `vitAttention` are backward compatible. `edgellm_version` mismatch only warns
     (`cpp/common/version.cpp:73-105`).
* Builder: the 0.11.0 `llm_build` requires a `positions` input and rank-2 `inputs_embeds` (`llmBuilder.cpp:389-392`,
  "Re-export the model"); the 0.10.0 export has `context_lengths`/`kvcache_start_index`/3-D `inputs_embeds`. So the
  ONNX at `~/tensorrt-edgellm-workspace/gemma-4-E4B/onnx-int8emb` (AGX) is rejected.
* Visual engine: `gemma4ViTRunner.cpp` only changed preprocessing and `vitAttentionPlugin` is unchanged, so the
  0.10.1 `visual.engine` would probably load; it is rebuilt anyway from the new export (393 s on the AGX last time).
* Re-export source: the W4A16 AWQ checkpoint and `quantvenv` on the x86 host
  (`~/tensorrt-edgellm-workspace/gemma-4-E4B/quantized-w4a16-awq`, modelopt 0.45.0, torch 2.13.0+cpu,
  transformers 5.14.1 = the 0.11.0 pins, which are unchanged from 0.10.x apart from `huggingface-hub>=1.5`).
* "INT4 export accuracy fixes": the change that touches Gemma 4 int4_awq is in **quantization**, not export:
  `tensorrt_edgellm/quantization/quantize.py:1113-1133` now keeps the text `down_proj` in FP16 by default
  (4-bit cannot represent the ±2048 GeGLU clamp; opt back in with `--quantize_gemma4_down_proj`). The other fixes
  (per-layer mixed-precision group sizes, NVFP4/FP8/TP/MoE) do not apply to a plain W4A16 AWQ checkpoint. A re-export
  of the existing checkpoint therefore gains nothing from this fix; re-quantizing would, at the cost of
  42 x 10240 x 2560 x 2 B = 2.2 GB of FP16 `down_proj` instead of ~0.55 GB INT4 (+1.65 GB weights, more decode
  bandwidth) on a 16 GB NX. Decision for this port: re-export the existing checkpoint; a down_proj-FP16 re-quant is a
  separate accuracy/memory decision to measure on the NX.

### (g) Free wins for batch-one Gemma 4 E4B W4A16 on Orin NX

| Item | Applies? | Enablement |
|---|---|---|
| bounded sliding-window KV | **No while context reuse is on**: `deploymentConfig.cpp:283-289` selects full storage when `contextReuseEnabled`; also needs export-time `kv_layer_configs[].kv_cache_capacity` (written by the 0.11.0 exporter, `modeling_gemma4_text.py:128-177`) and `llm_build` inputs `swa_kv_page_table`/`swa_kv_cache_mode`. The re-export carries the metadata, so it can be measured later with `--enable-context-reuse` off. |
| GPU media hashing | automatic, payloads >= 1 MiB (`blockHash.cpp:167-173`), zero-copy on iGPU; camera frames are usually below 1 MiB so expect no change | nothing |
| PDL decode/RoPE/GDN | **No**: gated to SM90+ (`attentionPlugin.cpp:78-85`, `xqaLaunchConfig.h:33,64`) and SM120/121 | — |
| XGrammar guided decoding | yes: always compiled (new `3rdParty/xgrammar` submodule, static lib, aarch64 ok), initialised at startup (`llmRankRuntime.cpp:574-578`), per-request via `response_format` / `guided_decoding` (`api/protocol.py:82-83`, `engine.py:634-735`) | nothing beyond the request field; grammars are cached |
| in-flight batching (`--enable-in-flight-batching`) | no gain at `max_batch_size 1` | — |
| fused ViT resize+normalize (`imageUtils::resizeAndNormalizeToRgb`) | automatic for Gemma 4 ViT; supersedes df018e1's lazy resize scratch | nothing |
| CuTe DSL LayerNorm (Orin-tuned) | used by ViTs with LayerNorm; Gemma 4 uses RMSNorm | — |
| tokenizer BPE / special-token fixes, union EOS ids | yes, via the re-export | — |

### Build prerequisites that changed

* `git submodule update --init --recursive` is now required: `cmake/GuidedDecoding.cmake:16-25` is fatal without the
  nested `3rdParty/xgrammar/3rdparty/dlpack`. XGrammar is built unconditionally (no option), C++17, CMake >= 3.20.
* `3rdParty/inja` is a vendored header (no submodule). `jinja2` dropped from the server extras.
* CuTe DSL artifact: still in-tree `kernelSrcs/cuteDSLPrebuilt/`, no download; the `sm_87` tarball was regenerated
  (4.7.0, 100 variants, adds `layernorm`). `EMBEDDED_TARGET=jetson-orin` unchanged (`CMAKE_CUDA_ARCHITECTURES 87`).
* NVRTC still required (`cuda-nvrtc-dev-13-2` already installed on the AGX). No new apt packages.
* New unit-test targets: `unitTestScheduler`; `unitTestCommon` now includes tokenizer/chatTemplate.
* Python: `huggingface-hub>=1.5.0,<2.0`, `httpx` for tools; `require_tensorrt()` now runs before any server-side
  build (not hit by the pre-built bundle path).

## Port plan

1. INT8 sidecars (b5edc51), with the PLE view hunk re-applied to the token-major preprocessor.
2. `tensor.cpp` allocation diagnostics (half of df018e1); the ViT scratch half is dropped.
3. JP7.2 docs (1167182, dd43600, e2b5e4e, e7ed1fd, bc19a43, 70d93ed, 252c047) plus a 0.11.0 addendum.
4. Server: pre-built engine bundles (c914b74); Gemma 4 tool protocol (b9c6ae3 minus the Python template, with the
   forced-tool instruction injected into the messages); terminal-token stripping (d0efe92 minus the re-decode);
   ingress hardening + `/health` metrics (7889673 minus warm-up); unknown fields (f532bd7); budget flag (86557fb).
5. Export stop tokens: validation + tests from dcd590c adapted to upstream's union semantics.
6. Context-cache coordinator test (5788c57) with `ResidentRef{0, 1}`.

Dropped: d2f79b8, cc76708, a5173bf (adopted upstream), ViT scratch half of df018e1.
