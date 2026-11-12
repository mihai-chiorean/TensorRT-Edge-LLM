# LoRA adapter hot replacement: design

Status: draft for review, 2026-11-12. Fork `main` c090e173. Implementation is split into issues only after the
prerequisites below have merged.

## 1. Problem

The robot should receive behaviour updates as LoRA adapter directories (about 140 MB at rank 16, FP16) while
the engine files (about 2.65 GB base + 0.16 GB MTP draft, plus embedding tables) stay in place. Copying a new
adapter and restarting `trt-llm.service` already works once the adapter is packaged (MIT-2100). What does not
exist is replacing or adding an adapter **without restarting the server**, with a defined failure behaviour:

- the server keeps answering with the previous adapter if anything about the new one is wrong;
- a half-downloaded, corrupt, mismatched or out-of-memory adapter never becomes active;
- a restart at any moment comes back to a known-good adapter;
- nothing on the network can trigger the swap.

A restart costs about the server's start time (engine load and graph capture) and drops the warm context
cache. Hot replacement avoids both. It is worth building only if the swap window is short and the failure
behaviour is at least as safe as a restart, which is the bar this design sets.

## 2. What the code does today (verified on c090e173)

| Fact | Where |
|---|---|
| All adapters are loaded once, at runtime construction. Names are immutable; registering an existing name throws. There is no remove. | `SharedResources::createForLLM` / `createForSpecDecode` -> `LoRAManager::loadWeights`; `cpp/runtime/state/loraManager.cpp` |
| Every request switches the active adapter at prefill (`switchWeights` / `resetWeights`, then `refreshTensorMap`). | `LLMRankRuntime::setUpForPrefillExecution` |
| Decode graphs are captured once at server start, one per registered adapter plus base, per capture shape. | `LLMRankRuntime::captureBaseGraphWithLoraFanout`, called through `BaseEngineResources` by the decoders' `captureCudaGraphs` |
| Capturing mutates shared runtime state: it resets the KV cache manager's sequence lengths, rewrites pipeline IO tensors and re-prepares the base executor. It cannot run alongside inference. | `Gemma4MTPDecoder::captureCudaGraphs` (the `stateGuard` teardown) |
| The executor's graph cache is keyed by a hash of every binding address and shape; there is no per-adapter eviction. Graphs live until the executor is destroyed. | `TrtEngineExecutor::execute` / `captureGraph`, `mGraphs` |
| Adapter-dependent KV caches are keyed by adapter name today; MIT-2098 keys them by adapter content. | `makeContextCacheSequenceAdmission`, `keySystemPromptWithLoraWeights` |
| The Python server serializes every call into the native runtime with one lock. | `LLM._infer_guard` / `_infer_lock`, `experimental/server/runtime/engine.py` |
| Adapter requests are refused under in-flight batching. | `LLM._build_request` |

## 3. Approach

**Central decision: activation is a short stop-the-world transaction under the server's existing inference
lock, not concurrent preparation.** Loading, verifying and hashing the candidate happen outside the lock; GPU
upload, graph capture, the health probe and the alias flip happen inside it. Requests that arrive during the
window queue behind the lock exactly as they queue behind a long generation today.

Adapters become **versioned registrations** (`tool@2026.11.12-1`) behind a stable **alias** (`tool`) that
requests use. The runtime holds at most a bounded number of registrations. A local-only admin socket triggers
activation. A small on-disk state file per alias makes restarts land on the last known-good version.

Why stop-the-world: the robot serves one conversation at a time (batch 1, serial requests), so a 1-3 s pause
between requests is invisible in practice, while concurrent capture would need a second execution context,
separate pipeline IO and KV scratch, and memory the 16 GB Orin NX does not have (section 7).

## 4. Data model and contracts

### 4.1 On-disk layout (inside the engine's `llm/` directory)

```
lora_weights/
  tool/
    ACTIVE                         # one line: the active version, written atomically (tmp + rename)
    PREVIOUS                       # one line: last version that was ACTIVE before it, for rollback
    2026.11.12-1/                  # one directory per version
      processed_adapter_model.safetensors
      config.json                  # edgellm_lora_processing record
      edgellm_adapter_manifest.json  # MIT-2100: version, base_identity, sha256, rank, targets
    2026.11.20-1/
    .staging/                      # downloads land here; never read by the server
```

- Backward compatibility: a `lora_weights/<name>/processed_adapter_model.safetensors` with no version
  directories is treated as a single version named by its manifest's `version` and is never hot-replaced in
  place.
- A version directory is immutable once it exists. The OTA client writes into
  `.staging/<version>.partial/`, verifies the manifest and hashes with `tensorrt-edgellm-lora-package
  verify`, then renames it to `lora_weights/<name>/<version>/` (same filesystem, atomic). Leftover `.partial`
  directories are deleted at server start.
- `ACTIVE` and `PREVIOUS` are written only by the server, after a successful activation.

### 4.2 Runtime registrations (C++)

```cpp
struct AdapterRegistration
{
    std::string alias;        // "tool": what requests name
    std::string version;      // manifest version
    std::string identity;     // MIT-2098 content identity (32 hex)
    WeightMap weights;        // GPU tensors
    enum class State { kCandidate, kActive, kRetiring } state;
};
```

- `LoRAManager` keys registrations by `alias + "@" + version`; an alias table maps alias -> active
  registration key. `switchWeights(alias)` resolves through the alias table at prefill, so an in-flight
  request never changes adapter mid-sequence (the request already holds its registration key in the context;
  the alias flip only affects the next prefill).
- New API: `stageAdapter(alias, version, path, stream)` (load + validate + upload, state `kCandidate`),
  `activate(alias, version)` (flip the alias, previous active becomes `kRetiring`), `retire(alias, version)`
  (free GPU tensors once no context references the registration), `registrations()` for reporting.
- `EngineExecutor` gains graph ownership tags: `captureGraph(stream, tag)` records the registration key with
  the graph; `evictGraphs(tag)` destroys them. Graphs captured for the base model use the empty tag and are
  never evicted. Freed adapter memory can be reused at the same addresses by a later allocation; replaying a
  graph with matching addresses and shapes is correct because graphs read tensor contents at launch, but
  graphs of retired registrations still leak executable memory, hence explicit eviction.
- `LLMRankRuntime::captureAdapterGraphs(registrationKey)` captures only that registration's graphs for every
  capture shape (the per-adapter part of `captureBaseGraphWithLoraFanout`), then restores the base executor
  binding state the same way `Gemma4MTPDecoder::captureCudaGraphs` does in its teardown.

### 4.3 Residency budget

- At most **two** GPU-resident registrations per alias: active plus one candidate (or the retiring one while
  in-flight requests finish). The rollback version stays on disk and is reloaded if needed; at 140 MB per
  version, a resident rollback copy is not worth the memory on the NX.
- A global cap `--lora-max-resident-adapters` (default: number of aliases + 1) bounds the total across aliases.
  Staging a candidate that would exceed the cap fails before allocation.
- Memory accounted per registration: adapter weights plus the bytes of its captured graph executables
  (measured as the device free-memory delta across capture; reported, not enforced, in the first version).

### 4.4 Admin trigger (local only)

- `--lora-admin-socket PATH`: a Unix domain socket created mode 0600, owned by the service user. Off by default.
  No TCP listener, no endpoint on the public API port, no browser access.
- Requests (JSON over HTTP on the socket, served by a second uvicorn server bound to the UDS):
  - `POST /lora/activate {"alias": "tool", "version": "2026.11.20-1"}` -> runs section 5 synchronously and
    returns the outcome and timings;
  - `POST /lora/rollback {"alias": "tool"}` -> activates `PREVIOUS`;
  - `GET /lora/status` -> registrations, ACTIVE/PREVIOUS per alias, last activation result.
- The edge-conversation OTA agent (a separate edge-conversation issue) is the only intended client.

### 4.5 Observability

`/health.lora.adapters[]` gains `version`, `identity` (short), `state`, and `last_activation`
(`{version, result, error, window_ms, finished_utc}`). The MIT-2099 graph counters show whether the new
registration's decode actually runs from graphs.

## 5. Activation transaction

Outside the inference lock:

1. Resolve `lora_weights/<alias>/<version>/`; refuse if missing, if it is a `.staging` path, or if the version
   is already active.
2. `verify_adapter` against the cached base identity (MIT-2100): manifest, base identity, sha256, targets,
   rank. Refuse on any mismatch. No GPU work has happened yet.

Inside the inference lock (requests queue):

3. `stageAdapter`: validate tensors and upload (MIT-1713 checks, MIT-2098 identity). An allocation failure
   frees anything allocated and refuses. Active adapter untouched.
4. `captureAdapterGraphs(candidate)`. Any capture failure evicts the candidate's graphs, frees it, refuses.
   With `--require-cuda-graphs` (MIT-2099) this is mandatory; without it, a capture failure still refuses
   activation, because a hot swap must not silently degrade speed.
5. Health probe with the candidate selected: a fixed greedy prompt from the manifest (`health_probe.prompt`,
   default a one-line chat turn), at most 8 new tokens. Pass requires no error, finite logits and a non-zero
   decode graph-launch delta. An optional `health_probe.expect_prefix` in the manifest must match. Any failure:
   evict graphs, free, refuse.
6. Flip the alias; mark the old registration `kRetiring`.

Outside the lock again:

7. Write `PREVIOUS` (old version) then `ACTIVE` (new version), each atomically.
8. Retire the old registration (evict graphs, free weights). Its KV state ages out of the context cache under
   the normal eviction policy; it can never be reused because the cache is keyed by content identity (MIT-2098).

Target window (steps 3-6) on the AGX: under 3 s for a rank-16 attention+MLP adapter with MTP. This number is
an acceptance criterion to be measured, not an estimate to rely on.

## 6. Failure modes

| Failure | Behaviour |
|---|---|
| Interrupted download | Directory stays under `.staging/*.partial`; never resolvable as a version; deleted at next start. |
| Corrupt or mismatched package | Step 2 refuses; no GPU work; active unchanged. |
| OOM during upload or capture | Step 3/4 frees the candidate and refuses; active unchanged; `/health` shows the error. |
| Capture fails | Refuse (section 5 step 4). |
| Health probe fails | Refuse (step 5). |
| Crash or power loss before step 7 | `ACTIVE` still names the old version; restart loads it. |
| Crash between writing `PREVIOUS` and `ACTIVE` | `ACTIVE` still old; `PREVIOUS` may already equal it; harmless. |
| `ACTIVE` version fails verification at startup | Load `PREVIOUS` instead, report `degraded: ["lora_active_rejected"]`; refuse to start only if both fail and `--require-lora-adapters` is set. |
| Client disconnects mid-activation | The transaction runs to completion; the result is in `/lora/status`. |
| Request arrives during the window | Waits on the inference lock, like behind a long generation. |

## 7. Alternatives considered

| Option | Tradeoff | Decision |
|---|---|---|
| **A. Stop-the-world activation under the inference lock** (this design) | Pause of a few seconds between requests; no extra GPU memory beyond the candidate's weights and graphs; reuses the single execution context. | Chosen. The robot is serial and memory-bound. |
| **B. Concurrent preparation** in a second execution context with its own pipeline IO, KV scratch and stream, then hand off | No pause. Needs a second `IExecutionContext` sized to the context arena (1.9-2.9 GB measured for the LoRA engine), duplicated IO tensors, and a proof that graphs captured on one context replay correctly when handed to another, which TensorRT's guidance argues against (graphs bind to context state). | Rejected for the NX; revisit only for the AGX if pause time measures badly. |
| **C. Restart the service with the new adapter** | Zero new code; restart cost plus cold context cache; a bad adapter means a failed start and a systemd restart loop unless the OTA agent rolls back on disk itself. | Remains the fallback and the first OTA release path, using MIT-2100's manifest check at startup. |
| **D. TensorRT refit of a refittable LoRA engine** | Could remove the LoRA binding overhead. Unproven through this fork's INT4 plugins; requires a refit-enabled build; out of scope here. | Separate feasibility experiment. |

Reversibility: the on-disk layout (4.1) and the admin socket contract (4.4) are the one-way doors because the
edge-conversation OTA agent will depend on them; they get reviewed with that agent's design. Everything inside
the runtime is a two-way door.

## 8. Prerequisites and sequencing

Must be merged first:

1. MIT-2098: adapter content identity in every cache key.
2. MIT-2099: graph capture gate and graph counters (step 4 and 5 rely on them).
3. MIT-2100: adapter manifest and base identity.

Then, as separate issues, in order:

1. C++: versioned registrations and alias table in `LoRAManager`; `stageAdapter` / `activate` / `retire`;
   GTests for the alias flip at a request boundary and for retire-after-release.
2. C++: tagged graph capture and eviction in `EngineExecutor`; `captureAdapterGraphs`. Verify on the AGX that
   capturing after startup leaves the context cache's published records intact (the MTP capture teardown
   resets sequence lengths; this must not invalidate resident prefix records). This check gates the rest.
3. pybind: `stage_adapter`, `activate_adapter`, `retire_adapter`, `adapter_registrations`.
4. Server: layout of 4.1 (with backward compatibility), startup recovery from `ACTIVE`/`PREVIOUS`, the
   activation transaction, the admin socket, `/health` fields.
5. edge-conversation: OTA agent that downloads into `.staging`, verifies, renames and calls the socket.
6. AGX acceptance (bench owner): window time, memory before/after five activate/retire cycles (no growth),
   graph-launch counters after activation, base output identical before and after base -> A -> B -> base.

## 9. Risks and validations

- **Capture after startup corrupts live cache state.** Validate in step 2 of the sequencing before building on
  it. If it does, activation must also flush the context cache, which makes the swap cost a cold cache.
- **Pause longer than budget.** Measure on the AGX; if graph capture dominates, capture only the shapes the
  deployment serves (batch 1) for the candidate.
- **Memory fragmentation after repeated swaps.** The five-cycle memory check in acceptance.
- **Alias flip mid-request.** Covered by resolving the registration at prefill and holding its key in the
  request context; GTest required.

## 10. Definition of code-complete

- The issues in section 8 are merged with their tests green on the Spark (`edgellm-gtest`, `edgellm-pybind`,
  `edgellm-pytest`).
- AGX acceptance in section 8 item 6 recorded in `docs/JP72-ENGINE-BENCH.md` with device, engine, adapter,
  measured window and memory.
- The fallback path (C) documented in `docs/JP72-ARTIFACTS.md` for releases that do not need hot swap.
