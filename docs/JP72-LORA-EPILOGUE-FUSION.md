# LoRA fused into the W4A16 GEMM/GEMV epilogue: design

Status: draft for review, 2026-11-13. Fork `main` 5a56baaa. This is step 4 of
the LoRA decode speed-up plan. It builds on the grouped LoRA plugin branches,
stacked on `main` in this order: `lora-group-plugin` (2ed53e5 `LoraGroupPlugin`,
then b95e8b4 for 8-SM GPUs), `lora-group-insert` (55b3358, `insert-lora
--lora-group-plugin`) and `lora-inactive-rank` (74f7de7, engine-recorded
`lora_inactive_rank`). On INT4 engines it replaces that plugin.

## 1. Problem

The Gemma 4 E4B robot engine runs every attention and MLP projection through
`Int4GroupwiseGemmPluginV2` (W4A16, Q4_0 group 32). Dynamic LoRA adds
`out += fp16(fp16(x @ A) @ B)` to 258 of those projections. Measured on the
AGX Orin (16 SMs, MAXN) with the rank-16 tool adapter:

| Engine | ms/token | tok/s |
|---|---:|---:|
| rank 0 (no LoRA inputs) | 20.89 | 47.87 |
| grouped `LoraGroupPlugin`, adapter on | 23.09 | 43.30 |
| unfused MatMul branches, adapter on | 26.3 | 38.0 |

The current AGX baseline with that adapter on the grouped-plugin branches,
as a fraction of rank-0 throughput, is 90.7% non-MTP, 91.0% for the MTP count
prompt and 93.6% for the MTP story prompt.

The grouped plugin costs 168 extra launches per decode token (about 11.9 µs
each, 2.0 ms in all). The launch cost fits `6.5 µs + bytes / 75 GB/s`: half of
it is fixed per launch, and the other half moves the 69.8 MB of FP16 factors
at half the bandwidth the INT4 kernels reach. Each launch also passes a grid
barrier and does a read-modify-write of the base output. In prefill the same
plugin adds 336 to 408 launches outside CUDA graphs (about +7 ms TTFT on a
short prompt) and about 42 ms of LoRA kernels per 800-token prompt.

Fusing LoRA into the INT4 kernel removes the extra launches, the barrier and
the read-modify-write. What remains is the factor bytes at the INT4 kernel's
bandwidth. The projection is about 97 to 98% of rank-0 decode throughput with
FP16 factors and about 98.5% with INT8 factors, on both the AGX and the NX.

## 2. Which kernels take the epilogue

The task statement says "decode GEMV". The traces say otherwise for most
targets. With group 32, `int4RouteToGemv` sends an M = 1 row to the GEMM when
`N * K > 4 Mi` weights (`kInt4G32M1GemmMinWeights`), and since 02995f4 it
sends M = 2..8 to the GEMM unless `N <= 512`. Decode on the robot engine
therefore runs as follows (AGX nsys, rank-0 engine, M = 1, per token):

| Target | Shape (K x N) | Kernel at M = 1 | Grid (CTAs) | Mean |
|---|---|---|---|---:|
| q, sliding (35) | 2560 x 2048 | GEMM 16x128x64, split-K 2 | 32 | 22.6 µs |
| q, full (7) | 2560 x 4096 | GEMM 16x128x64, split-K 1 or 2 | 32 or 64 | 38.8 µs |
| k, v, sliding (20 + 20) | 2560 x 512 | GEMV | 32 | 8.9 µs |
| k, v, full (4 + 4) | 2560 x 1024 | GEMV | 64 | 12.7 µs |
| o (42) | 2048 x 2560 | GEMM 16x128x64, split-K 4 | 80 | 58.5 µs (o and down) |
| down (42) | 10240 x 2560 | GEMM 16x128x64, split-K 4 | 80 | (as above) |
| gate, up (42 + 42) | 2560 x 10240 | GEMM 16x128x64, split-K 1 | 80 | 88.2 µs |

The split-K factor is the tactic TensorRT autotuned at the profile's optimum
M, so it differs between builds.

Decisions:
- **Both kernels take the epilogue.** `Int4Fp16GemmAmpere` carries q, o,
  gate, up and down in decode, MTP verify (M = 2..8) and prefill. The
  `Int4Fp16GemvAmpere` carries k and v in decode, and k and v sliding in MTP
  verify.
- **Order of work.** The first milestone gives every GEMM tactic and every GEMV
  M a LoRA variant, because the decode profile and the prefill profile autotune
  different tactics (section 8). Decode is measured first. Large-M prefill
  performance comes after it.
- The per-layer-embedding projections (`per_layer_input_gate`,
  `per_layer_projection`) are INT4 but not LoRA targets on the robot engine.
  They keep their non-LoRA variants.

## 3. Kernel design

### 3.1 Contract

For one projection with base weights W (INT4), activation x `[M, K]` and
factors A `[K, r]`, B `[r, N]` (B already carries `lora_alpha / r`):

```
base16 = the existing kernel's FP16 output for this tactic, computed by the
         existing operations unchanged (GEMM: FP32 MMA accumulation, FP16
         rounding of each split-K partial, FP32 add and FP16 rounding at every
         serial fold; GEMV: fp16x2 partial sums, FP32 cross-lane and
         cross-warp reduction, final FP16 cast)
t16    = fp16(T),  T = sum over chunks c = 0..C-1, in order, of P[c]
         P[c] = FP32 partial of x[:, c] @ A[c, :], reduction tree fixed by the
                variant (section 3.2), FMA in round-to-nearest
d16    = fp16(D),  D = sum over j = 0..r-1, in order, of
         fma.rn.f32(float(t16[:, j]), float(B[j, :]), D)
out    = fp16(float(base16) + float(d16))
```

This is the rounding of the unfused branch and of `LoraGroupPlugin`: t and the
delta are rounded to FP16, and the final add is exact FP32 followed by one
rounding. Only the summation order inside t and d differs from those paths.

`base16` stays bit-identical to the non-LoRA variant **of the same tactic, on
the same inputs**. The LoRA work starts only after the base value is final,
including the last split-K slice's fold. It is not bit-identical to an engine
that autotuned a different tactic.

Rejected: folding the delta into the MMA accumulator (t as one extra K-tile
with B as its operand). It saves the separate accumulator but rounds
`base + delta` once, which is a different contract from every existing path,
and it moves greedy outputs for no measurable gain. It can be an opt-in later.

### 3.2 Shrink in the same launch: claimed jobs

Every CTA needs the whole `t` row for its output rows, but `A` must be read
from DRAM only once. The shrink is split into J jobs, one per
(M-tile, K-chunk) pair. The chunk size depends only on the shape and the M
bucket: 512 K elements (8 K-tiles) for M <= 64, and 2048 above. That gives 5
chunks for K = 2560, 4 for K = 2048 and 20 for K = 10240 in decode.

Executable outline (one CTA; `S` is the persistent per-context state, `W` the
TensorRT workspace):

```
entry:
  if tid == 0: issue L2 prefetch of this CTA's B tile (final writer only, 3.3)
  loop:                                   # drain the job queue first
    if tid == 0: s_job = atomicAdd(&S.job, 1)          # relaxed
    __syncthreads()
    j = s_job
    if j >= J: break
    compute FP32 partial P[j] (all threads), store to W.partial[j]
    __syncthreads()                       # all threads' stores issued
    if tid == 0:
      __threadfence()                     # makes the CTA's stores visible
      red.release.gpu.add(&S.done[mtile(j)], 1)
  main loop and base epilogue, unchanged, including the serial split-K lock
  wait and fold; the final writer keeps base16 in registers
  if final writer of the tile:
    if tid == 0:
      while ld.acquire.gpu(&S.done[mtile]) < chunks: spin
    __syncthreads()
    t16 = fp16(sum over c of ld.global.cg(W.partial[mtile][c]))   # chunk order
    out = fp16(float(base16) + float(d16)); store
  __syncthreads()                         # every thread finished with S and W
  if tid == 0:
    if atomicAdd(&S.exited, 1) == gridX * gridY * gridZ - 1:   # acq_rel
      S.job = 0; S.exited = 0; S.done[0 .. mtiles-1] = 0
```

Every CTA, including every split-K slice, reaches the exit block exactly once.
The current GEMM `return`s from inside the serial split-K branch, and the LoRA
variant moves that return after the exit block.

Forward progress. A CTA claims jobs before it waits on anything, and it keeps
claiming until the queue is empty, so no CTA enters its main loop while an
unclaimed job exists. Every claimed job belongs to a CTA that has started and
that finishes the job without waiting. A CTA that waits on `done` therefore
waits only for running CTAs that do not wait. The existing serial split-K wait
happens after the queue is drained, so it adds no new dependency: its lock
order (slice z waits for z - 1) is the one the kernel already relies on. No
co-residency is needed, so 80-CTA grids on 8 SMs (two waves) are fine. In
practice no CTA waits: the jobs are a few microseconds of work at kernel
start, and the main loop takes 20 to 90 µs.

Reset. The last CTA to exit has seen every CTA pass the barrier after its
last use of `S` and `W`, so it can zero the counters. The next launch on the
stream starts after this kernel completes, so it sees the zeros. Because the
reset uses the grid volume of the current launch, the grid may change between
launches: decode and prefill use the same plugin instance. Graph replays are
launches like any other. `S` is zeroed once at allocation (`attachToContext`,
outside capture) before the first launch.

Reduction order. `T` is summed over chunks in index order by every consumer,
so **the reduction order of t depends only on the shape and the variant, not
on the SM count, the grid or which CTA computed which job.** The grouped
plugin's G-way split could not guarantee this: 16-SM and 8-SM runs differed.

Split-K locks are unchanged: the plugin-owned lock buffer, its per-launch
`cudaMemsetAsync` and its separate allocation all stay. LoRA state `S`, the
LoRA partials in `W` and the split-K locks are three distinct buffers.

Rejected alternatives:
- **Every CTA computes t itself.** No synchronisation, but `A` is read once
  per CTA. For gate/up that is 80 CTAs x 80 KB = 6.4 MB of L2 reads per
  projection, about +50% on top of each CTA's 184 KB of weights and scales.
- **Cooperative launch with a grid barrier.** This needs every CTA resident.
  The gate/up grid (80 CTAs) does not fit on 8 SMs at the current occupancy.
- **A separate shrink kernel.** That is one launch per target again, 258 per
  decode token. It stays as the fallback for long prefill if the in-kernel
  jobs measure slower there (section 7).
- **One shrink for q/k/v together.** This would couple three plugin nodes
  through shared state. Per-target shrinks read the same total A bytes, and x
  (5 KB at M = 1) is cheap to read three times.

### 3.3 Expand in the epilogue

GEMM: after the coalesced epilogue reload (`tCrC_epilogue`, eight consecutive
FP16 values of one row per thread per vector), each thread computes the eight
deltas for its vector with FP32 FMAs in rank order. It reads `t16` from shared
memory and `B[0:r, n:n+8]` as 128-bit loads.

Only the final writer stages B. That is every CTA without split-K, and slice
`split_k - 1` with serial split-K, which is always the last in lock order. So
B is read once per output tile, not once per slice. To keep shared memory
unchanged, B and `t16` are staged after the main loop into the pipeline
buffers that the main loop no longer needs (the epilogue's `sC` already
aliases `sA`). The latency is hidden by an L2 prefetch of the B tile
(`r x bN` FP16, 4 KB at rank 16) issued at kernel entry. Rows at or beyond M
are predicated off as today. At bM = 16 the expand is 16 x 8 FMAs per output
vector, negligible next to the main loop.

GEMV: warp 0 lanes with `r == 0` own two channels times M rows and already
sum the per-warp partials before their final FP16 cast. They form `base16`
with that same cast, then add the delta for those channels. The `r x 16` B
tile is read through L2 (prefetched at entry), 512 B at rank 16.

The epilogue is written for any bM. It is generic in tile shape, so every
variant in the tactic table gets a LoRA version from the first milestone. The
prefill-specific work (section 8, milestone D) is the performance of large-M
jobs, not coverage.

### 3.4 Rank, rank 0 and adapter off

- The rank is a runtime argument of the LoRA variants, `1 <= r <= 16`, and is
  not baked. The plugin serializes `max_lora_rank` and refuses, in
  `configurePlugin`, a profile whose maximum rank exceeds the variants' cap,
  so an over-rank engine fails at build time rather than at enqueue.
  `insert-lora` has no rank argument and does not check it. A variant set for
  `r <= 64` is added only if an engine needs it.
- **Rank 0 costs nothing.** The plugin reads `r` from the `lora_A` input
  dimensions in `enqueue`. At `r == 0` it launches the existing non-LoRA
  variant with the same tactic, so the kernel, its output and its time are
  the rank-0 path's. TensorRT allows empty inputs, but they need distinct
  non-null addresses. The runtime already binds rank-1 allocations reshaped
  to rank 0 (`lora_inactive_rank = 0`), which satisfies that.
- Per-request selection does not change. The adapter switch at prefill
  rebinds the factor pointers and rank dimensions, and decode graphs are
  captured per adapter by `captureBaseGraphWithLoraFanout`, so each graph
  holds the variant and rank it was captured with.

### 3.5 State and workspace

- Persistent per-context state `S`: one zeroed device block per plugin
  instance, allocated in `attachToContext` outside graph capture and zeroed
  once with `cudaMemsetAsync`. It holds `job`, `exited` and `done[mtiles]`,
  sized for the maximum M-tile count over the plugin's profiles and the
  smallest bM in the tactic table (2 KB for M = 8192 at bM = 16). The capacity
  is recomputed from the profiles in `configurePlugin` and serialized, so a
  clone and a deserialized plugin allocate the same size.
- Partials live in the TensorRT workspace (`getWorkspaceSize`), which layers
  share because they run in order on one stream. The size is the maximum over
  every M bucket boundary and every tactic of
  `ceil(M / bM) * bM * ceil(K / chunk(M)) * 16 * 4` bytes. A maximum taken
  only at the profile's largest M under-allocates, because the chunk count
  drops when M crosses 64. For example, K = 10240 at bM = 16 needs 81,920 B
  at M = 64 but 40,960 B at M = 128. For down at M = 8192 the bound is about
  2.6 MB.
- Different contexts and different layers never share state. Two graphs
  captured from one context never run concurrently.

### 3.6 INT8 factors (step 3)

Step 3 (another change, in progress) keeps the factors in INT8 on the device.
The fused kernels take that format directly:

- Inputs: `lora_A` INT8 `[K, r]` with `lora_A.weight_scale` FP16 `[r]`, and
  `lora_B` INT8 `[r, N]` with `lora_B.weight_scale` FP16 `[N]`. These are the
  tensor names and per-column scales of the `int8_per_channel_v1` transport
  from #73.
- Producers dequantize A and consumers dequantize B as
  `fp16(float(code) * float(scale))`, the expression the runtime uses at load
  today. Bytes halve: about 37 MB per token instead of 69.8 MB.
- Equality: the INT8 variant is bit-identical to the **fused FP16 variant fed
  the same factors dequantized on the host**. It is not bit-identical to the
  grouped or MatMul paths, for the reason in 3.1.
- Runtime ABI. Today `LoRAManager` dequantizes to HALF, drops the scales,
  validates 2-D HALF factors and allocates HALF dummies, and
  `setupLoraProfiles` gives profiles only to the 2-D factors. An INT8-resident
  engine needs:
  - typed factor bindings (INT8) and 1-D scale bindings, with the A scale's
    dynamic `<stem>.rank` dimension in the profiles;
  - inactive shapes for all four tensors at rank 0, where the B scale stays
    `[N]`;
  - validation by dtype;
  - registration of the codes and scales as stored.
  Milestone E specifies this together with step 3.
- A plugin attribute `lora_factor_format` (`0` FP16, `1`
  `int8_per_channel_v1`) selects the variant set; it is fixed per engine. If
  step 3 lands with other binding names, the fused plugin uses those.
- An INT8-resident engine refuses an FP16 adapter file instead of quantizing
  it at load. Requantization changes the delta by 0.56 to 0.77% for the tool
  adapter, so it must be a deliberate `process-lora --weight-format int8` step
  that is qualified like any other adapter.

## 4. MTP

- **Verify (base engine).** Verify rounds run the base engine at
  M = batch x verify width (`activeBatchSize * verifySize`), which is at most
  8 on the batch-one robot engine. The GEMM handles any M with any tile, and
  the GEMV variants cover M <= 8. A larger flattened M routes to the GEMM,
  which the LoRA variants also cover. A verify step costs the same LoRA bytes
  as a plain decode step and no extra launches. The grouped plugin's 15% per
  group penalty at M > 1 disappears.
- **Draft.** The assistant draft engine has no LoRA inputs. #75 designs draft
  adapters with TensorRT-native MatMul branches from the direct builder. This
  design does not change the draft. The acceptance loss with the adapter (65
  against 71 verify rounds on a 129-token story) is untouched by kernel work.
- **What equality MTP has.** Every draft token is checked against the adapted
  base engine's verify output, so MTP output is the greedy output **of the
  verify computation**. It is not guaranteed to equal non-speculative decode
  token for token. The same row can take different kernels at different M:
  full-attention k/v route M = 1 to the GEMV and M = 2..8 to the GEMM. The
  verify width also changes the tile residue, and the chunk size changes at
  M = 64. That holds today without LoRA. The equality plan (section 6)
  measures MTP against autoregressive output at identical prefixes instead of
  assuming it.

## 5. Engine, plugin and adapter interfaces

### 5.1 Plugin

A new plugin, `Int4GroupwiseGemmLoraPlugin` version 1, in
`cpp/plugins/int4GroupwiseGemmPluginV2/`, shares its code with
`Int4GroupwiseGemmPluginV2` (same tactic table, same lock workspace, same
GEMV routing). `Int4GroupwiseGemmPluginV2` and its serialized engines do not
change.

| Index | Input | Type and shape |
|---|---|---|
| 0 | activation (smoothed, as today) | FP16 `[M, K]` or `[b, s, K]` |
| 1 | fragment weights | INT8 `[rows, 512]` |
| 2 | scales | FP16 `[K / G, N]` |
| 3 | LoRA activation (unsmoothed) | FP16, same shape as 0 |
| 4 | `<stem>.lora_A.weight` | FP16 or INT8 `[K, rank]` |
| 5 | `<stem>.lora_B.weight` | FP16 or INT8 `[rank, N]` |
| 6, 7 | `lora_A/B.weight_scale` (INT8 only) | FP16 `[rank]`, `[N]` |

Attributes: `gemm_n`, `gemm_k`, `group_size`, `max_lock_workspace_bytes` as
today, plus `max_lora_rank` and `lora_factor_format`.

Input 3 is separate because AWQ exports feed the plugin `x * s`, while the
adapter was trained on `x` (`_unsmoothed_int4_activation`). For the Gemma 4
QAT import there is no smoothing, and 0 and 3 are the same tensor.

The output is a plain output, not an alias, so the builder no longer needs
`kALIASED_PLUGIN_IO_10_03` on these engines. Delta and base are added inside
the kernel, before any downstream consumer, so the Gemma 4 PLE projection's
output scale applies to both, as `_check_lora_add_precedes_consumers` requires
today.

Tactics: one LoRA variant per existing GEMM and GEMV variant (32 GEMM and 16
GEMV functions per group-size set become 64 and 32). The tactic id indexes
the same table, and TensorRT autotunes with the LoRA inputs at the profile's
optimum rank. The `int4_fp16_gemm` group's share of each AOT artifact
roughly doubles.

### 5.2 Insertion and build

- `insert-lora --lora-backend {matmul,group,fused-int4}`, with `matmul` as the
  default. `group` is the `--lora-group-plugin` path from the split PRs.
  `fused-int4` rewrites each selected `Int4GroupwiseGemmPluginV2` node to
  `Int4GroupwiseGemmLoraPlugin` and appends inputs 3 to 5. Non-INT4 targets
  in the same graph keep MatMul branches, and then `lora_inactive_rank`
  stays 1. The rank cap is enforced by the plugin (3.4), not here.
- Graph input names, dtypes and the dynamic `<stem>.rank` dimension do not
  change in FP16 mode.
- `builder_config` records `lora_backend` and `lora_factor_format`, and
  `lora_inactive_rank = 0` when every LoRA-reading layer is a fused or group
  plugin. Aliased plugin I/O is enabled separately, when any group plugin is
  present (section 9).
- `scripts/jp72/build_engines.sh` gains `LORA_BACKEND=fused-int4`.

### 5.3 Adapter manifest and base identity (#69, #72)

- Processed FP16 adapter files are byte-identical for the fused engine:
  same names, shapes and rank. `process-lora` output, adapter weight hashes,
  module lists and `LoRAManager` validation do not change.
- The engine file changes, so the base identity changes, as with any rebuild.
  The base identity is written first, then every adapter manifest is
  re-issued against it (`lora-package adapter`) and verified with `--full`.
  The server's refusal of a manifest naming another base identity stays as it
  is.
- An INT8-resident engine accepts only `int8_per_channel_v1` files. The
  manifest already records the weights hash. `lora-package` adds a check that
  the file's `edgellm_lora_format` matches the engine's `lora_factor_format`.
- Adapter-dependent KV cache keys (#68) use adapter content and are unaffected.

## 6. Equality strategy

Bit-equality with the grouped plugin or the MatMul branches is not possible,
because the summation order inside t and d differs. It was already not
possible between AGX and NX with the grouped plugin. The strategy proves what
can be proven exactly and measures the rest.

Kernel level (C++ gtest against the AOT artifacts, plus the CuPy harness).
The CuPy harness builds a debug-only variant that also writes `t16` and
`base16` to side buffers; the AOT variants do not have these outputs.
1. **Base untouched.** The debug `base16` of every LoRA variant equals the
   output of the non-LoRA variant of the same tactic, bit for bit. Coverage:
   every GEMM tile and split-K, every GEMV M from 1 to 8, M in
   {1, 2, 3, 5, 7, 8, 15, 16, 17, 33, 64, 65, 128}, and N not a multiple of
   128.
2. **t exactly.** A host emulation of the variant's reduction tree (the job's
   per-thread FMA order, its intra-CTA reduction, then chunks in order)
   reproduces the debug `t16` bit for bit. An FP64 reference bounds the
   emulation's error, as a check on the emulation itself.
3. **Output exactly.** From the kernel's own `base16` and `t16`, the host
   computes `d16` and `out` by the contract in 3.1 (FMA in rank order), and
   the kernel's `out` must match bit for bit. The full output is compared, not
   `out - base16`: when `d16` is below half an ulp of `base16`, the correct
   output equals `base16`. Ranks 1, 7, 8 and 16. K with a partial last chunk
   (K = 2048 + 64). Signed zeros and large `t16` (near FP16 overflow) are
   included.
4. **Determinism.** Repeated launches, swizzle on and off, and artificial
   producer delays (a debug spin in the job path) give bitwise-identical
   output.
5. **Engine-level rank 0** (not a direct plugin call). Build a small engine
   with the fused plugin and run it with the output buffer poisoned, under:
   - an uncovered target;
   - rank 0 to positive to 0 transitions across both profiles;
   - adapter-specific graph replays.

   The rank-0 output must equal `Int4GroupwiseGemmPluginV2` with the same
   tactic.
6. **State.** Alternate M = 1 and M = 300 on one instance. Alternate graph
   replay with eager launches. Run two contexts on two streams with
   independent plugins. Replay 1000 times. Then the counters must read zero
   and `compute-sanitizer --tool memcheck,racecheck,synccheck` must stay
   clean. Every test runs under a watchdog timeout, so a deadlock fails fast
   instead of holding the GPU.

Engine level, on the AGX and then the NX:
7. Adapter off on the fused engine against the rank-0 engine: decode time
   within noise. Greedy output is compared on the 9-prompt tool set. A
   difference there can only come from a different tactic choice, which the
   build log shows.
8. Adapter on: greedy token equality on the 9-prompt tool set and the
   66-check golden set, against the grouped-plugin engine and the unfused
   MatMul engine. Report every divergence with its first-divergence position
   and the top-2 logit margin there. Gate: no golden-set or tool-call accuracy
   regression; isolated near-tie divergences are expected and accepted.
9. MTP against autoregressive decode on the fused engine, at identical
   prefixes and adapter on and off, for verify widths 2 to 8. Report the
   divergences, which can also occur without LoRA (section 4).
10. Cross-device: with `HW_COMPAT=same_compute_capability` the AGX-built
    engine runs on the NX with the same tactics, and the fused LoRA path has
    no SM-count dependence. Its outputs are expected to be bit-identical
    across the two devices. This is checked, not assumed.
11. INT8 factors: bit equality between the INT8 engine and the FP16 fused
    engine given host-dequantized factors.

## 7. Orin NX (SM87, 8 SMs, 102.4 GB/s)

- **Bandwidth per SM is the same as on the AGX** (12.8 GB/s), and both base
  and LoRA bytes scale with bandwidth, so the relative cost is about the same
  on both devices.
  - The NX base step is about 43 ms.
  - Fused LoRA moves A once and B once per output tile. That is 69.8 MB with
    FP16 factors and about 37 MB with INT8 factors.
  - Prefill also re-reads A once per M tile, a small fraction at prefill M.
  - At the INT4 kernels' efficiency that projects to about 0.9 to 1.1 ms (97.5
    to 98%) with FP16 and 0.5 to 0.6 ms with INT8.
  - This projection is checked in milestone C against measured DRAM and L2
    traffic (nsys and ncu) and job stalls before it is quoted as a result.
  - The grouped plugin is projected at 3.4 to 4.0 ms on the NX.
- **No SM-count tuning.** Chunk sizes, job counts and variants depend only on
  the shape. The job queue needs no residency, so the 8-SM grids that forced
  the grouped plugin into rounds are not a special case.
- **Shared memory and occupancy.** B and `t16` reuse the pipeline buffers
  after the main loop (3.3), so a LoRA variant has the same shared-memory
  footprint as its base variant. A dedicated B buffer would not fit the
  budget: group 32 bM = 16 at 4 stages would grow from 26 KB to 30.5 KB, which
  lowers the SM87 residency from six blocks to five, and bM = 128 at 4 stages
  could drop from two blocks to one. The job path's scratch for the partial
  reduction also reuses the pipeline buffers, because it runs before the
  prologue.
- **Registers.** Register allocation is per kernel, so the epilogue and job
  code can raise the main loop's allocation even though their live ranges do
  not overlap. Each LoRA variant's compiled register count, spills and
  achievable occupancy on SM87 are compared with its base variant (cubin
  resource usage plus ncu on the NX). This includes the GEMV `min_blocks`
  budgets (85 registers at `min_blocks = 3`). A variant that spills or loses a
  resident block fails milestone A and is reworked before it is measured.
- **Clocks.** The removed fixed cost per launch scaled with GPU clock (25 W
  mode at 918 MHz is about 1.4 times the AGX's per-launch cost), so the NX
  gains relatively more from fusion than the AGX.
- **Memory.** The extra device memory is the per-context state (at most 2 KB
  per plugin at M = 8192) and the partial workspace (at most about 2.6 MB,
  shared). INT8 factors save about 35 MB per resident adapter.
- **Long prefill.** For M > 64 the jobs compute bM-row partials on CUDA cores
  before the main loop. If the 800-token TTFT on the NX is worse than with a
  separate shrink kernel (one launch per target, about 258 per prefill),
  prefill above a threshold M uses that kernel instead. The choice is made per
  M bucket and still depends only on the shape.

## 8. Plan

TensorRT picks a tactic per optimization profile. On the robot engine the
decode profile picks bM = 16 at M = 1, and the prefill profile picks larger
tiles. Its tactic filter drops bM = 16 when the optimum M is at least 256. The
LoRA epilogue is generic in tile shape, so milestone A gives every tactic a
LoRA variant, and the engine needs no tactic filtering changes at any
milestone.

| Milestone | Content | Gate |
|---|---|---|
| A. Kernels | LoRA variants for every GEMM tactic and every GEMV M with FP16 factors; job queue; final-writer B staging; CuPy debug variant; tests 1 to 4 and 6. | Kernel tests pass on the AGX; resource usage equal to the base variants; microbench per decode shape against base + grouped plugin. |
| B. Plugin and insertion | `Int4GroupwiseGemmLoraPlugin`, `insert-lora --lora-backend fused-int4`, builder config and predicates, `build_engines.sh`. | Test 5; engine builds on the AGX; tests 7 and 8. |
| C. Decode measurement | b8 harness, nsys node and graph modes, ncu traffic for the four decode shapes. | Decode ms/token against rank 0 and the grouped plugin; zero LoRA kernels in the decode trace; test 9. |
| D. Prefill performance | Large-M job path tuned; shrink-kernel fallback only if measured necessary. | TTFT at 20, 100, 400, 800 and 1600 tokens against rank 0 and the grouped plugin. |
| E. INT8 factors | `lora_factor_format = 1` variants, runtime ABI (3.6), manifest format check. Coordinated with step 3. | Test 11; decode bytes halved in nsys. |
| F. NX | Build on the NX (or run the AGX-built engine), repeat C and D, test 10. | NX decode and TTFT against rank 0. |

Milestones A to C are the decode result. D to F follow in that order.

## 9. Risks

- **Spin-wait semantics in CuTe DSL.** The kernel needs acquire loads,
  release reductions and `ld.global.cg`. The existing serial split-K path
  uses `atomic_add`, `atomic_exch` and `fence_acq_rel_gpu`, so the primitives
  exist; inline PTX covers anything missing.
- **Deadlock in a test holds a shared GPU.** All kernel tests run under a
  watchdog and the AGX GPU lock, and never on a GPU that serves other
  traffic.
- **A kernel that faults mid-launch leaves the counters non-zero.** A fault is
  a sticky context error, so the context is lost anyway. The state is zeroed
  again in `attachToContext`.
- **Tactic choice moves.** Autotuning with the LoRA epilogue can choose a
  different split-K than the base engine. That changes base rounding between
  engines (as any rebuild can) and is visible in the build log. The rank-0
  path always uses the LoRA engine's own tactic.
- **AOT build time and size double** for the `int4_fp16_gemm` group, in every
  artifact of the Docker builder matrix.
- **Dependency on the grouped-plugin branches** (`lora-group-plugin`,
  `lora-group-insert`, `lora-inactive-rank`) for `lora_inactive_rank`, the
  rank-0 binding path and the insertion plumbing. The builder there uses one
  predicate ("every LoRA-reading layer is a `LoraGroupPlugin`") for both
  inactive rank 0 and aliased plugin I/O. This design splits it in two:
  - rank 0 is allowed when every LoRA-reading layer is a group or fused
    plugin;
  - aliased I/O is enabled when any group plugin is present.

  A mixed graph then keeps the aliasing its group plugins need.

## 10. Open questions

1. Should the separate-shrink fallback for long prefill exist from the start,
   or only if milestone D measures it necessary? Proposed: only if measured.
2. Rank 64 variants: needed by any planned engine? Proposed: no until asked.
3. Should the grouped plugin stay as a backend after this lands (for FP16 or
   FP8 GEMMs that the fused path does not cover)? Proposed: keep it; the robot
   engine has no such targets.
