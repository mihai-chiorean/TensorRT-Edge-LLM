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
- **Order of work:** the GEMM at bM = 16 (all decode and verify GEMM traffic)
  together with the GEMV, as one milestone, because decode needs both. Then
  the larger GEMM tiles (bM = 32, 64, 128) for prefill.
- The per-layer-embedding projections (`per_layer_input_gate`,
  `per_layer_projection`) are INT4 but not LoRA targets on the robot engine.
  They keep their non-LoRA variants.

## 3. Kernel design

### 3.1 Contract

For one projection with base weights W (INT4), activation x `[M, K]` and
factors A `[K, r]`, B `[r, N]` (B already carries `lora_alpha / r`):

```
base16 = fp16(x @ dequant(W))                 # exactly today's kernel output
t16    = fp16(sum over chunks c, in order, of fp32(x[:, c] @ A[c, :]))
d16    = fp16(sum over j < r, in order, of float(t16[:, j]) * float(B[j, :]))
out    = fp16(float(base16) + float(d16))
```

This is the rounding of the unfused branch and of `LoraGroupPlugin`: t and the
delta are rounded to FP16, and the final add is exact FP32 followed by one
rounding. Only the summation order inside t and inside d differs from those
paths.

`base16` must stay bit-identical to the non-LoRA variant of the same tactic.
The LoRA work starts after the base value is final, including the last
split-K slice's fold.

Rejected: folding the delta into the MMA accumulator (t as one extra K-tile
with B as its operand). It saves the separate accumulator but rounds
`base + delta` once, which is a different contract from every existing path,
and it moves greedy outputs for no measurable gain. It can be an opt-in later.

### 3.2 Shrink in the same launch: ticketed producers

Every CTA needs the whole `t` row for its output rows, but `A` must be read
from DRAM only once. The kernel splits the shrink into fixed K chunks and lets
the first CTAs to start compute them:

1. At entry, thread 0 takes a ticket with `atomicAdd(&state.ticket, 1)` and
   broadcasts it through shared memory. Tickets are assigned in start order,
   not by `blockIdx`.
2. A CTA whose ticket is below the number of shrink jobs `P` is a producer.
   It computes job `p` before its own main loop: an FP32 partial
   `x[mtile, chunk] @ A[chunk, :r]`, written to `partial[mtile][chunk]` in the
   TensorRT workspace. Then it issues `__threadfence()` and a release
   `atomicAdd(&state.done[mtile], 1)`. If there are more jobs than CTAs, a
   producer takes jobs `p, p + grid, ...`.
3. Every CTA runs its unchanged main loop and base epilogue up to the final
   `base16` fragment.
4. The CTA that writes the final value of a tile (every CTA without split-K;
   the last slice with serial split-K) waits for `done[mtile] == chunks` with
   an acquire load in thread 0, then `__syncthreads()`. It sums the partials in
   chunk order with L2 loads (`ld.global.cg`), rounds to `t16`, computes `d16`
   for its own columns, and stores `out`.
5. Every CTA then does `atomicAdd(&state.exited, 1)`. The CTA that sees
   `grid - 1` resets `ticket`, `done[]` and `exited` to 0 for the next launch.
   The next launch on the stream starts after this kernel completes, so it
   sees the reset. The grid may change between launches (decode and prefill
   use the same plugin), and the reset works for any grid.

Forward progress: a CTA only waits after it has finished its main loop, and
it waits only for producers, which by construction have already started. No
co-residency is needed, so this works for 80-CTA grids on 8 SMs (two waves at
the current occupancy). In practice no CTA waits: producers do the shrink
first, and it takes a few microseconds, against a 20 to 90 µs main loop.

The chunk size is a function of the shape only: 512 K elements (8 K-tiles)
for M <= 64, then 2048. That gives 5 chunks for K = 2560, 4 for K = 2048 and
20 for K = 10240 in decode. **The reduction order of t therefore does not
depend on the SM count or the grid**, which the grouped plugin's
`G`-way split-K could not guarantee (16-SM and 8-SM runs differed). The
producer count `P = min(jobs, grid)` only decides who computes a job, not the
order.

Rejected alternatives:
- **Every CTA computes t itself.** No synchronisation, but `A` is read once
  per CTA. For gate/up that is 80 CTAs x 80 KB = 6.4 MB of L2 reads per
  projection, about +50% on top of each CTA's 184 KB of weights and scales.
- **Cooperative launch with a grid barrier.** This needs every CTA resident.
  The gate/up grid (80 CTAs) does not fit on 8 SMs at the current occupancy.
- **A separate shrink kernel.** That is one launch per target again, 258 per
  decode token. It stays as the fallback for long prefill if the in-kernel
  producers measure slower there (section 7).
- **One shrink for q/k/v together.** This would couple three plugin nodes
  through shared state. Per-target shrinks read the same total A bytes, and x
  (5 KB at M = 1) is cheap to read three times.

### 3.3 Expand in the epilogue

GEMM: after the coalesced epilogue reload (`tCrC_epilogue`, eight consecutive
FP16 values of one row per thread per vector), each thread computes the eight
deltas for its vector with CUDA-core FP32 FMAs in rank order, reading `t16`
from shared memory and `B[0:r, n:n+8]` as 128-bit loads. The CTA's B tile
(`r x 128` FP16, 4 KB at rank 16) is issued as `cp.async` at kernel entry into
a dedicated shared-memory buffer, so it arrives during the main loop. At
bM = 16 that is 16 x 8 FMAs per output vector, which is negligible next to the
main loop. Rows at or beyond M are predicated off as today.

GEMV: warp 0 lanes with `r == 0` own two channels times M rows and already
sum the per-warp partials before storing. They add the delta for those
channels from a `r x 16` B tile staged in shared memory at entry (512 B at
rank 16).

### 3.4 Rank, rank 0 and adapter off

- The rank is a runtime argument of the LoRA variants, `1 <= r <= 16`. Rank
  is not baked. A variant set for `r <= 64` is added only if an engine needs
  it; until then `insert-lora` refuses the fused backend for
  `max_lora_rank > 16` and keeps the MatMul branches.
- **Rank 0 costs nothing.** The plugin reads `r` from the `lora_A` input
  dimensions in `enqueue`. At `r == 0` it launches the existing non-LoRA
  variant with the same tactic, so the kernel, its output and its time are
  the rank-0 engine's. The runtime binds rank-0 inputs for no adapter and for
  modules the adapter does not cover, through `lora_inactive_rank = 0`
  (recorded by the builder when every LoRA-reading layer can skip rank 0;
  this design adds the fused plugin to that set).
- Per-request selection does not change. The adapter switch at prefill
  rebinds the factor pointers and rank dimensions, and decode graphs are
  captured per adapter by `captureBaseGraphWithLoraFanout`, so each graph
  holds the variant and rank it was captured with.

### 3.5 State and workspace

- Persistent per-context state: one small zeroed device block per plugin
  instance, allocated in `attachToContext` outside graph capture and zeroed
  once with `cudaMemsetAsync`, holding `ticket`, `exited` and `done[]`
  (`done` sized for the profile's maximum M-tile count). This follows
  `LoraGroupPlugin`'s barrier state.
- The partials live in the TensorRT workspace (`getWorkspaceSize`), which
  layers share because they run in order on one stream. Size:
  `ceil(Mmax / bMmin) * chunks * bM * 16 * 4` bytes. For down (K = 10240) at
  `Mmax = 8192` with 2048-element chunks that is 2.6 MB.
- Different contexts and different layers never share state. Two graphs
  captured from one context never run concurrently.

### 3.6 INT8 factors (step 3)

Step 3 (being added by another change) keeps the factors in INT8 on the
device. The fused kernels take that format directly:

- Inputs: `lora_A` INT8 `[K, r]` with `lora_A.weight_scale` FP16 `[r]`, and
  `lora_B` INT8 `[r, N]` with `lora_B.weight_scale` FP16 `[N]`. These are the
  tensor names and per-column scales of the `int8_per_channel_v1` transport
  from #73.
- Producers dequantize A and consumers dequantize B as
  `fp16(float(code) * float(scale))`, the expression the runtime uses at load
  today. The registered values, and so the outputs, are bit-identical to the
  INT8 transport path. Bytes halve: about 37 MB per token instead of 69.8 MB.
- A plugin attribute `lora_factor_format` (`0` FP16, `1`
  `int8_per_channel_v1`) selects the variant set; it is fixed per engine. If
  step 3 lands with other binding names, the fused plugin uses those; the
  kernel side does not depend on the names.
- An INT8-resident engine refuses an FP16 adapter file instead of quantizing
  it at load. Requantization changes the delta by 0.56 to 0.77% for the tool
  adapter, so it must be a deliberate `process-lora --weight-format int8` step
  that is qualified like any other adapter.

## 4. MTP

- **Verify (base engine).** Verify rounds run the base engine at
  M = 1 + drafts (at most 8). The GEMM at bM = 16 and the GEMV at M <= 8 are
  the decode variants, so verify is covered by the first milestone with no
  extra work. The verify step costs the same LoRA bytes as a plain decode
  step and no extra launches. The grouped plugin's 15% per-group penalty at
  M > 1 disappears.
- **Draft.** The assistant draft engine has no LoRA inputs. #75 designs draft
  adapters with TensorRT-native MatMul branches from the direct builder. This
  design does not change the draft. If a draft projection is ever INT4 V2,
  the direct builder can emit the fused plugin through the same interface;
  that is not planned. The acceptance loss with the adapter (65 against 71
  verify rounds on a 129-token story) is untouched by kernel work.
- **Losslessness.** Every draft token is still verified against the adapted
  base engine, so MTP output equals non-speculative output with the same
  fused engine.

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
  stays 1.
- Graph input names, dtypes and the dynamic `<stem>.rank` dimension do not
  change in FP16 mode.
- `builder_config` records `lora_backend` and `lora_factor_format`, and
  `lora_inactive_rank = 0` when every LoRA-reading layer is a fused or group
  plugin.
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

Kernel level (C++ gtest against the AOT artifacts, plus the CuPy harness):
1. **Base untouched:** the LoRA variant with `B = 0` gives output
   bit-identical to the non-LoRA variant, for every tactic, at M in
   {1, 2, 4, 8, 16, 33, 128} and for a non-128-aligned N. This proves the base
   path and split-K fold are unchanged.
2. **Delta accuracy:** `out - base16` against an FP64 reference of the
   contract in 3.1, within one FP16 ulp of `d16` plus the propagation of one
   ulp of `t16`. Ranks 1, 8 and 16. INT8 factors use the dequantized values as
   the reference.
3. **Determinism:** repeated launches, different grid orders (swizzle), and a
   forced producer count of 1 and of `jobs` all give bitwise-identical
   output.
4. **Rank 0:** the plugin launches the non-LoRA variant (checked through the
   plugin test by output equality with `Int4GroupwiseGemmPluginV2`).
5. **State reset:** alternating M = 1 and M = 300 launches on one plugin
   instance, and replaying a captured graph 1000 times, keep the output
   correct and leave the counters at zero.

Engine level, on the AGX and then the NX:
6. Adapter off on the fused engine against the rank-0 engine: decode time
   within noise. Greedy output equal on the 9-prompt tool set (both engines
   use the same base kernels, and the rank-0 path launches them unchanged;
   any difference comes from different tactic choices, which the engine build
   log shows).
7. Adapter on: greedy token equality on the 9-prompt tool set and the
   66-check golden set, against the grouped plugin engine and the unfused
   MatMul engine. Report every divergence with its first-divergence position
   and the top-2 logit margin there. Gate: no golden-set or tool-call accuracy
   regression; isolated near-tie divergences are expected and accepted.
8. Cross-device: with `HW_COMPAT=same_compute_capability` the AGX-built engine
   runs on the NX with the same tactics, and the fused LoRA path has no
   SM-count dependence. Its outputs are expected to be bit-identical across
   the two devices. This is checked, not assumed.

## 7. Orin NX (SM87, 8 SMs, 102.4 GB/s)

- **Bandwidth per SM is the same as on the AGX** (12.8 GB/s), and both base
  and LoRA bytes scale with bandwidth, so the relative cost is about the same
  on both devices. NX base step about 43 ms; fused LoRA about 0.9 to 1.1 ms
  with FP16 factors (97.5 to 98%) and about 0.5 to 0.6 ms with INT8 factors
  (about 98.7%). The grouped plugin is projected at 3.4 to 4.0 ms on the NX.
- **No SM-count tuning.** Chunk sizes, job counts and variants depend only on
  the shape. The ticket protocol needs no residency, so the 8-SM grids that
  forced the grouped plugin into rounds are not a special case.
- **Shared memory.** The largest existing config uses about 81 KB. The LoRA
  variants add the B tile (4 KB at bN = 128 and rank 16) and the t tile
  (`bM x 16` FP16, at most 4 KB), so they stay under 90 KB, below the 99 KB
  opt-in floor used for SM86/89 and far below the NX's 163 KB. No per-device
  shared-memory branches.
- **Registers and occupancy.** The epilogue runs after the main loop, so its
  live range does not overlap the main loop's. The GEMV variants keep their
  `min_blocks` budgets (85 registers at `min_blocks = 3`); a LoRA variant that
  spills or loses a resident block at its budget fails the milestone and is
  reworked before it is measured.
- **Clocks.** The removed fixed cost per launch scaled with GPU clock (25 W
  mode at 918 MHz is about 1.4 times the AGX's per-launch cost), so the NX
  gains relatively more from fusion than the AGX.
- **Memory.** No extra device memory beyond the per-context state (under 1
  KB per plugin) and the partial workspace (at most 2.6 MB, shared). INT8
  factors save about 35 MB per resident adapter.
- **Long prefill.** For M > 64 the producers compute bM-row partials on CUDA
  cores before their main loop. If the 800-token TTFT on the NX is worse than
  with a separate shrink kernel (one launch per target, about 258 per
  prefill), prefill above a threshold M uses that kernel instead. The choice
  is per M bucket and still depends only on shape.

## 8. Plan

| Milestone | Content | Gate |
|---|---|---|
| A. Decode kernels | GEMM bM = 16 LoRA variants (all split-K), GEMV LoRA variants M = 1..8, ticket protocol, FP16 factors. CuPy harness and gtest 1 to 5. | Kernel tests pass; microbench per decode shape on the AGX against base + grouped plugin. |
| B. Plugin and insertion | `Int4GroupwiseGemmLoraPlugin`, `insert-lora --lora-backend fused-int4`, builder config, `build_engines.sh`. | Plugin tests; engine builds on the AGX; tests 6 and 7. |
| C. Decode measurement | b8 harness, nsys node and graph modes. | Decode ms/token against rank 0 and the grouped plugin; zero LoRA kernels in the decode trace. |
| D. Prefill tiles | bM = 32, 64, 128 LoRA variants; producer path for M > 64 or the shrink-kernel fallback. | TTFT at 20, 100, 400, 800 and 1600 tokens. |
| E. INT8 factors | `lora_factor_format = 1` variants, manifest format check. Coordinated with step 3. | Bit-equality with the INT8 transport path; decode bytes halved in nsys. |
| F. NX | Build on the NX (or run the AGX-built engine), repeat C, D and test 8. | NX decode and TTFT against rank 0. |

Milestones A to C are the decode result. D to F follow in that order.

## 9. Risks

- **Spin-wait semantics in CuTe DSL.** The kernel needs acquire loads,
  release atomics and `ld.global.cg`. The existing serial split-K path uses
  `atomic_add`, `atomic_exch` and `fence_acq_rel_gpu`, so the primitives
  exist; inline PTX covers anything missing.
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
  rank-0 binding path and the insertion plumbing. If that stack changes, this
  design follows it; the kernels do not depend on it.

## 10. Open questions

1. Should the separate-shrink fallback for long prefill exist from the start,
   or only if milestone D measures it necessary? Proposed: only if measured.
2. Rank 64 variants: needed by any planned engine? Proposed: no until asked.
3. Should the grouped plugin stay as a backend after this lands (for FP16 or
   FP8 GEMMs that the fused path does not cover)? Proposed: keep it; the robot
   engine has no such targets.
