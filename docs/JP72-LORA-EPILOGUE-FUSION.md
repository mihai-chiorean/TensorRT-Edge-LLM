# LoRA fused into the W4A16 GEMM/GEMV epilogue: design and measurements

Status: revised after implementation, 2026-11-13. Fork `main` 5a56baaa. This
is step 4 of the LoRA decode speed-up plan. It builds on the grouped LoRA
plugin branches, stacked on `main` in this order: `lora-group-plugin` (2ed53e5
`LoraGroupPlugin`, then b95e8b4 for 8-SM GPUs), `lora-group-insert` (55b3358,
`insert-lora --lora-group-plugin`) and `lora-inactive-rank` (74f7de7,
engine-recorded `lora_inactive_rank`). On INT4 engines it replaces that plugin.

The first version of this document proposed computing the shrink inside the
INT4 kernel and projected 97 to 98% of rank-0 decode throughput. That design
was built and measured, and it did not reach the projection. This revision
describes what was built instead (section 3), reports the measurements
(section 7), and explains why the projection failed and why about 95% is the
realistic floor for this approach (section 8). Sections 1, 2, 4 and the
equality strategy keep their original reasoning where it still holds.

Summary of the result on the AGX Orin (MAXN, rank-16 tool adapter, same
session for every engine):

| Engine | Decode, adapter off | Decode, adapter on | LoRA cost per token |
|---|---:|---:|---:|
| rank 0 (no LoRA inputs) | 46.19 tok/s (100%) | - | - |
| grouped `LoraGroupPlugin` | 44.34 tok/s (96.0%) | 41.00 tok/s (88.8%) | 1.83 ms |
| fused (this design) | 44.53 tok/s (96.4%) | 41.83 tok/s (90.6%) | 1.44 ms |

Prompt processing with the adapter costs +27 ms on a ~1,600-token prompt with
the fused engine, against +66 ms with the grouped plugin (section 7.2).

## 1. Problem

The Gemma 4 E4B robot engine runs every attention and MLP projection through
`Int4GroupwiseGemmPluginV2` (W4A16, Q4_0 group 32). Dynamic LoRA adds
`out += fp16(fp16(x @ A) @ B)` to 258 of those projections.

The grouped plugin costs 168 extra launches per decode token (about 11.9 µs
each, 2.0 ms in all). The launch cost fits `6.5 µs + bytes / 75 GB/s`: half of
it is fixed per launch, and the other half moves the 69.8 MB of FP16 factors
at half the bandwidth the INT4 kernels reach. Each launch also passes a grid
barrier and does a read-modify-write of the base output. In prefill the same
plugin adds 336 to 408 launches outside CUDA graphs (about +7 ms TTFT on a
short prompt) and about 42 ms of LoRA kernels per 800-token prompt.

On the Orin NX (robot, 25 W) the grouped plugin costs +2.94 ms per decode
token, and the adapter raises TTFT from 57/487/935 ms to 68/568/1094 ms at
20/800/1600 prompt tokens. Prefill matters on the NX as much as decode.

## 2. Which kernels take the epilogue

With group 32, `int4RouteToGemv` sends an M = 1 row to the GEMM when
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

Decisions (unchanged):
- **Both kernels take the epilogue.** `Int4Fp16GemmAmpere` carries q, o,
  gate, up and down in decode, MTP verify (M = 2..8) and prefill.
  `Int4Fp16GemvAmpere` carries k and v in decode, and k and v sliding in MTP
  verify.
- Every GEMM tactic and every GEMV M has a LoRA variant, because the decode
  and prefill profiles autotune different tactics.
- The per-layer-embedding projections are INT4 but not LoRA targets. They
  keep their non-LoRA variants.

## 3. Design as built

Two plugins replace each grouped-plugin launch and the base GEMM it followed:

```
x (unsmoothed) ──> LoraShrinkPlugin (one per shared activation, 1 to 4 targets)
                     │  t_i = fp16(x @ A_i)   [M, 16], zeros past the rank
                     v
x (as today) ────> Int4GroupwiseGemmLoraPlugin (one per target)
                     base16 = unchanged INT4 GEMM/GEMV
                     out    = fp16(float(base16) + float(fp16(t_i @ B_i)))
```

Per decode token that is 168 shrink launches (24 q/k/v groups of three, 42
gate/up pairs, 102 singles: q on the shared-KV layers, o and down) and no
extra launch for the expand.

### 3.1 Contract

For one projection with base weights W (INT4), activation x `[M, K]` and
factors A `[K, r]`, B `[r, N]` (B already carries `lora_alpha / r`):

```
base16 = the existing kernel's FP16 output for this tactic, unchanged
t16    = fp16(T), T = x @ A with the summation tree of the shrink kernel
         (section 3.2), exactly specified for M <= 8
d16    = fp16(D),  D = sum over j = 0..r-1, in order, of
         fma.rn.f32(float(t16[:, j]), float(B[j, :]), D)
out    = fp16(float(base16) + float(d16))
```

This is the rounding of the unfused branch and of `LoraGroupPlugin`: t and the
delta are rounded to FP16, and the final add is exact FP32 followed by one
rounding. Only the summation order inside t and d differs from those paths.
`base16` is bit-identical to the non-LoRA variant of the same tactic on the
same inputs.

Rejected (unchanged): folding the delta into the MMA accumulator. It rounds
`base + delta` once, a different contract from every existing path.

### 3.2 Shrink: `LoraShrinkPlugin`

One plugin node per group of INT4 targets that read the same activation. It
reads the unsmoothed activation once and writes one `t_i` `[.., 16]` FP16
output per target. All buffers are ready before the INT4 kernel starts, so the
GEMM needs no synchronisation, no atomics and no state.

Decode path, M <= 8 (`loraShrinkKernel`, CUDA cores):
- Grid (K / 256 chunks, targets, ceil(M / 16)), 128 threads. Each thread pair
  owns slot q (0..63) and half h (ranks 8h..8h+7).
- Tree, fixed by the shape only: `S[q,h]` is an FMA chain over
  `k = 256 c + q + 64 i`; a warp XOR butterfly (16, 8, 4, 2) reduces the slots
  of a warp; the four warps are summed `((W0 + W1) + W2) + W3`. The block's
  FP32 partial goes to the workspace.
- The last block to finish a (target, row block) sums the chunk partials in a
  grouped order (`Q[g]` over chunks `j ≡ g mod 8` in order, then `T = ΣQ` in
  order), rounds to FP16 and resets its counter. This is the threadfence
  reduction pattern; the counter is zero again when the kernel exits, so graph
  replays and eager launches need no reset.
- Bit-exact against a NumPy emulation of the tree. The tree does not depend on
  the SM count or the grid, so it is the same on the AGX and the NX.

Prefill path, M >= 9 (`loraShrinkPrefillKernel`, tensor cores):
- WMMA FP16 with FP32 accumulation, 64 rows per block, K chunks of 512,
  cooperative 64-K tiles with register double buffering. Grid
  (K / 512, ceil(M / 64)); the last block per row block sums the chunks in
  order.
- Deterministic (fixed order), but the order inside an MMA is the hardware's,
  so the test is a bound against FP64, not a bit-exact emulation.
- 4 to 5 times faster than the CUDA-core tree at prefill M: three targets at
  M = 800, K = 2560 take 62.6 µs against 279 µs.

The launcher skips the kernel when every target's rank is 0, so the adapter-off
cost of a shrink node is the empty TensorRT node.

Requires K to be a multiple of 64 (all Gemma 4 E4B shapes are). Workspace:
the partials, sized as the maximum over the M buckets {min(maxM, 4),
min(maxM, 8), maxM}, because the chunk count changes at the bucket boundary.

### 3.3 Expand in the epilogue

The LoRA variant of each INT4 kernel copies its rows of `t16` (`bM x 16`) and
its B tile (`16 x bN`) into shared memory right after issuing the main-loop
prologue, with 128-bit loads, rows past M and ranks past r zero-filled. The
copy overlaps the prologue's own loads, and the next CTA barrier orders it
before any read. The epilogue then computes each output vector's delta from
shared memory with FP32 FMAs in rank order and adds it to `base16` before the
store. It reads no global memory beyond the base kernel's own.

- GEMM: only the final writer stages and expands (every CTA without split-K,
  the last slice with serial split-K). Shared memory grows by
  `bM * 16 * 2 + 16 * bN * 2 + 32` bytes (4.6 KB at bM = 16). For bM = 16 the
  LoRA variants are compiled with `min_blocks_per_mp = 5` to keep the register
  allocation from costing a resident block.
- GEMV: one 8 x 16 `t16` tile and a 16 x 16 B tile next to the reduction
  buffer; the lanes that already cast the final FP16 output add the delta.

Measured epilogue cost at decode (AGX, graph replay, L2-resident; positive is
slower than the base variant): GEMV +0.3 to +0.5 µs at M = 1 (+1.5 µs at
M = 8); q -1.1 to +1.9 µs (within the bench's noise); o +1.7 µs; down
+0.3 µs; gate/up +3.4 µs. In the engine the epilogues add about
0.3 to 0.4 ms per token.

### 3.4 Rank, rank 0 and adapter off

- The rank is a runtime argument, `1 <= r <= 16`. `Int4GroupwiseGemmLoraPlugin`
  refuses a profile whose maximum rank exceeds 16 in `configurePlugin`, so an
  over-rank engine fails at build time.
- **Rank 0.** The GEMM plugin reads r from the `lora_B` dimensions in
  `enqueue` and launches the non-LoRA variant of the same tactic at r = 0, so
  its output is bit-identical to `Int4GroupwiseGemmPluginV2`. The shrink
  launcher skips at r = 0. The runtime binds rank-1 allocations reshaped to
  rank 0 (`lora_inactive_rank = 0`).
- Adapter off is not free at engine level, but the cost is not in the LoRA
  kernels: see section 7.3.
- Per-request selection does not change. Decode graphs are captured per
  adapter by `captureBaseGraphWithLoraFanout`.

### 3.5 State and workspace

- `Int4GroupwiseGemmLoraPlugin` has no state and no workspace beyond the
  existing split-K locks.
- `LoraShrinkPlugin` allocates its zeroed counters per context in
  `attachToContext` (outside capture) and keeps its partials in the TensorRT
  workspace, which layers share because they run in order on one stream. The
  counters self-reset, as described in 3.2.

### 3.6 INT8 factors (step 3)

Not implemented yet. The split makes the INT8 path simpler than the first
design assumed:
- A is read only by the shrink kernel, which dequantizes
  `fp16(float(code) * float(scale))` as it loads.
- B is read only by the smem fill at kernel entry, which can dequantize while
  copying into the same FP16 tile; the epilogue does not change.
- The runtime ABI work (typed INT8 bindings, 1-D scale bindings, inactive
  shapes, validation by dtype) is as in the first revision and is coordinated
  with step 3.
- Bytes halve. Section 8 explains why that is worth less than the first
  revision expected: the decode shrink is latency-bound, not
  bandwidth-bound, so INT8 A saves well under half of the shrink time.

### 3.7 Rejected alternatives, with data

All on the AGX, same harness or same engine profile as the final design.

- **v5: shrink inside the INT4 kernel (the first revision's design).** CTAs
  claimed (M-tile, K-chunk) jobs at entry, published FP32 partials, and the
  final writer acquired and reduced them before its epilogue. Correct and
  bit-exact, but each dependent round trip (claim, publish, acquire) costs
  about 1 to 1.5 µs on Orin, and the final writer's wait and reduction land on
  the kernel's critical path: 4 to 6 µs per target, over 258 targets per token.
  The engine decoded at 41.1 tok/s with the adapter, the same as the grouped
  plugin.
- **Job warp.** A dedicated producer warp per CTA for the shrink jobs. The
  extra warp and its registers cost a resident block on bM = 16 tiles, and the
  80-CTA gate/up grids ran in a second wave (CTAs entering about 71 µs late).
- **Delta mid-loop.** Computing the delta in registers during the main loop to
  hide it. Raised register pressure for the main loop; loads behind
  per-iteration branches serialized. Slower than the epilogue form.
- **Inline shrink for single-target groups.** Shrink computed inside the GEMM
  for the 102 single-target groups, separate plugin for the rest. Measured
  -0.09 ms per token in the engine against the all-separate form, within
  noise, at the cost of 32 more AOT variants and a second synchronisation
  path. Reverted.
- **Smaller or larger decode chunks.** 128 and 512 were tried against 256.
  The L2-resident microbenchmark favoured 512; the in-engine profile, which is
  decisive, gave 954 µs of shrink per token at 256 against 1,040 µs at 128 and
  more at 512.

## 4. MTP

- **Verify (base engine).** Verify runs the base engine at M = verify width,
  at most 8 on the batch-one robot engine. The decode shrink tree and the LoRA
  GEMM/GEMV variants cover M <= 8, so a verify step costs the same LoRA
  launches as a plain decode step.
- **Draft.** The assistant draft engine has no LoRA inputs (#75 designs draft
  adapters separately). Use AGX-built drafts; NX-built MTP drafts are
  currently broken for reasons unrelated to this work.
- **Equality.** MTP output is the greedy output of the verify computation,
  not necessarily token-identical to non-speculative decode (different kernels
  at different M, as today without LoRA). Measured, not assumed (section 6).
- MTP throughput with the fused engine has not been measured yet.

## 5. Engine, plugin and adapter interfaces

### 5.1 Plugins

`LoraShrinkPlugin` version 1, `cpp/plugins/loraShrinkPlugin/`:

| Index | Input or output | Type and shape |
|---|---|---|
| in 0 | unsmoothed activation | FP16 `[M, K]` or `[b, s, K]` |
| in 1..n | `<stem_i>.lora_A.weight` | FP16 `[K, rank_i]` |
| out 0..n-1 | `t_i` | FP16 `[.., 16]` |

Attribute `num_targets` (1 to 4). The maximum M comes from the profiles.

`Int4GroupwiseGemmLoraPlugin` version 1, in
`cpp/plugins/int4GroupwiseGemmPluginV2/`, shares its code, tactic table, lock
workspace and GEMV routing with `Int4GroupwiseGemmPluginV2`, which does not
change:

| Index | Input | Type and shape |
|---|---|---|
| 0 | activation (smoothed, as today) | FP16 `[M, K]` or `[b, s, K]` |
| 1 | fragment weights | INT8 `[rows, 512]` |
| 2 | scales | FP16 `[K / G, N]` |
| 3 | `t` from the shrink | FP16 `[.., 16]` |
| 4 | `<stem>.lora_B.weight` | FP16 `[rank, N]` |

The output is a plain output, not an alias, so these engines do not need
`kALIASED_PLUGIN_IO_10_03`. Delta and base are added inside the kernel before
any consumer, so the Gemma 4 PLE projection's output scale applies to both.

Tactics: one LoRA variant per GEMM configuration and group size, and one per
GEMV M from 1 to 8. TensorRT autotunes with the LoRA inputs at the profile's
optimum rank.

### 5.2 Insertion and build

- `insert-lora --lora-fused-int4` groups the selected
  `Int4GroupwiseGemmPluginV2` nodes by their unsmoothed activation (up to 4 per
  group), emits one `LoraShrinkPlugin` per group, rewrites each node to
  `Int4GroupwiseGemmLoraPlugin` and appends `t_i` and `lora_B`. Non-INT4
  targets keep MatMul branches. Graph input names, dtypes and the dynamic
  `<stem>.rank` dimension do not change.
- The builder allows `lora_inactive_rank = 0` when every LoRA-reading layer is
  a group, fused or shrink plugin, and enables aliased plugin I/O only when a
  group plugin is present.
- Builds need `-DENABLE_CUTE_DSL=ALL` for the LoRA variants.

### 5.3 Adapter manifest and base identity (#69, #72)

Unchanged from the first revision: processed FP16 adapter files are
byte-identical for the fused engine; the engine file changes, so the base
identity changes and manifests are re-issued against it; INT8-resident engines
accept only `int8_per_channel_v1` files.

## 6. Equality strategy

Bit equality with the grouped plugin or the MatMul branches is not possible
(different summation order inside t and d). The strategy proves what can be
proven exactly and measures the rest.

Done (kernel and plugin level, AGX):
1. **Base untouched.** At rank 0 the plugin output equals
   `Int4GroupwiseGemmPluginV2`, bit for bit, on the GEMM and GEMV routes.
2. **t exactly, M <= 8.** The NumPy emulation of the decode tree reproduces
   the kernel's `t16` bit for bit (ranks 1, 7, 16; K = 2048, 2112, 2560,
   10240; M = 1, 2, 3, 5, 8). For M >= 9 the tensor-core shrink is checked
   against FP64 with a tolerance.
3. **Output exactly.** From the kernel's own `t16`, a host epilogue (FMA in
   rank order, then the FP32 add) reproduces every LoRA variant's output bit
   for bit across GEMM tiles, split-K, GEMV M = 1..8, M in {1, 5, 8, 16, 17,
   32, 64, 65, 100, 200, 300, 800}, and ranks 7, 8 and 16.
4. **Plugin tests.** Shrink plugin with three targets at ranks 16, 0 and 8
   and M = 1, 5, 80; GEMM plugin rank cap and serialization.

Still to run (engine level, AGX then NX): adapter-off output against rank 0;
adapter-on greedy equality on the 9-prompt tool set and the 66-check golden
set against the grouped and MatMul engines, with first-divergence position and
top-2 margin; MTP against autoregressive decode; cross-device equality of an
AGX-built engine run on the NX; repeated-replay and `compute-sanitizer`
checks of the shrink counters.

## 7. Measurements (AGX Orin, MAXN)

Same session, same adapter, engines built with `--max-lora-rank 16
--max-input-len 8192 --max-kv-cache-capacity 16384 --max-batch-size 1`.
Decode is 257 tokens on two prompts with CUDA graphs; TTFT is the median of
5 uncached requests generating one token.

### 7.1 Decode

| Engine | Off tok/s | On tok/s | Off ms/token | On ms/token | On vs rank 0 |
|---|---:|---:|---:|---:|---:|
| rank 0 | 46.19 | - | 21.65 | - | 100% |
| grouped | 44.34 | 41.00 | 22.56 | 24.39 | 88.8% |
| fused | 44.53 | 41.83 | 22.46 | 23.90 | 90.6% |

The fused engine is 2.0% faster than the grouped engine with the adapter on.
The LoRA cost relative to the same engine with the adapter off is 1.44 ms per
token (fused) against 1.83 ms (grouped).

Where the fused 1.44 ms goes (nsys, kernel time per decode step):
- shrink kernels: about 0.95 ms (168 launches, 4.5 to 6.6 µs each);
- expand epilogues: about 0.3 to 0.4 ms over 258 kernels;
- the remaining ~0.1 ms is within noise; the gaps around shrink nodes are
  under 1 µs, and total inter-kernel gaps match the adapter-off profile.

### 7.2 Prompt processing (TTFT)

| Engine | Short, off -> on | ~1,600 tokens, off -> on |
|---|---:|---:|
| rank 0 | 44.3 ms | 551.0 ms |
| grouped | 42.8 -> 52.1 ms (+9.3) | 553.9 -> 620.4 ms (+66.5) |
| fused | 45.4 -> 52.7 ms (+7.3) | 476.6 -> 503.5 ms (+26.9) |

The fused engine halves the long-prompt LoRA cost. Its adapter-off long TTFT
is lower than the rank-0 engine's; that is not explained yet (the rank-0
engine is an older build, and its long TTFT ranged 516 to 555 ms across
sessions), so compare off -> on within each engine, not across engines.

At prefill M the expand epilogue still costs +4 to +8% of the base GEMM at
M = 800 (gate/up 1805 -> 1946 µs, down 1840 -> 1913 µs) and more at mid M
(+29% for q at M = 300, rank 8). It runs on CUDA cores; moving it onto the MMA
is the next prefill item.

### 7.3 Adapter-off overhead (not a kernel cost)

Both LoRA engines run 0.8 to 0.9 ms per token slower than rank 0 with the
adapter off (96.0% and 96.4%). nsys shows the same kernels and the same
kernel time as rank 0 (2,632.7 against 2,636.7 ms busy over the profile), but
about 0.8 ms more host time between decode steps: the step-boundary gap after
the final norm grows from 899 µs to 1,674 µs. It is the same on the grouped
and fused engines, so it comes from the runtime's LoRA handling, not from
either plugin. It has been handed to the runtime work separately. Removing it
would raise the fused adapter-on figure from 90.6% to about 93.8% of rank 0.

## 8. Why 97 to 98% did not hold, and the realistic floor

The first revision projected about 97 to 98% of rank-0 throughput with FP16
factors by assuming that, once the extra launches were gone, LoRA would cost
only its factor bytes at the INT4 kernels' bandwidth (69.8 MB per token, about
0.4 to 0.5 ms on the AGX). Three assumptions failed:

1. **A dependency point is not free, inside a kernel or between kernels.**
   The projection treated the in-kernel shrink as free because its work is
   small. On Orin each dependent round trip between CTAs costs 1 to 1.5 µs,
   and the shrink needs several before the final writer can expand. Measured,
   that was 4 to 6 µs per target on the kernel's critical path, about as much
   as a separate launch. Whether t is produced by a separate launch or by an
   in-kernel handshake, every point where one set of CTAs waits for another
   costs about 5 to 10 µs on this hardware.
2. **The shrink is latency-bound, not bandwidth-bound.** A decode shrink moves
   80 to 240 KB of A. At that size the kernel takes 4.5 to 6.6 µs, which is
   18 to 50 GB/s; it never reaches the INT4 kernels' bandwidth. Halving the
   bytes (INT8) would save only part of the shrink time.
3. **The tail is not free.** Staging t and B and computing the delta lengthen
   every kernel a little, up to 3.4 µs each, about 0.3 ms per token in all.
   The first revision counted only the bytes.

The adapter-off host overhead (7.3) was also not foreseen, but it is outside
this design.

The floor for this approach follows from the dependency count. t depends on
the full activation, and the 258 targets share 168 distinct activations, so
there are at least 168 dependency points per token. At the fastest measured
shrink (about 4.5 µs) that is 0.75 ms, plus about 0.3 ms of epilogues: about
1.05 ms on a 21.65 ms step, or **about 95% of rank 0** once the host overhead
is gone. The current kernels are at 1.44 ms (about 93.8% without the host
overhead), so roughly 0.4 ms is left to gain in the shrink and epilogue.

Getting meaningfully above 95% needs fewer dependency points, not faster
kernels: for example computing the q/k/v and gate/up shrinks in the kernel
that produces their activation (the RMSNorm, generated by TensorRT today), or
the down shrink in the activation kernel. That changes kernels this fork does
not own and is not proposed here.

## 9. Orin NX (SM87, 8 SMs, 102.4 GB/s)

Not measured yet. What carries over from the AGX:
- No SM-count tuning. The shrink tree, chunking and variants depend only on
  the shape; the decode tree gives the same `t16` on 8 and 16 SMs.
- No residency requirement: the GEMM has no inter-CTA wait, and the shrink's
  last-block reduction needs none either.
- Shared memory: +4.6 KB per bM = 16 GEMM CTA (within the SM87 budget at the
  register-capped occupancy); GEMV +0.8 KB.
- The shrink is latency-bound, and the NX at 25 W runs a lower clock, so its
  fixed costs should be relatively larger than on the AGX. A rough expectation
  is a LoRA cost of 2.0 to 2.5 ms per token against the grouped plugin's
  2.94 ms, on a base step of about 43 ms (about 94 to 95% before any host
  overhead). This is an estimate to be replaced by the NX measurement.
- The prefill gain should carry over in proportion: the grouped plugin adds
  81 ms at 800 tokens and 159 ms at 1,600 tokens on the NX.

The NX measurements will be batched into one supervised window: decode off
and on, TTFT at 20/800/1600, for rank 0, grouped and fused, with AGX-built MTP
drafts.

## 10. Status and next steps

| Item | Status |
|---|---|
| LoRA GEMM and GEMV variants, smem expand epilogue | Built; bit-exact tests pass on the AGX |
| `LoraShrinkPlugin`, decode tree and tensor-core prefill | Built; tests pass |
| `Int4GroupwiseGemmLoraPlugin`, `insert-lora --lora-fused-int4`, builder predicates | Built; plugin and insertion tests pass |
| AGX decode and TTFT | Measured (section 7) |
| Engine-level equality, sanitizer, MTP | To run |
| Expand on MMA for prefill M | Next |
| INT8 factors | With step 3 |
| NX | Batched window to be scheduled |

The implementation branch will be split into reviewable PRs (kernels, shrink
plugin, GEMM plugin, insertion and builder) with the debug knobs and
superseded variants removed.

## 11. Risks

- **Shrink counters left non-zero by a faulting kernel.** A fault is a sticky
  context error, so the context is lost anyway; counters are zeroed again in
  `attachToContext`.
- **Tactic choice moves.** Autotuning with the LoRA inputs can choose a
  different split-K than the base engine, which changes base rounding between
  engines (as any rebuild can).
- **AOT build time and size grow** for the `int4_fp16_gemm` group (one LoRA
  variant per configuration).
- **Dependency on the grouped-plugin branches** for `lora_inactive_rank`, the
  rank-0 binding path and the insertion plumbing.

## 12. Open questions

1. Should the grouped plugin stay as a backend for FP16 or FP8 GEMMs the fused
   path does not cover? Proposed: keep it; the robot engine has no such
   targets.
2. Rank 64 variants? Proposed: not until an engine needs them.
3. Is the shrink-into-producer route (section 8) worth pursuing after the NX
   numbers? It is the only way found above about 95%.
