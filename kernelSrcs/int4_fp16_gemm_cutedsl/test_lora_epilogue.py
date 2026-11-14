# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bit-exact check of the INT4 GEMV/GEMM dynamic-LoRA epilogue (needs a GPU).

For each case:
- the plugin's shrink kernel (compiled from ``cpp/kernels/loraKernels`` with NVRTC)
  must equal ``lora_reference.lora_t16`` bit for bit for M <= 8, and an FP64
  reference within tolerance for M >= 9 (tensor-core path); its counters must
  read zero afterwards;
- the LoRA variant's output must equal ``lora_reference.lora_out`` computed from
  the non-LoRA variant's output (the same tactic) and the kernel's ``t16``, bit
  for bit;
- repeated eager launches and CUDA-graph replays must reproduce that output bit
  for bit, and a replay after B changes must follow the new B.

    python test_lora_epilogue.py --kernel gemv --cases "1,512,2560;8,512,2560" --rank 16
    python test_lora_epilogue.py --kernel gemm --cases "1,10240,2560" --bM 16 --split_k 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_saved_argv = list(sys.argv)
sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda  # noqa: E402
import cupy as cp  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import mark_lock_1d, mark_row_major_2d  # noqa: E402
from int4_reference import make_int4_inputs, repack_b_for_tile  # noqa: E402
from lora_reference import lora_out, lora_t16  # noqa: E402


def lora_chunk_k(m):
    return 256


_SHRINK_WRAPPER = r"""
extern "C" __global__ void __launch_bounds__(128) shrink_test(const __half* x, int m, int k, int nt,
    const __half* a0, __half* t0, const __half* a1, __half* t1, const __half* a2, __half* t2, int rank,
    float* partial, int* counter)
{
    trt_edgellm::kernels::LoraShrinkParams p{};
    p.x = x; p.m = m; p.k = k; p.numTargets = nt;
    p.target[0] = {a0, t0, rank}; p.target[1] = {a1, t1, rank}; p.target[2] = {a2, t2, rank};
    p.partial = partial; p.counter = counter;
    if (m >= trt_edgellm::kernels::kLoraShrinkPrefillMinM)
        trt_edgellm::kernels::loraShrinkPrefillBody(p);
    else
        trt_edgellm::kernels::loraShrinkBody(p);
}
"""


def _shrink_module():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", ".."))
    d = os.path.join(root, "cpp/kernels/loraKernels")
    h = open(os.path.join(d, "loraShrinkKernels.h")).read()
    cuh = open(os.path.join(d, "loraShrinkKernels.cuh")).read().replace('#include "loraShrinkKernels.h"', "")
    # NVRTC has no <stdint.h>.
    src = "typedef int int32_t;\ntypedef long long int64_t;\n" + h + cuh
    return cp.RawModule(code=src + _SHRINK_WRAPPER, options=("-std=c++17",), name_expressions=None)


_SHRINK = None


def run_shrink(x_d, a_list, rank, stream):
    """t16 for each A in a_list (1 or 3 targets) with the plugin's shrink kernel."""
    global _SHRINK
    if _SHRINK is None:
        _SHRINK = _shrink_module().get_function("shrink_test")
    M, K = x_d.shape
    if M >= 9:  # tensor-core path: grid (K chunks, M / 64 row blocks); chunk as loraShrinkPrefillChunkK
        ck = (128 if K <= 4096 else 256) if M <= 64 else 512
        chunks = (K + ck - 1) // ck
        grid = (chunks, (M + 63) // 64, 1)
    else:
        ck = lora_chunk_k(M)
        chunks = (K + ck - 1) // ck
        grid = (chunks, len(a_list), (M + 15) // 16)
    zb = (M + 15) // 16
    nt = len(a_list)
    ts = [cp.full((M, 16), np.nan, cp.float16) for _ in range(nt)]
    part = cp.zeros((nt * chunks * M * 16,), cp.float32)
    cnt = cp.zeros((nt * zb,), cp.int32)
    a3 = list(a_list) + [a_list[0]] * (3 - nt)
    t3 = ts + [ts[0]] * (3 - nt)
    args = (x_d, np.int32(M), np.int32(K), np.int32(nt), a3[0], t3[0], a3[1], t3[1], a3[2], t3[2],
            np.int32(rank), part, cnt)
    _SHRINK(grid, (128,), args, stream=stream)
    return ts, cnt, (lambda: _SHRINK(grid, (128,), args, stream=stream))


def run_case(kind, M, N, K, rank, group_size, seed, iters, scale_b, cfg):
    rng = np.random.default_rng(seed)
    act, qweight, scales = make_int4_inputs(M, N, K, group_size=group_size, device="cpu")
    rep = repack_b_for_tile(qweight, N, K, 128, 64)
    a_h = act.numpy().astype(np.float16)
    x_h = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    la_h = [(rng.standard_normal((K, rank)) * 0.02).astype(np.float16) for _ in range(3)]
    lb_h = (rng.standard_normal((rank, N)) * scale_b).astype(np.float16)

    cstream = cp.cuda.Stream(non_blocking=True)
    cstream.use()
    stream = cuda.CUstream(cstream.ptr)
    a_d = cp.asarray(a_h)
    qw_d = cp.asarray(rep.contiguous().numpy()).view(cp.uint32)
    sc_d = cp.asarray(scales.numpy().astype(np.float16))
    x_d = cp.asarray(x_h)
    la_d = [cp.asarray(a) for a in la_h]
    lb_d = cp.asarray(lb_h)
    base_d = cp.zeros((M, N), cp.float16)
    out_d = cp.zeros((M, N), cp.float16)
    ok = True

    # Shrink: three targets sharing x (q/k/v-like); each t16 must equal the host tree bit for bit.
    ts, cnt, launch_shrink = run_shrink(x_d, la_d, rank, cstream)
    cstream.synchronize()
    chunk = lora_chunk_k(M)
    for i in range(3):
        got = cp.asnumpy(ts[i])
        if M >= 9:
            # Prefill path: tensor-core summation, checked against FP64.
            ref = (x_h.astype(np.float64) @ la_h[i].astype(np.float64))
            err = np.abs(got[:, :rank].astype(np.float64) - ref)
            if not (err <= 2e-3 + 4e-3 * np.abs(ref)).all() or got[:, rank:].any():
                ok = False
                print(f"  PREFILL SHRINK off target {i}: max err {err.max():.4g}")
            continue
        want = np.zeros((M, 16), np.float16)
        want[:, :rank] = lora_t16(x_h, la_h[i], chunk)
        if (got.view(np.uint16) != want.view(np.uint16)).any():
            ok = False
            bad = np.argwhere(got.view(np.uint16) != want.view(np.uint16))[:3]
            print(f"  SHRINK MISMATCH target {i}: " + ", ".join(f"[{a},{b}] got {got[a, b]} want {want[a, b]}"
                                                         for a, b in bad))
    if cp.asnumpy(cnt).any():
        ok = False
        print("  SHRINK counters not reset")
    t_d = ts[0]
    t16_h = cp.asnumpy(t_d)

    tA, tQ, tS = mark_row_major_2d(a_d), mark_row_major_2d(qw_d), mark_row_major_2d(sc_d)
    tB, tO = mark_row_major_2d(base_d), mark_row_major_2d(out_d)
    tT, tLB = mark_row_major_2d(t_d), mark_row_major_2d(lb_d)
    if kind == "gemv":
        from int4_fp16_gemv_ampere import (Int4Fp16GemvAmpere, Int4Fp16GemvLoraAmpere,
                                           _gemv_defaults)
        u2, eager, minb = _gemv_defaults(M)
        kw = dict(m=M, group_size=group_size, unroll2=u2, u2_eager=eager, min_blocks=minb)
        base_k = cute.compile(Int4Fp16GemvAmpere(**kw), tA, tQ, tS, tB, stream)
        t0 = time.time()
        lora_k = cute.compile(Int4Fp16GemvLoraAmpere(**kw), tA, tQ, tS, tO, tT, tLB, cutlass.Int32(rank), stream)
        ct = time.time() - t0

        def launch_base():
            base_k(tA, tQ, tS, tB, stream)

        def launch_lora():
            lora_k(tA, tQ, tS, tO, tT, tLB, cutlass.Int32(rank), stream)
    else:
        from int4_fp16_gemm_ampere import Int4Fp16GemmAmpere, Int4Fp16GemmLoraAmpere
        sk = cfg["split_k"]
        locks_d = cp.zeros(max(1, ((M + cfg["bM"] - 1) // cfg["bM"]) * ((N + 127) // 128)), cp.int32)
        tL = mark_lock_1d(locks_d)
        kw = dict(cta_tiler_mnk=(cfg["bM"], 128, 64), num_stages=cfg["stages"],
                  group_size=group_size, split_k=sk)
        sw = cutlass.Int32(cfg["swizzle"])
        base_k = cute.compile(Int4Fp16GemmAmpere(**kw), tA, tQ, tS, tB, tB, tL, sw, stream)
        t0 = time.time()
        lora_k = cute.compile(Int4Fp16GemmLoraAmpere(**kw), tA, tQ, tS, tO, tO, tL, sw, tT, tLB,
                              cutlass.Int32(rank), stream)
        ct = time.time() - t0

        def launch_base():
            locks_d.fill(0)
            base_k(tA, tQ, tS, tB, tB, tL, sw, stream)

        def launch_lora():
            locks_d.fill(0)
            lora_k(tA, tQ, tS, tO, tO, tL, sw, tT, tLB, cutlass.Int32(rank), stream)

    launch_base()
    launch_lora()
    cstream.synchronize()
    base_h = cp.asnumpy(base_d)
    out_h = cp.asnumpy(out_d)
    ref = lora_out(base_h, t16_h[:, :rank], lb_h)
    neq = (out_h.view(np.uint16) != ref.view(np.uint16))
    if neq.any():
        ok = False
        idx = np.argwhere(neq)[:5]
        print(f"  MISMATCH {neq.sum()} of {neq.size}: " + ", ".join(
            f"[{i},{j}] got {out_h[i, j]} want {ref[i, j]} base {base_h[i, j]}" for i, j in idx))
    first = out_h.copy()
    for _ in range(iters):
        launch_shrink()
        launch_lora()
    cstream.synchronize()
    if (cp.asnumpy(out_d).view(np.uint16) != first.view(np.uint16)).any():
        ok = False
        print("  NONDETERMINISTIC across launches")
    if cp.asnumpy(cnt).any():
        ok = False
        print("  SHRINK counters not reset after repeats")

    # Timing: CUDA-graph replay of n back-to-back launches; interleaved rounds, minimum of each.
    n = 50

    def capture(fn):
        cstream.begin_capture()
        for _ in range(n):
            fn()
        g = cstream.end_capture()
        g.launch(cstream)
        cstream.synchronize()
        return g

    def replay_us(g):
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record(cstream)
        for _ in range(4):
            g.launch(cstream)
        e1.record(cstream)
        e1.synchronize()
        return cp.cuda.get_elapsed_time(e0, e1) / (4 * n) * 1e3

    gb, gl, gs = capture(launch_base), capture(launch_lora), capture(launch_shrink)
    tb = tl = tsh = float("inf")
    for _ in range(6):
        tb = min(tb, replay_us(gb))
        tl = min(tl, replay_us(gl))
        tsh = min(tsh, replay_us(gs))
    # Replays must reproduce the eager result and leave the shrink counters at zero.
    cstream.synchronize()
    if (cp.asnumpy(out_d).view(np.uint16) != first.view(np.uint16)).any():
        ok = False
        print("  GRAPH REPLAY output differs from the eager launch")
    if (cp.asnumpy(ts[0]).view(np.uint16) != t16_h.view(np.uint16)).any() or cp.asnumpy(cnt).any():
        ok = False
        print("  GRAPH REPLAY shrink output differs or counters not reset")
    # A replay reads the current B: negate it (exact in FP16) and expect the new result.
    lb_d *= -1
    gl.launch(cstream)
    cstream.synchronize()
    ref_neg = lora_out(base_h, t16_h[:, :rank], -lb_h)
    if (cp.asnumpy(out_d).view(np.uint16) != ref_neg.view(np.uint16)).any():
        ok = False
        print("  GRAPH REPLAY after a B change does not match the reference")
    lb_d *= -1
    changed = int((out_h.view(np.uint16) != base_h.view(np.uint16)).sum())
    print(f"{'PASS' if ok else 'FAIL'} {kind} M={M} N={N} K={K} r={rank} g={group_size} {cfg} "
          f"changed={changed}/{out_h.size} base={tb:.1f}us lora={tl:.1f}us shrink3={tsh:.1f}us compile={ct:.1f}s")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", choices=["gemv", "gemm"], default="gemv")
    p.add_argument("--cases", default="1,512,2560")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--bM", type=int, default=16)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--split_k", type=int, default=1)
    p.add_argument("--swizzle", type=int, default=1)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--scale_b", type=float, default=0.05)
    a = p.parse_args(_saved_argv[1:])
    cfg = dict(bM=a.bM, stages=a.stages, split_k=a.split_k, swizzle=a.swizzle)
    ok = True
    for i, case in enumerate(a.cases.split(";")):
        M, N, K = (int(v) for v in case.split(","))
        ok &= run_case(a.kernel, M, N, K, a.rank, a.group_size, i, a.iters, a.scale_b, cfg)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
