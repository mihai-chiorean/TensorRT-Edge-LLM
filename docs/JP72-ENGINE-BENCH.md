<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# JetPack 7.2 engine build + benchmark vs the llama.cpp baseline

Third and final document in the JP7.2 series. [`JP72-BUILD.md`](JP72-BUILD.md)
covers the SM87 toolchain; [`JP72-ARTIFACTS.md`](JP72-ARTIFACTS.md) covers
quantization and the ONNX/INT8-sidecar export. This one covers the engine
build, the OpenAI-compatible server, and the measured comparison against the
llama.cpp runtime that `edge-conversation` ships today.

Scripts: [`scripts/jp72/`](../scripts/jp72/).

## 1. Engine build

Built with `llm_build` / `visual_build` against the ONNX directory — **not**
`tensorrt-edgellm-build --model-dir`, which would rebind FP16 embedding tables
and discard the 3.21 GiB INT8 sidecar saving.

Both engines built at the **full requested geometry on the first attempt**. No
fallback to 8192/8192 was needed.

| | `llm_build` | `visual_build` |
|---|---|---|
| Profile | `--maxInputLen 8192 --maxKVCacheCapacity 16384 --maxBatchSize 1` | `--minImageTokens 4 --maxImageTokens 560 --maxImageTokensPerImage 560` |
| Wall time | **360 s** (6 m 00 s) | **384 s** (6 m 24 s) |
| Peak `VmHWM` | **11.86 GB** | 2.65 GB |
| Free RAM at start | 10.99 GB (`edge-llm` stopped) | 11 GB |

TensorRT logged many `Tactic Device request: 30720MB Available: 9981MB ...
Skipping tactic` warnings. These are normal tactic rejection on a 15.6 GB
unified-memory device, not a build failure.

### Artifacts

`~/tensorrt-edgellm-workspace/gemma-4-E4B/engines-int8-16384`, **7,336,823,673 B
(6.83 GiB)** total:

| File | Bytes |
|---|---:|
| `llm/llm.engine` | 3,423,328,260 |
| `llm/ple_embedding.safetensors` (INT8 sidecar) | 2,862,612,824 |
| `llm/embedding.safetensors` (INT8 sidecar) | 672,137,552 |
| `llm/tokenizer.json` | 32,169,878 |
| `visual/visual.engine` | 346,558,652 |
| `llm/config.json`, `tokenizer_config.json`, `processed_chat_template.json`, `visual/config.json`, `visual/preprocessor_config.json` | small |

The two sidecars carry the documented INT8 sizes, confirming the saving
survived into the deployed engine directory.

## 2. Server

`experimental.server` on `127.0.0.1:8003`, loopback only, under transient unit
`trt-llm.service` (`system.slice/trt-llm.service`, readable by the harness's
`--cgroup-unit`). Engine load to `Engine loaded and ready`: **18 s**.

A `systemd-run --scope` will **not** work: scopes reject exec properties, so
`SupplementaryGroups=` cannot be set, the process loses the `video` group, and
`/dev/nvmap` fails with `NvRmMemInitNvmap: Permission denied` → `CUDA
initialization failure with error: 100`. Use a transient *service* unit.

## 3. Smoke test — output is coherent

`/v1/models` reports one model, id `llm`. No `<unused49>` spam, no `<think>`
leakage, in any of the 66 golden records.

- Text: `"A cow makes a moo sound."`
- Vision (the baseline's own camera JPEG): `"The image is heavily overexposed,
  making most of the details unclear. I can vaguely make out a desk area with a
  keyboard and mouse in the foreground."`

### Streaming works

`Content-Type: text/event-stream`. 19 SSE content deltas for a
count-to-ten prompt, arriving **incrementally ~42 ms apart** (`0.083 'One'`,
`0.126 ','`, `0.168 ' two'`, ...), first delta at 83 ms, last at 884 ms.
The mid-generation clause split that `edge-conversation` uses to start TTS
early is supported.

## 4. No prefix cache — the single most important finding

`ContextCacheConfig::enabled` defaults to `false` and `experimental/server/`
never constructs one, so **every request full-prefills**. Confirmed
empirically: the *same* ~5.4k-token prompt sent three times gave TTFT
**5.722 / 5.678 / 5.695 s** — no reuse whatsoever.

llama.cpp, by contrast, serves the ~890-token system prompt from its prefix
cache: in the baseline's `short` scenario it evaluated only **12 of 905**
prompt tokens. That single difference dominates the comparison below.

## 5. Latency — `bench_latency.py`, n = 15, identical config to the baseline

Same `system_prompt.txt`, same `golden_set.json`, same camera JPEG (32,230 B,
md5 `5f14399…`), same `max_tokens` 96 / 192, `temperature 0.0`, `seed 1234`,
`plain: false`. Medians / p95.

| scenario | TTFT llama.cpp | TTFT **TRT** | total llama.cpp | total **TRT** | gen tok/s llama.cpp | gen tok/s **TRT** |
|---|---:|---:|---:|---:|---:|---:|
| `short` | **155 / 178 ms** | 961 / 966 ms | 1,805 / 2,328 ms | 2,176 / 2,612 ms | 18.0 | **23.0** |
| `history` | **350 / 366 ms** | 1,102 / 1,105 ms | 1,732 / 1,990 ms | 2,184 / 2,701 ms | 18.0 | **23.1** |
| `longctx` ~4.3k | 5,360 / 5,587 ms | **4,724 / 4,904 ms** | 6,348 / 6,580 ms | **5,442 / 5,621 ms** | 17.3 | **22.3** |
| `vision` | **1,054 / 1,083 ms** | 1,736 / 1,762 ms | 2,393 / 2,708 ms | 2,298 / 2,908 ms | 17.8 | **22.9** |

Generation is **+28 % across the board** (22.3–23.1 vs 17.3–18.0 tok/s).

Prefill throughput, computed by hand because the server does not report prompt
tokens when streaming (see §7): TRT evaluates the *whole* prompt at roughly
**917–943 tok/s** (`longctx` 4,329 tok / 4.72 s) against llama.cpp's measured
**642 tok/s** over the 3,440 tokens it did not have cached — about **1.45x
faster prefill**. `longctx` is the one scenario where that outweighs the cache,
and TRT wins TTFT by 636 ms *while evaluating 889 more tokens*.

Everywhere else the cache wins: TRT re-prefills the ~890-token system prompt
on every single turn, which costs ~0.95 s and is exactly the `short` gap.

`longctx` needle recovery: **15/15**, matching the baseline.

## 6. Memory

Sampled at 4 Hz on `system.slice/trt-llm.service`.

| | llama.cpp | **TRT Edge-LLM** |
|---|---:|---:|
| cgroup peak (sampled) | 9.76 – 10.49 GB | **11.97 GB** |
| `memory.peak` lifetime | 11.79 GB | 12.09 GB |
| anon / kernel / file | 3.77 / 5.78 / 0.99 GB | 1.52 / **10.21** / 0.23 GB |
| cgroup swap | 0.34 – 0.38 GB | **0.001 GB** |
| system `MemAvailable` min | 1.32 GB | **0.99 GB** |

TRT is ~1.5 GB *above* the baseline's peak and breaches the ≤10.5 GB
acceptance bar, but it touches essentially no swap where llama.cpp held ~350 MB.
Nearly all of TRT's charge is `kernel` (nvmap/CUDA unified), consistent with
weights + a fully preallocated 16,384-token KV pool + a 1.48 GB shared
execution-context arena.

## 7. What could not be measured

The server reports `"prompt_tokens": 0` in the streaming `usage` block
(non-streaming reports it correctly). Consequences:

- The harness's `prompt tok` / `evald tok` / `prompt tok/s` columns are empty
  for TRT. Prefill rates in §5 are hand-computed from the baseline's token
  counts for the identical prompts.
- `n_cache_hit_excluded` is **0 for TRT, but vacuously** — the check keys off
  `prompt_eval_n`, which only llama.cpp's `timings` block supplies. §4's
  repeated-prompt probe is the real evidence, and it is stronger: there is no
  cache to hit.

## 8. Golden set — `run_golden.py`, 22 prompts x 3 repeats

| Metric | llama.cpp | **TRT** |
|---|---|---|
| Deterministic checks | 66/66 | **66/66** |
| Errors | 0 | **0** |
| Hidden-reasoning / `<unused>` leakage | none | **none** |
| Text TTFT (n=54) med / p95 | 152 / 309 ms | 1,002 / 1,106 ms |
| Text total (n=54) med / p95 | 1,313 / 1,772 ms | 1,861 / 2,443 ms |
| Text gen tok/s (n=54) | 18.1 | **23.0** |
| Vision TTFT (n=12) med / p95 | 1,047 / 1,088 ms | 1,751 / 2,522 ms |
| Vision total (n=12) med / p95 | 2,250 / 2,423 ms | 2,551 / 3,397 ms |

Every shipped safety and identity behaviour reproduced: "moo"; Jupiter;
twelve / forty-eight / six; `gracias` / `samedi` / `apa` with the
approximate-pronunciation caveat; refuses to keep a secret from a parent;
routes injury and stove to a trusted adult; "I am Reachy, a robot" with no AI
disclaimer; and with no image says it cannot see *right now*.

Thinking is off: the server's `enable_thinking` body field defaults to `False`,
and 0 of 66 records carried any reasoning characters.

## 9. Verdict

**Quality is a wash (66/66 both). TRT wins throughput; llama.cpp wins the
latency that this product is built around.**

Against the acceptance bar in `edge-conversation`'s
`tests/golden/results/BASELINE-2026-08-29.md`:

| # | Bar | Result |
|---|---|---|
| 1 | `short` TTFT <= 155 ms, `vision` TTFT <= 1,054 ms | **FAIL** (961 ms, 1,736 ms) |
| 2 | generation >= 18.0 tok/s | **PASS** (22.3–23.1) |
| 3 | 66/66 deterministic checks | **PASS** |
| 4 | `longctx` needle 15/15 | **PASS** |
| 5 | peak cgroup <= ~10.5 GB, swap no worse | **FAIL** on peak (11.97 GB); better on swap |
| 6 | `n_cache_hit_excluded` 0, `token_sources` reported | partial — vacuous, see §7 |

Do not swap the runtime on these numbers. `short` TTFT is what a child feels at
the start of every turn, and it regresses 6.2x (155 ms -> 961 ms). The +28 %
generation rate does not buy that back: at 96 max tokens it saves ~0.9 s of
tail while the turn starts ~0.8 s later, and TTS begins at the *first clause*,
so the front of the turn is what matters.

The gap is not a TensorRT kernel deficiency — TRT prefills ~1.45x faster than
llama.cpp. It is the absence of prompt-prefix reuse. The C++ runtime already
ships a `contextCache` with a `ReusePlan`; nothing in the experimental Python
server turns it on. **Wiring `ContextCacheConfig{enabled=true}` through
`experimental/server/engine.py` is the one change that would decide this
comparison**, and it would plausibly flip every row: TRT would keep its +28 %
generation and its faster prefill while paying the system prompt only once.
That is the experiment to run next, and it should be run before any swap
decision.

Secondary follow-ups: the streaming `prompt_tokens: 0` bug (§7), and the
1.5 GB memory overage, which is dominated by a fully preallocated 16,384-token
KV pool that an 8,192 build would shrink.
