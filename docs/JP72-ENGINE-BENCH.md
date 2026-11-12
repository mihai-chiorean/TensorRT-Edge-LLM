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

These measurements are from **2026-08-29**, using the v0.10.0 fork's source
revision `f9368b9b`, and were recorded in `c1170812`. Context reuse was disabled
in that run. The current `scripts/jp72/start_server.sh` passes
`--enable-context-reuse`; the measurements and verdict below describe the
historical run, not the current launcher configuration.

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

## 4. No prefix cache in the benchmarked revision

In `f9368b9b`, `ContextCacheConfig::enabled` defaulted to `false` and
`experimental/server/` never constructed one, so **every request full-prefilled**.
Confirmed empirically: the *same* ~5.4k-token prompt sent three times gave TTFT
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
  repeated-prompt probe is the real evidence, and it is stronger: there was no
  cache enabled in that run.

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

The gap in this run came from the absence of prompt-prefix reuse: TRT
prefilled ~1.45x faster than llama.cpp, but re-evaluated the system prompt on
every turn. The current server exposes `ContextCacheConfig` through
`experimental/server/runtime/engine.py`, and the launcher enables it with
`--enable-context-reuse`. Re-measure with the current revision and launcher
before making a runtime swap decision; these no-cache results do not predict
its latency.

Secondary follow-ups: the streaming `prompt_tokens: 0` bug (§7), and the
1.5 GB memory overage, which is dominated by a fully preallocated 16,384-token
KV pool that an 8,192 build would shrink.

## 10. LoRA engine, rank 16 (2026-11-12)

Jetson AGX Orin 32 GB, MAXN, JetPack 7.2 / CUDA 13.2 / TensorRT 10.16.2, fork
`main` `17ec7c6` built with `BUILD_PYTHON_BINDINGS=ON`. Export: the 0.11.1
vanilla Gemma 4 E4B QAT Q4_0 import (group-32 INT4, INT8 embedding/PLE
sidecars, `rebase-0111/onnx/e4b-van`), the deployed weights without MTP.
Both engines: `scripts/jp72/build_engines.sh`, 8192 input / 16384 KV / batch
1, `HW_COMPAT=same_compute_capability`; one with `LORA_RANK=16` from
`lora_model.onnx`, one with the default rank 0 from `model.onnx`. The service
was not running on the AGX.

Adapter: the PEFT tool-use adapter `adapter-qat-r16` (r 16, alpha 32, 258
attention+MLP modules) processed per `JP72-ARTIFACTS.md` Stage 3b; the 85 PLE
bindings stay at rank-1 zeros.

| | rank 0 | rank 16 | delta |
|---|---:|---:|---:|
| `llm_build` wall time | 82 s | 138 s | +56 s |
| `llm_build` peak `VmHWM` | 10,875,536 kB | 12,063,812 kB | +1.13 GiB |
| `llm/llm.engine` | 2,648,771,260 B | 2,676,791,836 B | +28,020,576 B |
| Context-profile activation (build log, = runtime context arena) | 1,477,777,408 B | 1,936,242,688 B | +458,465,280 B (+437 MiB) |
| Context-profile max scratch (build log) | 353,697,792 B | 573,087,744 B | +219,389,952 B |
| Decode-profile activation (build log) | 67,748,352 B | 67,779,072 B | +30,720 B |
| Decode, base request | 45.0–45.2 tok/s | 27.1–27.2 tok/s | -40 % |
| Decode, adapter request | — | 27.2–27.4 tok/s | — |

Arena: the runtime logs `Setup shared execution context memory` with the same
byte counts as the build log's context-profile `Total Activation Memory`. The
decode profile is not the arena; the vision encoder needs 176,440,320 B and
shares it. A first build without `HW_COMPAT` took 142 s at 11,901,936 kB (rank
16) and 89 s at 10,948,248 kB (rank 0), with the same activation sizes.

Decode: greedy, 256 tokens after a 28-token prompt, CUDA graphs captured,
context reuse on. Two runs, each with three repeats; tok/s = 256 /
(t(257 tokens) - t(1 token)), best of three; the prompt prefix is cached in
both terms, so the difference is decode time. A rank-16 base request costs the
same as an adapter request: the engine runs all 343 LoRA branches every step,
bound either to the adapter or to rank-1 zeros. Inserting only the adapter's
258 attention+MLP GEMMs would skip the 85 PLE branches; that selection is not
implemented in `insert-lora`.

Behaviour, on one runtime with the adapter loaded (top-5 first-token
logprobs, greedy 64 tokens, two tools declared):

| Prompt | Tools | First-token logprob, base | adapter | Greedy output |
|---|---|---:|---:|---|
| "Turn the volume down to 20 please." | yes | -0.108 (runner-up -2.28) | -0.0047 (runner-up -6.50) | `call:set_volume{level:20}` both |
| "Can you look to your left?" | yes | -0.087 (runner-up -2.48) | -0.0047 (runner-up -5.98) | `call:look_at{direction:left}` both |
| "Tell me a short joke about robots." | yes | -0.071 | -0.070 | punchline differs |
| "What is the capital of Romania?" | no | -0.020 | -0.084 | same sentence |

The adapter makes the tool-call opening near-certain where a tool applies;
the base model's runner-up there is a different opening token.

Base, adapter, base, adapter in that order gave the same tokens each time.
With CUDA graphs captured and without them, tokens and logprobs were
identical. With context reuse off, repeats were bit-identical; with it on, the
second identical request reused its own adapter's prefix and moved first-token
logprobs by up to 0.01 without changing tokens. The first adapter request after
an identical base request matched the reuse-off adapter result exactly, so it
did not reuse the base prefix (the cache key includes the adapter name). The runtime captures one decode graph per
adapter plus one for base at start.

### Through the server (2026-11-12)

The same rank-16 engine, with the processed adapter placed at
`llm/lora_weights/tool/`, served by `experimental.server` from fork `main`
`cf8d968` (`--enable-context-reuse --enable-auto-tool-choice
--served-model-name gemma`). The adapter is discovered without flags.
`/v1/models` lists `gemma` and `gemma:tool`, and `/health` reports
`max_lora_rank 16` and, for `tool`, rank 16, 258 bound modules and 85
unbound bindings.

Six tool prompts (three tools declared, temperature 0) were sent as base,
`model: "gemma:tool"`, `model: "gemma"` + `lora_name: "tool"`, then base
again. Both selector spellings gave identical output and echoed `gemma:tool`,
and the second base run repeated the first.

| Prompt | base | adapter |
|---|---|---|
| "Turn the volume down to 20 please." | `set_volume{level: 20}` | same |
| "Can you look to your left?" | `look_at{direction: left}` | same |
| "What time is it?" | `get_time{}` | same |
| "Make it louder, like 80." | `set_volume{level: 80}` | same |
| "Look up and then tell me what you see." | `look_at{direction: center}` | `look_at{direction: up}` |
| "Tell me a short joke about robots." | "...it had a virus!" | "...a little \"buggy\"!" |

An unknown alias, conflicting selectors and an unknown `lora_name` each
returned 400. An adapter given to the rank-0 engine stopped the server at
startup. Decode through the server: 27.3 tok/s base, 27.4 tok/s adapter.

At startup the runtime warns once that `tool` leaves 170 of 686 LoRA
bindings (the 85 PLE pairs) on the rank-1 zero dummy. An adapter above the
engine's rank (the same adapter zero-padded to rank 32) now fails runtime
construction. Before that check, it loaded, decode-graph capture failed with
only a warning, and the first adapter request failed in `setInputShape`.

Verdict: rank 16 is usable for an adapter-serving bundle, but costs about 40 %
of decode throughput on every request and 437 MiB more arena. Production
engines stay at rank 0, which is the script default.
