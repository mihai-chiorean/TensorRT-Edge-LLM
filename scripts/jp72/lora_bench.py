#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""LoRA/MTP engine bench: load memory, decode tok/s, verify acceptance, TTFT, greedy outputs.

One ``LLMRuntime`` is constructed per run and every variant (base + each
adapter) is measured on it. Decode is measured as the runtime serves it:
decoding CUDA graphs are captured once after construction (the run fails if
capture fails), each variant gets one untimed warm-up decode, and the reported
``decode_tok_s`` is the median over ``--decode-reps`` timed runs of the same
prompt, counted from the decode phase only — wall time minus the prefill wall
time that the runtime's prefill metrics record for that request (prefill
counters only advance while profiling is on, so the pass that reads them runs
with profiling enabled and restores the flag afterwards).

``SpecDecodeGenerationMetrics`` (cpp/profiling/metrics.h) only accumulates
while the thread-local profiling flag is on and its counters are cumulative
for the lifetime of the runtime, so each variant's metrics pass reads the
counters before and after its profiled run and reports the deltas.

Usage
-----
    # base engine only
    python3 scripts/jp72/lora_bench.py --engine <engine_dir>

    # MTP engine + two adapters, measured on the Orin NX
    python3 scripts/jp72/lora_bench.py \
        --engine /data/engines/gemma4-mtp --mtp \
        --adapter lora_a=/data/adapters/lora_a \
        --adapter lora_b=/data/adapters/lora_b \
        --build-dir "$B" \
        --out /data/bench/$(date +%s).jsonl

One ``LLMRuntime`` is constructed for the whole run: every adapter loads once
into the runtime's ``lora_weights_map`` and each request selects its adapter
via ``lora_weights_name`` (empty for base), so base and adapter share one
context cache and one set of captured decode graphs. The multimodal engine
dir is ``<engine>/visual`` when that directory exists. The MTP drafting shape
defaults to the benchmarked top-2 / 5-step / verify-12 tree
(``--draft-top-k``, ``--draft-step``, ``--verify-tree-size`` to override);
``--no-reuse`` builds the runtime with the context cache disabled.

Env: ``B`` (or ``--build-dir``) must point at the EdgeLLM build tree; the
plugin lib and ``pybind`` module are loaded from there. Jetson-only: every run
touches CUDA via the pybind runtime.

Record schema (one JSON line per record in ``--out``; see ``--out`` docs):
    tag                     "base" or the adapter name given via --adapter
    mtp                     engine runs speculative decoding
    engine                  engine directory
    adapter                 adapter directory (null for base)
    peak_host_rss_mb        peak RSS of the process after the runtime loads
                            (one runtime for the whole run, so it is monotone
                            across records)
    prompt_tokens           measured prompt token count (see --ttft-lengths)
    ttft_ms_tool            TTFT of the original tool-prompt request
    ttft_sweep              list of {"target_tokens", "prompt_tokens",
                            "ttft_ms", "cached_tokens"}; cached_tokens is the
                            context-cache tokens reused by that row (-1 when
                            the binding does not expose it)
    decode_*                median of --decode-reps timed decode runs of
                            DECODE_PROMPTS[0] (one untimed warm-up first):
                            decode_tok_s counts decode-phase wall time only
                            (wall minus prefill); decode_seconds is the
                            median run's wall time, decode_prefill_seconds
                            the measured prefill wall time subtracted from
                            it, and decode_generated/prompt_tokens are that
                            run's token counts
    tokens_per_verify_step  verify-step metrics from the profiled pass:
                            generated / iterations, both taken as the delta
                            of the runtime's cumulative counters across the
                            pass, so each variant reports only its own run
    verify_steps            speculative verification iterations in that pass
                            (delta of total_iterations)
    accepted_draft_tokens   delta of total_accepted_draft_tokens (when exposed)
    proposed_draft_tokens   delta of total_proposed_draft_tokens (when exposed)
    matches_base            fraction of the 9 greedy prompts whose output
                            token ids equal the base variant's output; this
                            measures how much an adapter changes the output,
                            not correctness

For the Orin NX measurement harness see scripts/jp72/lora_nx_measure.sh.
"""
import argparse
import ctypes
import json
import os
import statistics
import sys
import time

TOOL_DEFINITIONS = [
    {
        "name": "set_light",
        "description": "Turn a light on or off",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {
                    "type": "string",
                    "description": "Room name"
                },
                "state": {
                    "type": "string",
                    "enum": ["on", "off"]
                }
            },
            "required": ["room", "state"]
        },
    },
]

TOOL_PROMPT = "Turn the kitchen light off."

DECODE_PROMPTS = [
    "What sound does a cow make? Answer in one short sentence.",
    "Name three primary colors.",
    "What is 17 * 24? Give the number only.",
    "Give a one-sentence summary of the plot of Romeo and Juliet.",
    "Translate 'good morning' into French.",
    "Who wrote Pride and Prejudice?",
    "What is the chemical symbol for gold?",
    "Count from 1 to 5.",
    "What year did the Berlin Wall fall?",
]

GREEDY_PROMPTS = [
    "What is the capital of France?",
    "Explain why the sky is blue in one sentence.",
    "List the first five prime numbers.",
    "What does DNA stand for?",
    "Name the largest ocean on Earth.",
    "What is 15 percent of 200?",
    "Who directed the movie Jaws?",
    "Give a synonym for 'happy'.",
    "What temperature does water boil at in Celsius?",
]

# Fixed paragraph repeated by the TTFT prompt builder; a unique time_ns prefix
# per candidate defeats engine context reuse, as in the original probe.
TTFT_PARAGRAPH = (
    "The Orin NX boots JetPack 7.2 with a 1024-core Ampere GPU and a power "
    "budget capped at twenty-five watts, which shapes every kernel choice the "
    "runtime makes when it schedules prefill work against the L2 carveout. "
    "Builders on the bench log each engine's identity hash before measuring "
    "anything, because a stale adapter directory silently reuses the wrong "
    "context cache and skews the time-to-first-token numbers. ")

DEFAULT_TTFT_LENGTHS = "20,100,400,800,1600"
DEFAULT_REPS = 5
TTFT_MAX_TOKENS = 1
DECODE_MAX_TOKENS = 256
# Timed decode runs per variant after one untimed warm-up; the record reports
# the median over these.
DEFAULT_DECODE_REPS = 5
MIN_DECODE_REPS = 3
GREEDY_MAX_TOKENS = 64

# MTP drafting shape the engines are built and benchmarked with (draft top-2,
# 5 draft steps, verify tree 12; see the production MTP runtime config).
DEFAULT_DRAFT_TOP_K = 2
DEFAULT_DRAFT_STEP = 5
DEFAULT_VERIFY_TREE_SIZE = 12

RECORD_FIELDS = [
    "tag",
    "mtp",
    "engine",
    "adapter",
    "peak_host_rss_mb",
    "cuda_graphs",
    "prompt_tokens",
    "ttft_ms_tool",
    "ttft_sweep",
    "decode_prompt_tokens",
    "decode_generated",
    "decode_seconds",
    "decode_prefill_seconds",
    "decode_tok_s",
    "tokens_per_verify_step",
    "verify_steps",
    "accepted_draft_tokens",
    "proposed_draft_tokens",
    "matches_base",
]


def parse_args(argv=None):
    """Parse the CLI; importable without the runtime for --help/tests."""
    parser = argparse.ArgumentParser(
        prog="lora_bench.py",
        description=
        "LoRA/MTP engine bench (Jetson; touches CUDA via the pybind runtime).")
    parser.add_argument(
        "--engine",
        required=True,
        help="Engine bundle directory containing llm/ (base+draft for --mtp).")
    parser.add_argument(
        "--mtp",
        action="store_true",
        help=
        "Engine is a speculative-decoding (MTP) engine; collect verify-step metrics."
    )
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=
        "LoRA adapter as NAME=PATH; repeatable. Base is always measured too.")
    parser.add_argument(
        "--build-dir",
        default=os.environ.get("B", ""),
        metavar="DIR",
        help=
        "EdgeLLM build tree with the plugin lib and pybind module (default: $B)."
    )
    parser.add_argument(
        "--ttft-lengths",
        default=DEFAULT_TTFT_LENGTHS,
        help="Comma-separated target prompt token counts for the TTFT sweep "
        f"(default: {DEFAULT_TTFT_LENGTHS}).")
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        help="Repetitions per TTFT sweep point (default: %(default)s).")
    parser.add_argument(
        "--decode-reps",
        type=int,
        default=DEFAULT_DECODE_REPS,
        help="Timed decode runs per variant after one untimed warm-up; the "
        f"record reports the median (default: %(default)s, minimum "
        f"{MIN_DECODE_REPS}).")
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Construct the runtime with the context cache disabled; unique "
        "time_ns prefixes in the TTFT prompts defeat reuse regardless.")
    parser.add_argument(
        "--draft-top-k",
        type=int,
        default=DEFAULT_DRAFT_TOP_K,
        help="MTP draft top-k (--mtp only; default: %(default)s).")
    parser.add_argument(
        "--draft-step",
        type=int,
        default=DEFAULT_DRAFT_STEP,
        help="MTP draft step (--mtp only; default: %(default)s).")
    parser.add_argument(
        "--verify-tree-size",
        type=int,
        default=DEFAULT_VERIFY_TREE_SIZE,
        help="MTP verify tree size (--mtp only; default: %(default)s).")
    parser.add_argument(
        "--device-embedding-tables",
        action="store_true",
        help=
        "MTP only: keep the embedding tables on device instead of host-mapped "
        "(the default mirrors the original probe, which passed True for "
        "host_embedding_tables and matches --host-embedding-tables in the "
        "server).")
    parser.add_argument(
        "--out",
        metavar="FILE.jsonl",
        help="Append one JSON line per record to this file instead of stdout.")
    parser.add_argument(
        "--skip-outputs",
        action="store_true",
        help=
        "Do not include generated texts in records (they are still generated)."
    )
    args = parser.parse_args(argv)
    if not args.build_dir:
        parser.error("--build-dir is required (or set $B)")
    if args.reps < 1:
        parser.error("--reps must be >= 1")
    if args.decode_reps < MIN_DECODE_REPS:
        parser.error(
            f"--decode-reps must be >= {MIN_DECODE_REPS} so the median is "
            "not a single un-warmed run")
    for pair in args.adapter:
        if "=" not in pair:
            parser.error(f"--adapter must be NAME=PATH, got {pair!r}")
    return args


def parse_adapters(pairs):
    """Parse NAME=PATH pairs; raises ValueError on a malformed pair."""
    adapters = []
    for pair in pairs:
        name, sep, path = pair.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--adapter must be NAME=PATH, got {pair!r}")
        adapters.append((name, path))
    return adapters


def parse_ttft_lengths(spec):
    """Parse a comma-separated list of target prompt token counts."""
    lengths = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise ValueError(
                f"--ttft-lengths must be comma-separated integers, got {spec!r}"
            ) from None
        if value <= 0:
            raise ValueError(
                f"--ttft-lengths entries must be positive, got {value}")
        lengths.append(value)
    if not lengths:
        raise ValueError("--ttft-lengths is empty")
    return lengths


def median(values):
    """Median of a non-empty sequence."""
    return statistics.median(values)


def count_tokens(tokenizer, text):
    """Token count of text via a tokenizers.Tokenizer; -1 when unavailable."""
    if tokenizer is None:
        return -1
    return len(tokenizer.encode(text).ids)


def build_ttft_prompt(target_tokens, tokenizer=None, paragraph=TTFT_PARAGRAPH):
    """Build an uncached prompt of approximately ``target_tokens`` tokens.

    The first line carries a unique time_ns stamp so the engine's context
    cache never reuses a previous candidate's prefix; the rest repeats a fixed
    paragraph until the tokenizer counts ``target_tokens`` tokens (within 5%).
    Without a tokenizer the paragraph is scaled by a 4-chars/token heuristic.
    Returns (prompt_text, measured_tokens); measured is -1 without a tokenizer.
    """
    stamp = time.time_ns()
    unique = f"[probe {stamp}]\n"
    if tokenizer is None:
        body = paragraph * max(
            1, int(round(target_tokens * 4.0 / len(paragraph))))
        candidate = unique + body
        return candidate, -1

    # Targets at or below one paragraph: binary-search the character cut point
    # (the token count is monotone in text length for whitespace/BPE token-
    # izers). Larger targets: start from whole repetitions and refine the
    # character cut point of the final partial paragraph.
    para_len = len(paragraph)
    para_tokens = max(1, len(tokenizer.encode(paragraph).ids))
    whole_reps = max(target_tokens - 10, 1) // para_tokens
    if whole_reps == 0:
        lo, hi = 1, para_len
        best_small = unique + paragraph
        best_small_err = None
        for _ in range(20):
            mid = (lo + hi) // 2
            candidate = unique + paragraph[:mid]
            measured = count_tokens(tokenizer, candidate)
            if measured <= 0:
                break
            err = abs(measured - target_tokens) / target_tokens
            if best_small_err is None or err < best_small_err:
                best_small, best_small_err = candidate, err
            if err <= 0.05:
                return candidate, measured
            if measured > target_tokens:
                hi = mid - 1
            else:
                lo = mid + 1
        return best_small, count_tokens(tokenizer, best_small)

    best = None
    best_err = None
    base = unique + paragraph * whole_reps
    base_n = count_tokens(tokenizer, base)
    if base_n > 0:
        err = abs(base_n - target_tokens) / target_tokens
        if err <= 0.05:
            return base, base_n
        best, best_err = base, err
    # Refine the partial paragraph by characters: binary-search the cut point
    # so the last (partial) paragraph lands closest to the target.
    lo, hi = 1, para_len
    for _ in range(20):
        mid = (lo + hi) // 2
        candidate = base + paragraph[:mid]
        measured = count_tokens(tokenizer, candidate)
        if measured <= 0:
            continue
        err = abs(measured - target_tokens) / target_tokens
        if best_err is None or err < best_err:
            best, best_err = candidate, err
        if err <= 0.05:
            return candidate, measured
        if measured > target_tokens:
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        return base, base_n
    return best, count_tokens(tokenizer, best)


def make_tool_request(rt, prompt, lora=""):
    """Build a one-shot tool-call request via the pybind helpers."""
    tools = []
    for tool in TOOL_DEFINITIONS:
        item = rt.ToolDefinition()
        item.name = tool["name"]
        item.description = tool["description"]
        item.parameters = json.dumps(tool["parameters"])
        item.strict = True
        item.has_description = True
        item.has_parameters = True
        item.has_strict = True
        tools.append(item)
    req = rt.create_generation_request(
        [[rt.create_text_message("user", prompt)]],
        temperature=0.0,
        top_k=1,
        max_generate_length=TTFT_MAX_TOKENS,
        apply_chat_template=True,
        add_generation_prompt=True,
        enable_thinking=False,
        lora_weights_name=lora,
        save_system_prompt_kv_cache=False,
        disable_spec_decode=False)
    req.tools = tools
    req.tool_choice = rt.ToolChoice()
    req.tool_choice.mode = rt.ToolChoiceMode.AUTO
    return req


def make_decode_request(rt, prompt, max_tokens=DECODE_MAX_TOKENS, lora=""):
    """Build a plain decode request."""
    return rt.create_generation_request(
        [[rt.create_text_message("user", prompt)]],
        temperature=0.0,
        top_k=1,
        max_generate_length=max_tokens,
        apply_chat_template=True,
        add_generation_prompt=True,
        enable_thinking=False,
        lora_weights_name=lora,
        save_system_prompt_kv_cache=False,
        disable_spec_decode=False)


def run_decode(rt, runtime, prompt, max_tokens=DECODE_MAX_TOKENS, lora=""):
    """Run one decode request; returns (prompt_tokens, generated, seconds, output_ids)."""
    req = make_decode_request(rt, prompt, max_tokens, lora=lora)
    t0 = time.perf_counter()
    resp = runtime.handle_request(req)
    seconds = time.perf_counter() - t0
    prompt_tokens = int(
        resp.prompt_token_counts[0]) if resp.prompt_token_counts else 0
    out_ids = list(resp.output_ids[0]) if resp.output_ids else []
    return prompt_tokens, len(out_ids), seconds, out_ids


def read_prefill_wall_seconds(runtime):
    """Delta of the runtime's cumulative prefill wall time, in seconds.

    The pybind module does not expose the runtime's stage timer, so prefill
    wall time is derived from ``LLMPrefillMetrics``: with profiling enabled
    the runtime records ``prefill_seconds`` per request (and the counters are
    cumulative), so the caller snapshots before and after a run and
    subtracts. None when the binding does not expose the metrics.
    """
    getter = getattr(runtime, "get_prefill_metrics", None)
    if getter is None:
        return None
    try:
        metrics = getter()
    except (AttributeError, RuntimeError):
        return None
    seconds = getattr(metrics, "prefill_seconds", None)
    if seconds is None:
        return None
    return float(seconds)


def read_context_cache_metrics(runtime):
    """Snapshot the context-cache token counters; None when unavailable."""
    getter = getattr(runtime, "get_context_cache_metrics", None)
    if getter is None:
        return None
    try:
        metrics = getter()
    except (AttributeError, RuntimeError):
        return None
    if metrics is None:
        return None
    return {
        "matched_tokens": int(getattr(metrics, "matched_tokens", 0)),
        "reused_tokens": int(getattr(metrics, "reused_tokens", 0)),
    }


def verify_step_math(total_generated, total_iterations):
    """tokens_per_verify_step = generated / iterations; None if iterations <= 0."""
    if total_iterations is None or total_iterations <= 0:
        return None
    return total_generated / total_iterations


def spec_metrics_delta(before, after):
    """Per-run spec-decode counters: after - before for all four fields.

    ``SpecDecodeGenerationMetrics::recordRun`` (cpp/profiling/metrics.h) adds
    to totals for the lifetime of the runtime, so a variant's own numbers are
    the deltas of two snapshots around its profiled run.
    """
    if after is None:
        return None
    if before is None:
        before = {key: 0 for key in after}
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in ("total_iterations", "total_generated_tokens",
                    "accepted_draft_tokens", "proposed_draft_tokens")
    }


def assemble_record(tag,
                    mtp,
                    engine_dir,
                    adapter_path,
                    peak_rss_mb,
                    ttft_tool_ms,
                    ttft_sweep,
                    decode,
                    spec_metrics,
                    matches_base=None):
    """Assemble one output record dict; pure (no runtime).

    ``decode`` carries the median run's seconds, the mean prefill wall time
    across the timed runs and the per-run prompt/generated token counts;
    ``decode_tok_s`` divides the median run's generated tokens by its
    decode-phase time (wall time minus prefill).
    """
    record = {
        "tag":
        tag,
        "mtp":
        mtp,
        "engine":
        engine_dir,
        "adapter":
        adapter_path,
        "peak_host_rss_mb":
        peak_rss_mb,
        "cuda_graphs":
        decode["cuda_graphs"],
        "prompt_tokens":
        decode["prompt_tokens"],
        "ttft_ms_tool":
        ttft_tool_ms,
        "ttft_sweep":
        ttft_sweep,
        "decode_prompt_tokens":
        decode["prompt_tokens"],
        "decode_generated":
        decode["generated"],
        "decode_seconds":
        round(decode["seconds"], 4),
        "decode_prefill_seconds":
        round(decode["prefill_seconds"], 4)
        if decode["prefill_seconds"] is not None else None,
        "decode_tok_s":
        decode["tok_s"],
    }
    if spec_metrics is None:
        record["tokens_per_verify_step"] = None
        record["verify_steps"] = None
    else:
        tps = verify_step_math(spec_metrics["total_generated_tokens"],
                               spec_metrics["total_iterations"])
        record["tokens_per_verify_step"] = tps
        record["verify_steps"] = spec_metrics["total_iterations"]
        record["accepted_draft_tokens"] = spec_metrics["accepted_draft_tokens"]
        record["proposed_draft_tokens"] = spec_metrics["proposed_draft_tokens"]
    record["matches_base"] = matches_base
    return record


def run_ttft_sweep(rt, runtime, lengths, reps, tokenizer, lora=""):
    """For each target length build an uncached prompt, generate 1 token, take the median.

    Each rep gets its own time_ns-prefixed prompt, so engine context reuse
    never hits; ``prompt_tokens`` is the engine's measured count from the
    engine tokenizer. ``cached_tokens`` is the delta of the runtime's
    context-cache reused_tokens across the point's requests (-1 when the
    binding does not expose it), so a row served from the cache is visible
    instead of silently trusted as uncached.
    """
    sweep = []
    for target in lengths:
        ttfts = []
        prompt_tokens = None
        cc_before = read_context_cache_metrics(runtime)
        for _ in range(reps):
            prompt, measured = build_ttft_prompt(target, tokenizer)
            req = make_tool_request(rt, prompt, lora=lora)
            t0 = time.perf_counter()
            resp = runtime.handle_request(req)
            ttfts.append((time.perf_counter() - t0) * 1000.0)
            if prompt_tokens is None:
                prompt_tokens = int(resp.prompt_token_counts[0]
                                    ) if resp.prompt_token_counts else measured
        cc_after = read_context_cache_metrics(runtime)
        if cc_before is None or cc_after is None:
            cached_tokens = -1
        else:
            cached_tokens = cc_after["reused_tokens"] - cc_before[
                "reused_tokens"]
        sweep.append({
            "target_tokens": target,
            "prompt_tokens": prompt_tokens,
            "ttft_ms": round(median(ttfts), 3),
            "cached_tokens": cached_tokens,
        })
    return sweep


def read_spec_metrics(runtime):
    """Snapshot the spec-decode metrics counters; None when unavailable (plain engine)."""
    getter = getattr(runtime, "get_spec_decode_generation_metrics", None)
    if getter is None:
        return None
    try:
        metrics = getter()
    except (AttributeError, RuntimeError):
        return None
    return {
        "total_iterations":
        int(getattr(metrics, "total_iterations", 0)),
        "total_generated_tokens":
        int(getattr(metrics, "total_generated_tokens", 0)),
        "accepted_draft_tokens":
        int(getattr(metrics, "total_accepted_draft_tokens", 0)),
        "proposed_draft_tokens":
        int(getattr(metrics, "total_proposed_draft_tokens", 0)),
    }


def load_runtime(rt, args, lora_map):
    """Construct LLMRuntime for a plain or MTP engine.

    Mirrors the original probe and the production runtime wiring: the LLM
    engine dir is ``<engine>/llm`` (the runtime loads ``config.json`` /
    ``base_config.json`` relative to it, not the bundle root), the
    multimodal dir is ``<engine>/visual`` when present (empty string
    otherwise, the pybind default for text-only engines), the context cache
    config is always a real ``rt.ContextCacheConfig`` (``None`` fails the
    pybind constructor), MTP defaults to host-mapped embedding tables like
    the original probe (``--device-embedding-tables`` opts out), and the
    MTP drafting shape defaults to the benchmarked top-2 / 5-step /
    verify-12 tree, overridable by flags.
    """
    engine_dir = os.path.join(args.engine, "llm")
    visual_dir = os.path.join(args.engine, "visual")
    if not os.path.isdir(visual_dir):
        visual_dir = ""
    cc = rt.ContextCacheConfig()
    cc.enabled = not args.no_reuse
    host_embedding_tables = args.mtp and not args.device_embedding_tables
    if args.mtp:
        return rt.LLMRuntime(engine_dir, visual_dir, lora_map,
                             args.draft_top_k, args.draft_step,
                             args.verify_tree_size, "", "", cc, 0,
                             host_embedding_tables, False)
    return rt.LLMRuntime(engine_dir, visual_dir, lora_map, "", cc, False,
                         False)


def run_greedy_check(rt, runtime, base_outputs=None, lora=""):
    """Run the 9 greedy prompts; returns (token_id lists, fraction matching base).

    The fraction compares each adapter's output token ids with the base
    variant's: it measures how much an adapter changes the output, not
    correctness (it is reported as ``matches_base`` in the record).
    """
    outputs = []
    for prompt in GREEDY_PROMPTS:
        _, _, _, out_ids = run_decode(rt,
                                      runtime,
                                      prompt,
                                      GREEDY_MAX_TOKENS,
                                      lora=lora)
        outputs.append(out_ids)
    match = None
    if base_outputs is not None:
        if len(base_outputs) != len(outputs):
            match = 0.0
        else:
            match = sum(1 for a, b in zip(base_outputs, outputs)
                        if a == b) / len(outputs)
    return outputs, match


def load_rt_module(build_dir):
    """Load the pybind runtime module: plugin lib via ctypes, then the module.

    The plugin load is skipped when LORA_BENCH_SKIP_PLUGIN_LOAD=1 so tests can
    run main() with a fake module; the real bench always loads the plugin.
    """
    if os.environ.get("LORA_BENCH_SKIP_PLUGIN_LOAD") != "1":
        ctypes.CDLL(os.path.join(build_dir, "libNvInfer_edgellm_plugin.so"),
                    mode=ctypes.RTLD_GLOBAL)
    if build_dir not in sys.path:
        sys.path.insert(0, os.path.join(build_dir, "pybind"))
    import _edgellm_runtime as rt  # noqa: E402  (CUDA/TensorRT - never imported by tests)

    return rt


def measure_variant(rt,
                    args,
                    runtime,
                    lengths,
                    tokenizer,
                    tag,
                    adapter_path,
                    base_outputs=None,
                    cuda_graphs=True):
    """Measure one variant (base or one adapter) on the shared runtime.

    Every request carries ``lora_weights_name`` — empty for base, the adapter
    name otherwise — so the runtime switches adapters per request, exactly as
    the original probe did. ``cuda_graphs`` is the capture result recorded in
    every record of this run.
    """
    lora = tag if adapter_path else ""

    peak_rss_mb = None
    try:
        import resource  # noqa: E402

        peak_rss_mb = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        pass

    # --- tool-prompt TTFT (original probe behavior) --------------------
    req = make_tool_request(rt, TOOL_PROMPT, lora=lora)
    t0 = time.perf_counter()
    runtime.handle_request(req)
    ttft_tool_ms = (time.perf_counter() - t0) * 1000.0

    # --- decode: one untimed warm-up, then timed runs -------------------
    # The warm-up absorbs first-run costs (adapter switch, allocator growth)
    # so the timed runs measure steady-state decode.
    decode_prompt = DECODE_PROMPTS[0]
    run_decode(rt, runtime, decode_prompt, lora=lora)
    runs = []
    for _ in range(args.decode_reps):
        runs.append(run_decode(rt, runtime, decode_prompt, lora=lora))
    # Median run by wall time; its token counts back the record fields.
    runs.sort(key=lambda run: run[2])
    prompt_tokens, generated, seconds, _ = runs[len(runs) // 2]

    # --- decode-phase tok/s: subtract the prefill wall time -------------
    # The binding exposes no stage timer, so prefill wall time comes from the
    # runtime's cumulative prefill counters, which only advance while
    # profiling is on: run the same prompt once with profiling enabled, read
    # LLMPrefillMetrics around it, and take the delta as that request's
    # prefill wall time.
    rt.set_profiling_enabled(True)
    try:
        prefill_before = read_prefill_wall_seconds(runtime)
        run_decode(rt, runtime, decode_prompt, lora=lora)
        prefill_after = read_prefill_wall_seconds(runtime)
    finally:
        rt.set_profiling_enabled(False)
    if prefill_before is None or prefill_after is None:
        prefill_seconds = None
        decode_seconds = None
        decode_tok_s = None
    else:
        prefill_seconds = max(0.0, prefill_after - prefill_before)
        decode_seconds = max(0.0, seconds - prefill_seconds)
        decode_tok_s = (round(generated / decode_seconds, 2)
                        if decode_seconds > 0 else None)

    # --- TTFT sweep -----------------------------------------------------
    sweep = run_ttft_sweep(rt,
                           runtime,
                           lengths,
                           args.reps,
                           tokenizer,
                           lora=lora)

    # --- metrics pass: profiling ON only here ---------------------------
    # recordRun() (cpp/profiling/metrics.h) is a no-op unless the
    # thread-local profiling flag is set, and the flag must be off for
    # the timed runs above; the counters therefore come from this extra
    # run, which is why it exists.
    spec_metrics = None
    if args.mtp:
        rt.set_profiling_enabled(True)
        try:
            # Counters accumulate for the lifetime of the runtime, so
            # snapshot before the profiled run and report the delta.
            spec_before = read_spec_metrics(runtime)
            run_decode(rt, runtime, decode_prompt, lora=lora)
            spec_metrics = spec_metrics_delta(spec_before,
                                              read_spec_metrics(runtime))
        finally:
            rt.set_profiling_enabled(False)
        if spec_metrics is None or spec_metrics["total_iterations"] <= 0:
            print(
                "ERROR: MTP engine reported zero spec-decode iterations "
                "(profiling was enabled for the metrics pass); "
                "cannot compute tokens_per_verify_step",
                file=sys.stderr)
            return None

    # --- greedy check ---------------------------------------------------
    outputs, match = run_greedy_check(rt,
                                      runtime,
                                      base_outputs=base_outputs,
                                      lora=lora)

    decode = {
        "prompt_tokens": prompt_tokens,
        "generated": generated,
        "seconds": seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "tok_s": decode_tok_s,
        "cuda_graphs": cuda_graphs,
    }
    record = assemble_record(tag=tag,
                             mtp=args.mtp,
                             engine_dir=args.engine,
                             adapter_path=adapter_path,
                             peak_rss_mb=peak_rss_mb,
                             ttft_tool_ms=round(ttft_tool_ms, 3),
                             ttft_sweep=sweep,
                             decode=decode,
                             spec_metrics=spec_metrics,
                             matches_base=match)
    # --skip-outputs keeps generated texts out of the record; only the
    # per-prompt token counts are retained for the greedy comparison.
    record["outputs"] = None if args.skip_outputs else [
        len(o) for o in outputs
    ]

    print(
        f"[{tag}] ttft_tool={record['ttft_ms_tool']:.1f}ms "
        f"decode={record['decode_tok_s']}tok/s "
        f"tpv={record['tokens_per_verify_step']}",
        flush=True)
    return record, outputs


def capture_decode_cuda_graphs(runtime):
    """Capture the decoding CUDA graphs the serving configuration uses.

    The server captures at startup (experimental/server/runtime/engine.py);
    without capture the decode numbers would describe eager execution instead
    of what the runtime actually serves with. Returns True when capture
    succeeded or the method is unavailable on the loaded binding.
    """
    capture = getattr(runtime, "capture_decoding_cuda_graph", None)
    if capture is None:
        return True
    try:
        return bool(capture())
    except (AttributeError, RuntimeError):
        return False


def main(argv=None):
    """Entry point; imports the pybind runtime (and CUDA) only here."""
    args = parse_args(argv)
    adapters = parse_adapters(args.adapter)
    lengths = parse_ttft_lengths(args.ttft_lengths)

    build_dir = os.path.abspath(args.build_dir)
    rt = load_rt_module(build_dir)

    engine_tokenizer = os.path.join(args.engine, "llm", "tokenizer.json")
    tokenizer = None
    try:
        from tokenizers import Tokenizer  # noqa: E402

        tokenizer = Tokenizer.from_file(engine_tokenizer)
    except Exception:  # noqa: BLE001 - bench continues with estimated counts
        print(
            f"WARNING: no tokenizer at {engine_tokenizer}; prompt_tokens will be estimates",
            file=sys.stderr)

    # One runtime serves every variant: all adapters load once into the
    # lora_weights_map and each request selects its adapter via
    # lora_weights_name (empty for base), matching the original probe.
    lora_map = dict(adapters)
    runtime = load_runtime(rt, args, lora_map)

    # Decode must run as the server runs it: graphs captured at startup.
    cuda_graphs = capture_decode_cuda_graphs(runtime)
    if not cuda_graphs:
        print(
            "ERROR: decoding CUDA graph capture failed; the decode numbers "
            "would not describe the serving configuration",
            file=sys.stderr)
        return 3

    # Tag ordering: base first, then adapters in the order given.
    variants = [("base", None)]
    for name, path in adapters:
        variants.append((name, path))

    records = []
    base_outputs = None
    for tag, adapter_path in variants:
        result = measure_variant(rt,
                                 args,
                                 runtime,
                                 lengths,
                                 tokenizer,
                                 tag,
                                 adapter_path,
                                 base_outputs=base_outputs,
                                 cuda_graphs=cuda_graphs)
        if result is None:
            return 2
        record, outputs = result
        if base_outputs is None:
            base_outputs = outputs
        records.append(record)

    lines = "".join(json.dumps(r) + "\n" for r in records)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(lines)
    else:
        sys.stdout.write(lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
