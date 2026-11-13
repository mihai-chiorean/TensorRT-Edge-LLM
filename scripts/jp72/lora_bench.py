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

Ported from the lab-jetson probe_mtp.py so the bench lives in the repo. The
probe's null ``tokens_per_verify_step`` is fixed here: the runtime only records
``SpecDecodeGenerationMetrics`` (cpp/profiling/metrics.h) when the thread-local
profiling flag is on, so a separate metrics pass per prompt and adapter runs
with profiling enabled and the flag is restored afterwards; timed decode runs
always execute with profiling off.

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
    ttft_sweep              list of {"target_tokens", "prompt_tokens", "ttft_ms"}
    decode_*                decode prompt results (see RECORD_FIELDS)
    tokens_per_verify_step  generated / iterations from the metrics pass
    verify_steps            total_iterations from the metrics pass
    accepted_draft_tokens   total_accepted_draft_tokens (when exposed)
    proposed_draft_tokens   total_proposed_draft_tokens (when exposed)
    greedy_match            fraction of 9 prompts matching the base output

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
    "prompt_tokens",
    "ttft_ms_tool",
    "ttft_sweep",
    "decode_prompt_tokens",
    "decode_generated",
    "decode_seconds",
    "decode_tok_s",
    "tokens_per_verify_step",
    "verify_steps",
    "accepted_draft_tokens",
    "proposed_draft_tokens",
    "greedy_match",
]


def parse_args(argv=None):
    """Parse the CLI; importable without the runtime for --help/tests."""
    parser = argparse.ArgumentParser(
        prog="lora_bench.py",
        description=
        "LoRA/MTP engine bench (Jetson; touches CUDA via the pybind runtime).")
    parser.add_argument("--engine",
                        required=True,
                        help="Engine directory (base+draft for --mtp).")
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
    whole_reps, remainder_target = divmod(max(target_tokens - 10, 1),
                                          para_tokens)
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
        best, best_err = base, base_n
    # Refine the partial paragraph by characters: binary-search the cut point
    # so the last (partial) paragraph contributes ~remainder_target tokens.
    lo, hi = 1, para_len
    for _ in range(20):
        mid = (lo + hi) // 2
        candidate = base + paragraph[:mid]
        measured = count_tokens(tokenizer, candidate)
        if measured <= 0:
            continue
        err = abs(measured - target_tokens) / target_tokens
        if best_err is None or err < best_err:
            best, best_err = candidate, measured
        if err <= 0.05:
            return candidate, measured
        if measured > target_tokens:
            hi = mid - 1
        else:
            lo = mid + 1
    return best, best_err


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


def verify_step_math(total_generated, total_iterations):
    """tokens_per_verify_step = generated / iterations; None if iterations <= 0."""
    if total_iterations is None or total_iterations <= 0:
        return None
    return total_generated / total_iterations


def assemble_record(tag,
                    mtp,
                    engine_dir,
                    adapter_path,
                    peak_rss_mb,
                    ttft_tool_ms,
                    ttft_sweep,
                    decode,
                    spec_metrics,
                    greedy_match=None):
    """Assemble one output record dict; pure (no runtime)."""
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
        "decode_tok_s":
        round(decode["generated"] /
              decode["seconds"], 2) if decode["seconds"] > 0 else None,
    }
    if spec_metrics is None:
        record["tokens_per_verify_step"] = None
        record["verify_steps"] = None
    else:
        tps = verify_step_math(spec_metrics["total_generated_tokens"],
                               spec_metrics["total_iterations"])
        record["tokens_per_verify_step"] = tps
        record["verify_steps"] = spec_metrics["total_iterations"]
        if "accepted_draft_tokens" in spec_metrics:
            record["accepted_draft_tokens"] = spec_metrics[
                "accepted_draft_tokens"]
        if "proposed_draft_tokens" in spec_metrics:
            record["proposed_draft_tokens"] = spec_metrics[
                "proposed_draft_tokens"]
    record["greedy_match"] = greedy_match
    return record


def run_ttft_sweep(rt, runtime, lengths, reps, tokenizer, lora=""):
    """For each target length build an uncached prompt, generate 1 token, take the median.

    Each rep gets its own time_ns-prefixed prompt, so engine context reuse
    never hits; ``prompt_tokens`` is the engine's measured count from the
    engine tokenizer.
    """
    sweep = []
    for target in lengths:
        ttfts = []
        prompt_tokens = None
        for _ in range(reps):
            prompt, measured = build_ttft_prompt(target, tokenizer)
            req = make_tool_request(rt, prompt, lora=lora)
            t0 = time.perf_counter()
            resp = runtime.handle_request(req)
            ttfts.append((time.perf_counter() - t0) * 1000.0)
            if prompt_tokens is None:
                prompt_tokens = int(resp.prompt_token_counts[0]
                                    ) if resp.prompt_token_counts else measured
        sweep.append({
            "target_tokens": target,
            "prompt_tokens": prompt_tokens,
            "ttft_ms": round(median(ttfts), 3),
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

    Mirrors the original probe and the production runtime wiring: the
    multimodal dir is ``<engine>/visual`` when present (empty string
    otherwise, the pybind default for text-only engines), the context cache
    config is always a real ``rt.ContextCacheConfig`` (``None`` fails the
    pybind constructor), and the MTP drafting shape defaults to the
    benchmarked top-2 / 5-step / verify-12 tree, overridable by flags.
    """
    visual_dir = os.path.join(args.engine, "visual")
    if not os.path.isdir(visual_dir):
        visual_dir = ""
    cc = rt.ContextCacheConfig()
    cc.enabled = not args.no_reuse
    if args.mtp:
        return rt.LLMRuntime(args.engine, visual_dir, lora_map,
                             args.draft_top_k, args.draft_step,
                             args.verify_tree_size, "", "", cc, 0, False,
                             False)
    return rt.LLMRuntime(args.engine, visual_dir, lora_map, "", cc, False,
                         False)


def run_greedy_check(rt, runtime, base_outputs=None, lora=""):
    """Run the 9 greedy prompts; returns (token_id lists, match_fraction_vs_base)."""
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
                    base_outputs=None):
    """Measure one variant (base or one adapter) on the shared runtime.

    Every request carries ``lora_weights_name`` — empty for base, the adapter
    name otherwise — so the runtime switches adapters per request, exactly as
    the original probe did.
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

    # --- decode ---------------------------------------------------------
    decode_prompt = DECODE_PROMPTS[0]
    prompt_tokens, generated, seconds, _ = run_decode(rt,
                                                      runtime,
                                                      decode_prompt,
                                                      lora=lora)

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
            _, metrics_generated, _, _ = run_decode(rt,
                                                    runtime,
                                                    decode_prompt,
                                                    lora=lora)
            spec_metrics = read_spec_metrics(runtime)
        finally:
            rt.set_profiling_enabled(False)
        if spec_metrics is None or spec_metrics["total_iterations"] <= 0:
            print(
                "ERROR: MTP engine reported zero spec-decode iterations "
                "(profiling was enabled for the metrics pass); "
                "cannot compute tokens_per_verify_step",
                file=sys.stderr)
            return None
        # Counters accumulate across every handle_request on this
        # runtime; the timed runs ran with profiling off, so the deltas
        # belong to this pass alone.
        if metrics_generated > 0:
            spec_metrics["total_generated_tokens"] = metrics_generated

    # --- greedy check ---------------------------------------------------
    outputs, match = run_greedy_check(rt,
                                      runtime,
                                      base_outputs=base_outputs,
                                      lora=lora)

    decode = {
        "prompt_tokens": prompt_tokens,
        "generated": generated,
        "seconds": seconds,
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
                             greedy_match=match)
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
                                 base_outputs=base_outputs)
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
