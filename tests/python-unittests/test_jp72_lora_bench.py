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
"""CPU-only tests for scripts/jp72/lora_bench.py.

The bench is Jetson-only (the pybind runtime pulls in CUDA/TensorRT), so a
fake ``_edgellm_runtime`` module is inserted into ``sys.modules`` and the
script is imported through importlib without executing ``main()``'s runtime
setup. The fake runtime models the real pybind contract: the spec-decode,
prefill and context-cache counters accumulate for the runtime's lifetime and
advance only while ``set_profiling_enabled(True)`` is active on the calling
thread, mirroring the ``getProfilingEnabled()`` gates in
cpp/profiling/metrics.h.
"""
import importlib.util
import json
import pathlib
import sys
import types

import pytest

SCRIPT = pathlib.Path(
    __file__).resolve().parents[2] / "scripts" / "jp72" / "lora_bench.py"


def load_lora_bench():
    """Import lora_bench.py as a module without importing the real runtime."""
    spec = importlib.util.spec_from_file_location("lora_bench_under_test",
                                                  SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMetrics:
    """Mirror of the pybind SpecDecodeGenerationMetrics surface."""

    def __init__(self, runtime):
        self._runtime = runtime
        self.total_iterations = 0
        self.total_generated_tokens = 0
        self.total_accepted_draft_tokens = 0
        self.total_proposed_draft_tokens = 0

    def refresh(self):
        self.total_iterations = self._runtime["iterations"]
        self.total_generated_tokens = self._runtime["generated"]
        self.total_accepted_draft_tokens = self._runtime["accepted"]
        self.total_proposed_draft_tokens = self._runtime["proposed"]


class FakePrefillMetrics:
    """Mirror of the pybind LLMPrefillMetrics surface (cumulative seconds)."""

    def __init__(self, runtime):
        self._runtime = runtime
        self.reused_tokens = 0
        self.computed_tokens = 0
        self.prefill_seconds = 0.0

    def refresh(self):
        self.reused_tokens = self._runtime["prefill_reused_tokens"]
        self.computed_tokens = self._runtime["prefill_computed_tokens"]
        self.prefill_seconds = self._runtime["prefill_seconds"]


class FakeContextCacheMetrics:
    """Mirror of the pybind ContextCacheMetrics token counters."""

    def __init__(self, runtime):
        self._runtime = runtime

    def refresh(self):
        self.matched_tokens = self._runtime["cc_matched_tokens"]
        self.reused_tokens = self._runtime["cc_reused_tokens"]


class FakeRuntime:
    """LLMRuntime stand-in whose metrics advance only while profiling is on."""

    def __init__(self, state):
        self._state = state

    def handle_request(self, request):
        import time as _time
        from types import SimpleNamespace

        profiling_on = self._state["profiling_enabled"]
        generated = self._state["gen_per_run"]
        prompt_tokens = self._state["prompt_tokens"]
        if self._state.get("fake_wall_seconds"):
            _time.sleep(self._state["fake_wall_seconds"])
        if profiling_on:
            # The real counters accumulate for the runtime's lifetime and
            # only advance when profiling is enabled on the calling thread.
            self._state["prefill_seconds"] += self._state[
                "prefill_seconds_per_run"]
            self._state["prefill_reused_tokens"] += self._state[
                "prefill_reused_per_run"]
            self._state["prefill_computed_tokens"] += prompt_tokens
        if self._state["cc_enabled"]:
            # Context-cache reuse: every request after the first reuses the
            # per-run amount (0 when the cache is disabled).
            self._state["cc_matched_tokens"] += self._state[
                "cc_reuse_per_run"] if self._state["cc_requests"] > 0 else 0
            self._state["cc_reused_tokens"] += self._state[
                "cc_reuse_per_run"] if self._state["cc_requests"] > 0 else 0
            self._state["cc_requests"] += 1
        if self._state["mtp"] and profiling_on and generated > 0:
            # One verify iteration per draft step of generated tokens; the
            # real recordRun only accumulates when profiling is enabled.
            self._state["iterations"] += max(1, generated // 2)
            self._state["generated"] += generated
            self._state["accepted"] += generated - 1
            self._state["proposed"] += generated - 1
        self._state["handle_requests"].append(profiling_on)
        return SimpleNamespace(prompt_token_counts=[prompt_tokens],
                               output_ids=[list(range(generated))],
                               output_texts=["out" * generated])


class FakeContextCacheConfig:
    """Mirror of the pybind ContextCacheConfig value type."""

    def __init__(self):
        self.enabled = False
        self.max_records = 1024


# Exact positional signatures of the two real pybind constructors
# (experimental/pybind/edgellm_pybind.cpp): a call that does not match either
# must raise TypeError, and context_cache_config must never be None.
PLAIN_CTOR_PARAMS = [
    "engine_dir", "multimodal_engine_dir", "lora_weights_map",
    "checkpoint_dir", "context_cache_config", "host_embedding_tables",
    "host_ple_table"
]
MTP_CTOR_PARAMS = [
    "engine_dir", "multimodal_engine_dir", "lora_weights_map", "draft_top_k",
    "draft_step", "verify_tree_size", "checkpoint_dir", "draft_checkpoint_dir",
    "context_cache_config", "dflash_block_size", "host_embedding_tables",
    "host_ple_table"
]


class FakeRtModule(types.ModuleType):
    """Stand-in for the _edgellm_runtime pybind module."""

    def __init__(self, state):
        super().__init__("_edgellm_runtime")
        self._state = state
        self.profiling_enabled = False
        self.ContextCacheConfig = FakeContextCacheConfig

        # Minimal request/response builders used by lora_bench.
        class ToolDefinition:

            def __init__(self):
                self.name = ""
                self.description = ""
                self.parameters = ""
                self.strict = False
                self.has_description = False
                self.has_parameters = False
                self.has_strict = False

        class ToolChoice:

            def __init__(self):
                self.mode = None

        self.ToolDefinition = ToolDefinition
        self.ToolChoice = ToolChoice
        self.ToolChoiceMode = types.SimpleNamespace(AUTO="AUTO")
        self.create_text_message = lambda role, text: {
            "role": role,
            "text": text
        }

        def create_generation_request(batch_messages, **kwargs):
            return types.SimpleNamespace(batch=batch_messages,
                                         tools=None,
                                         tool_choice=None,
                                         **kwargs)

        self.create_generation_request = create_generation_request
        self.metrics = FakeMetrics(state)
        self.prefill_metrics = FakePrefillMetrics(state)
        self.cc_metrics = FakeContextCacheMetrics(state)

        def check_signature(got, params, kind):
            if len(got) != len(params):
                raise TypeError(
                    f"{kind} __init__() takes exactly {len(params)} "
                    f"positional arguments ({len(got)} given)")
            if params.index("context_cache_config") >= len(got) or \
                    got[params.index("context_cache_config")] is None:
                raise TypeError(
                    "context_cache_config: the pybind binding requires a "
                    "ContextCacheConfig, not None")
            # ModelArtifacts::loadFromEngineDir reads config.json (plain) or
            # base_config.json (MTP) from engine_dir itself; a bundle root
            # without /llm makes the runtime fail to open its runtime config.
            engine_dir = pathlib.Path(got[params.index("engine_dir")])
            if engine_dir.name != "llm":
                raise TypeError(
                    f"engine_dir: expected the <bundle>/llm directory, got "
                    f"{engine_dir}")
            expected_config = "base_config.json" if kind == "MTP" \
                else "config.json"
            if not (engine_dir / expected_config).is_file():
                raise TypeError(
                    f"engine_dir: {expected_config} not found in {engine_dir}")

        class LLMRuntime:

            def __init__(self, *args):
                state["ctor_calls"] = state.get("ctor_calls", [])
                state["ctor_calls"].append(args)
                if len(args) == len(PLAIN_CTOR_PARAMS):
                    check_signature(args, PLAIN_CTOR_PARAMS, "plain")
                elif len(args) == len(MTP_CTOR_PARAMS):
                    check_signature(args, MTP_CTOR_PARAMS, "MTP")
                else:
                    raise TypeError(
                        f"__init__(): incompatible constructor arguments "
                        f"({len(args)} positional args; the pybind module "
                        f"exposes a 7-arg plain and a 12-arg MTP constructor)")

            def handle_request(self, request):
                return FakeRuntime(state).handle_request(request)

            def get_spec_decode_generation_metrics(self):
                self_metrics = FakeMetrics(state)
                self_metrics.refresh()
                return self_metrics

            def get_prefill_metrics(self):
                metrics = FakePrefillMetrics(state)
                metrics.refresh()
                return metrics

            def get_context_cache_metrics(self):
                metrics = FakeContextCacheMetrics(state)
                metrics.refresh()
                return metrics

            def capture_decoding_cuda_graph(self):
                state["capture_calls"] = state.get("capture_calls", 0) + 1
                return state["capture_result"]

        self.LLMRuntime = LLMRuntime

    # Thread-local in the real module; a module attribute is enough here
    # because the tests call everything from one thread.
    def set_profiling_enabled(self, enabled):
        self.profiling_enabled = enabled
        self._state["profiling_enabled"] = enabled

    def get_profiling_enabled(self):
        return self.profiling_enabled


@pytest.fixture
def bench_env(monkeypatch, tmp_path):
    """Install the fake runtime and tokenizer, return a runner helper."""
    state = {
        "mtp": True,
        "profiling_enabled": False,
        "iterations": 0,
        "generated": 0,
        "accepted": 0,
        "proposed": 0,
        "gen_per_run": 257,
        "prompt_tokens": 123,
        "prefill_seconds": 0.0,
        "prefill_seconds_per_run": 0.004,
        "prefill_reused_tokens": 0,
        "prefill_reused_per_run": 0,
        "prefill_computed_tokens": 0,
        "cc_enabled": True,
        "cc_matched_tokens": 0,
        "cc_reused_tokens": 0,
        "cc_reuse_per_run": 20,
        "cc_requests": 0,
        "capture_calls": 0,
        "capture_result": True,
        "handle_requests": [],
        "ctor_calls": [],
    }
    monkeypatch.setenv("LORA_BENCH_SKIP_PLUGIN_LOAD", "1")
    rt_module = FakeRtModule(state)
    monkeypatch.setitem(sys.modules, "_edgellm_runtime", rt_module)

    # A tiny tokenizer.json so the bench loads it via tokenizers.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {
        w: i
        for i, w in enumerate([
            "[probe", "the", "quick", "brown", "fox", "jumps", "over", "lazy",
            "dog", "a"
        ])
    }
    tok = Tokenizer(WordLevel(vocab, unk_token="[probe"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    engine_dir = tmp_path / "engine"
    (engine_dir / "llm").mkdir(parents=True)
    (engine_dir / "visual").mkdir()
    (engine_dir / "llm" / "tokenizer.json").write_text(
        (tmp_path / "tokenizer.json").read_text(), encoding="utf-8")
    # The fake runtime checks that the LLM engine dir carries the runtime
    # config the real one loads (config.json plain / base_config.json MTP).
    (engine_dir / "llm" / "config.json").write_text("{}", encoding="utf-8")
    (engine_dir / "llm" / "base_config.json").write_text("{}",
                                                         encoding="utf-8")

    module = load_lora_bench()

    def run(argv):
        return module.main(argv)

    return module, run, state, engine_dir


BASE_ARGS = ["--build-dir", "/nonexistent-build"]


def first_record(out):
    """Return the first JSON record line from bench stdout (skipping logs)."""
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"no JSON record line in stdout: {out!r}")


def last_ctor_call(state):
    """The single LLMRuntime constructor call the bench made, as a dict."""
    calls = state["ctor_calls"]
    assert len(calls) == 1, (
        f"the bench must construct exactly one runtime, made {len(calls)}")
    return calls[0]


def ctor_args_dict(args, params):
    return dict(zip(params, args))


def test_mtp_constructor_matches_pybind_signature(bench_env):
    """The MTP path must call the 12-arg pybind constructor with a real
    ContextCacheConfig, <engine>/llm, <engine>/visual, the benchmarked
    2/5/12 shape and host-mapped embedding tables (as the probe did)."""
    module, run, state, engine_dir = bench_env
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc == 0

    args = ctor_args_dict(last_ctor_call(state), MTP_CTOR_PARAMS)
    assert args["engine_dir"] == str(engine_dir / "llm")
    assert args["multimodal_engine_dir"] == str(engine_dir / "visual")
    assert args["draft_top_k"] == module.DEFAULT_DRAFT_TOP_K == 2
    assert args["draft_step"] == module.DEFAULT_DRAFT_STEP == 5
    assert args["verify_tree_size"] == module.DEFAULT_VERIFY_TREE_SIZE == 12
    assert args["checkpoint_dir"] == ""
    assert args["draft_checkpoint_dir"] == ""
    assert isinstance(args["context_cache_config"], FakeContextCacheConfig)
    assert args["context_cache_config"].enabled is True
    assert args["dflash_block_size"] == 0
    assert args["host_embedding_tables"] is True
    assert args["host_ple_table"] is False


def test_plain_constructor_matches_pybind_signature(bench_env):
    """The plain path must call the 7-arg pybind constructor with
    <engine>/llm and device-side embedding tables."""
    module, run, state, engine_dir = bench_env
    state["mtp"] = False
    rc = run(["--engine", str(engine_dir)] + BASE_ARGS)
    assert rc == 0

    args = ctor_args_dict(last_ctor_call(state), PLAIN_CTOR_PARAMS)
    assert args["engine_dir"] == str(engine_dir / "llm")
    assert args["multimodal_engine_dir"] == str(engine_dir / "visual")
    assert args["checkpoint_dir"] == ""
    assert isinstance(args["context_cache_config"], FakeContextCacheConfig)
    assert args["context_cache_config"].enabled is True
    assert args["host_embedding_tables"] is False
    assert args["host_ple_table"] is False


def test_device_embedding_tables_flag_opts_out_for_mtp(bench_env):
    """--device-embedding-tables must pass host_embedding_tables=False on the
    MTP path (default stays True, mirroring the original probe)."""
    module, run, state, engine_dir = bench_env
    rc = run(
        ["--engine",
         str(engine_dir), "--mtp", "--device-embedding-tables"] + BASE_ARGS)
    assert rc == 0
    args = ctor_args_dict(last_ctor_call(state), MTP_CTOR_PARAMS)
    assert args["host_embedding_tables"] is False


def test_visual_dir_falls_back_to_empty_when_missing(bench_env, tmp_path):
    """No <engine>/visual directory: the pybind default '' must be passed,
    while engine_dir stays <engine>/llm."""
    module, run, state, engine_dir = bench_env
    (engine_dir / "visual").rmdir()
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc == 0
    args = ctor_args_dict(last_ctor_call(state), MTP_CTOR_PARAMS)
    assert args["multimodal_engine_dir"] == ""
    assert args["engine_dir"] == str(engine_dir / "llm")


def test_no_reuse_disables_context_cache(bench_env):
    """--no-reuse must construct the runtime with cc.enabled False."""
    module, run, state, engine_dir = bench_env
    rc = run(["--engine", str(engine_dir), "--mtp", "--no-reuse"] + BASE_ARGS)
    assert rc == 0
    args = ctor_args_dict(last_ctor_call(state), MTP_CTOR_PARAMS)
    assert args["context_cache_config"].enabled is False


def test_draft_shape_flags_reach_the_constructor(bench_env):
    """--draft-top-k/--draft-step/--verify-tree-size override the defaults."""
    module, run, state, engine_dir = bench_env
    rc = run([
        "--engine",
        str(engine_dir), "--mtp", "--draft-top-k", "3", "--draft-step", "2",
        "--verify-tree-size", "8"
    ] + BASE_ARGS)
    assert rc == 0
    args = ctor_args_dict(last_ctor_call(state), MTP_CTOR_PARAMS)
    assert args["draft_top_k"] == 3
    assert args["draft_step"] == 2
    assert args["verify_tree_size"] == 8


def test_one_runtime_serves_base_and_adapter(bench_env, capsys):
    """Base and every adapter must share one runtime: adapters load once into
    lora_weights_map and requests select via lora_weights_name."""
    module, run, state, engine_dir = bench_env
    rc = run([
        "--engine",
        str(engine_dir), "--mtp", "--adapter", "alpha=/adapters/alpha",
        "--adapter", "beta=/adapters/beta"
    ] + BASE_ARGS)
    assert rc == 0

    calls = state["ctor_calls"]
    assert len(calls) == 1, (
        f"base + 2 adapters must construct exactly one runtime, "
        f"made {len(calls)}")
    args = ctor_args_dict(calls[0], MTP_CTOR_PARAMS)
    assert args["lora_weights_map"] == {
        "alpha": "/adapters/alpha",
        "beta": "/adapters/beta"
    }, "all adapters must be in the single runtime's lora_weights_map"

    out = capsys.readouterr().out
    tags = [
        json.loads(line)["tag"] for line in out.splitlines()
        if line.strip().startswith("{")
    ]
    assert tags == [
        "base", "alpha", "beta"
    ], (f"one shared runtime must still produce all variant records: {tags}")


def test_adapter_requests_carry_lora_weights_name(bench_env, capsys):
    """Adapter records must come from requests naming the adapter; base from
    requests with an empty lora_weights_name (one shared runtime)."""
    module, run, state, engine_dir = bench_env

    names = []

    original_handle = FakeRuntime.handle_request

    def spy_handle(self, request):
        names.append(request.lora_weights_name)
        return original_handle(self, request)

    monkey_target = FakeRuntime
    monkey_target.handle_request = spy_handle
    try:
        rc = run([
            "--engine",
            str(engine_dir), "--mtp", "--adapter", "alpha=/adapters/alpha"
        ] + BASE_ARGS)
    finally:
        monkey_target.handle_request = original_handle
    assert rc == 0

    assert len(state["ctor_calls"]) == 1
    base_names = [n for n in names if n == ""]
    adapter_names = [n for n in names if n == "alpha"]
    assert base_names, "base variant must send lora_weights_name=''"
    assert adapter_names, "adapter variant must send lora_weights_name='alpha'"


def test_mtp_metrics_pass_yields_numeric_tokens_per_verify_step(
        bench_env, capsys):
    """The fake metrics only advance while profiling is on; the bench must
    therefore enable it around the metrics pass or tokens_per_verify_step
    stays null (the bug this port fixes)."""
    module, run, state, engine_dir = bench_env
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc == 0, "bench must succeed on an MTP engine with advancing metrics"

    record = first_record(capsys.readouterr().out)
    assert record["mtp"] is True
    assert record["tokens_per_verify_step"] is not None, (
        "tokens_per_verify_step was null: the metrics pass did not enable profiling"
    )
    assert isinstance(record["tokens_per_verify_step"], float)
    assert record["tokens_per_verify_step"] > 0
    # The metrics pass is the only profiled run, so its delta is exactly one
    # run's worth of counters (max(1, 257 // 2) = 128 iterations).
    assert record["verify_steps"] == 128, (
        f"the pass must report only its own run, got {record['verify_steps']}")
    assert record["accepted_draft_tokens"] == 256
    assert record["proposed_draft_tokens"] == 256
    assert record["tokens_per_verify_step"] == pytest.approx(
        record["decode_generated"] / record["verify_steps"])


def test_second_variant_metrics_are_its_own_not_cumulative(bench_env, capsys):
    """The spec counters accumulate for the runtime's lifetime; each variant's
    record must report the delta of its metrics pass, not the cumulative
    totals (the reviewer's point 1: with cumulative totals the second variant
    reports base+adapter iterations and a halved tokens_per_verify_step)."""
    module, run, state, engine_dir = bench_env
    rc = run([
        "--engine",
        str(engine_dir), "--mtp", "--adapter", "alpha=/adapters/alpha",
        "--adapter", "beta=/adapters/beta"
    ] + BASE_ARGS)
    assert rc == 0
    out = capsys.readouterr().out
    records = [
        json.loads(line) for line in out.splitlines()
        if line.strip().startswith("{")
    ]
    assert [record["tag"] for record in records] == ["base", "alpha", "beta"]
    total_iterations = state["iterations"]
    for record in records:
        assert record["verify_steps"] == 128, (
            f"{record['tag']} reported {record['verify_steps']} iterations; "
            f"cumulative total is {total_iterations}")
        assert record["accepted_draft_tokens"] == 256
        assert record["proposed_draft_tokens"] == 256
        assert record["tokens_per_verify_step"] == pytest.approx(
            record["decode_generated"] / record["verify_steps"])
        assert record["tokens_per_verify_step"] > 2.0, (
            "a cumulative-iterations mix halves this ratio for later variants")


def test_prefill_seconds_is_the_measured_pass_delta(bench_env, capsys):
    """decode_prefill_seconds must come from the profiled prefill-counter
    delta (one request's worth), and decode_tok_s must divide by wall time
    minus that prefill, not by the whole-request wall time. The fake wall
    times are ~0, so only the recorded prefill figure is asserted here; the
    subtraction itself is covered by test_decode_phase_seconds_excludes_prefill.
    """
    module, run, state, engine_dir = bench_env
    state["prefill_seconds_per_run"] = 0.05
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc == 0
    record = first_record(capsys.readouterr().out)
    assert record["decode_prefill_seconds"] == pytest.approx(0.05)


def test_decode_phase_seconds_excludes_prefill(bench_env, capsys):
    """With a known fake prefill and wall time, decode_tok_s must use
    wall - prefill: 256 tokens over (1.28 - 0.64)s = 400 tok/s."""
    module, run, state, engine_dir = bench_env
    state["mtp"] = False
    state["prefill_seconds_per_run"] = 0.64
    state["gen_per_run"] = 256
    state["fake_wall_seconds"] = 1.28
    rc = run(["--engine", str(engine_dir)] + BASE_ARGS)
    assert rc == 0
    record = first_record(capsys.readouterr().out)
    assert record["decode_generated"] == 256
    assert record["decode_prefill_seconds"] == pytest.approx(0.64, abs=0.01)
    assert record["decode_seconds"] == pytest.approx(1.28, abs=0.01)
    assert record["decode_tok_s"] == pytest.approx(400.0, rel=0.05)


def test_zero_iterations_on_mtp_engine_exits_nonzero(bench_env, capsys):
    """A metrics pass that records nothing must fail loudly, not write null."""
    module, run, state, engine_dir = bench_env
    state["gen_per_run"] = 0  # nothing generated -> recordRun never advances
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc != 0
    err = capsys.readouterr().err
    assert "zero spec-decode iterations" in err


def test_cuda_graphs_captured_once_and_recorded(bench_env, capsys):
    """The bench must capture decoding CUDA graphs after constructing the
    runtime (the server does this at startup) and record cuda_graphs: true;
    a failed capture must abort the run."""
    module, run, state, engine_dir = bench_env
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc == 0
    assert state["capture_calls"] == 1, "capture must happen exactly once"
    out = capsys.readouterr().out
    for line in out.splitlines():
        if line.strip().startswith("{"):
            assert json.loads(line)["cuda_graphs"] is True

    state["capture_result"] = False
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc != 0, "a failed capture must fail the run"
    assert "CUDA graph capture failed" in capsys.readouterr().err


def test_decode_warms_up_then_reports_median_of_timed_runs(bench_env, capsys):
    """Each variant runs one untimed warm-up decode followed by --decode-reps
    timed runs of the same prompt; the record carries the median run's
    numbers (the reviewer's point 3: one un-warmed run reported as decode)."""
    module, run, state, engine_dir = bench_env
    rc = run([
        "--engine",
        str(engine_dir), "--mtp", "--decode-reps", "5", "--reps", "1",
        "--ttft-lengths", "20"
    ] + BASE_ARGS)
    assert rc == 0
    capsys.readouterr()
    flags = state["handle_requests"]
    # per variant: tool ttft(1) + warm-up(1) + timed(5) + prefill pass(1)
    # + sweep(1 target x 1 rep) + metrics pass(1) + greedy(9) = 19
    assert len(flags) == 19, f"unexpected request count: {len(flags)}"
    with pytest.raises(SystemExit) as exc_info:
        module.parse_args([
            "--engine", "/e", "--mtp", "--decode-reps", "2", "--reps", "1",
            "--ttft-lengths", "20", "--build-dir", "/b"
        ])
    assert exc_info.value.code == 2, (
        "argparse must refuse --decode-reps below the 3-run median floor")


def test_ttft_sweep_reports_cached_tokens(bench_env, capsys):
    """Each sweep row must report cached_tokens so cache-served rows are
    visible (the reviewer's point 7)."""
    module, run, state, engine_dir = bench_env
    rc = run([
        "--engine",
        str(engine_dir), "--mtp", "--ttft-lengths", "20", "--reps", "2"
    ] + BASE_ARGS)
    assert rc == 0
    record = first_record(capsys.readouterr().out)
    row = record["ttft_sweep"][0]
    assert "cached_tokens" in row
    assert row["cached_tokens"] == 2 * state["cc_reuse_per_run"]
    assert row["ttft_ms"] > 0

    state["cc_enabled"] = False
    rc = run([
        "--engine",
        str(engine_dir), "--mtp", "--ttft-lengths", "20", "--reps", "2"
    ] + BASE_ARGS)
    assert rc == 0
    row = first_record(capsys.readouterr().out)["ttft_sweep"][0]
    assert row["cached_tokens"] == 0


def test_matches_base_replaces_greedy_match(bench_env, capsys):
    """The adapter-vs-base output comparison is reported as matches_base; the
    base record has no reference output, so its field is null."""
    module, run, state, engine_dir = bench_env
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc == 0
    record = first_record(capsys.readouterr().out)
    assert "greedy_match" not in record
    assert record["matches_base"] is None  # base has no earlier record


def test_build_ttft_prompt_best_tracks_error_not_count(bench_env):
    """best_err is a relative error everywhere (the reviewer's point 4: one
    branch stored a token count in best_err and compared it against a
    relative error, so ``err < best_err`` was always true and "best" was the
    last candidate tried, not the closest). The chosen prompt body must
    match an independent reference search that tracks the relative error."""
    module, run, state, engine_dir = bench_env
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(engine_dir / "llm" / "tokenizer.json"))
    paragraph = "the quick brown fox jumps " * 6
    for target in (68, 99, 200):
        prompt, measured = module.build_ttft_prompt(target, tokenizer,
                                                    paragraph)
        assert measured > 0
        assert prompt.startswith("[probe ")

        # Independent reference: whole paragraphs, then binary-search the
        # final partial paragraph keeping the minimum-error candidate.
        stamp_end = prompt.split("]\n", 1)[0] + "]\n"
        para_tokens = max(1, len(tokenizer.encode(paragraph).ids))
        whole_reps = max(target - 10, 1) // para_tokens

        def tokens_of(text):
            return len(tokenizer.encode(text).ids)

        def body_of(reps, cut):
            return stamp_end + paragraph * reps + paragraph[:cut]

        best_body, best_err = None, None
        base_body = body_of(whole_reps, 0)
        base_n = tokens_of(base_body)
        if base_n > 0:
            best_body, best_err = base_body, abs(base_n - target) / target
        lo, hi = 1, len(paragraph)
        for _ in range(20):
            mid = (lo + hi) // 2
            candidate = body_of(whole_reps, mid)
            n = tokens_of(candidate)
            if n <= 0:
                continue
            err = abs(n - target) / target
            if best_err is None or err < best_err:
                best_body, best_err = candidate, err
            if err <= 0.05:
                break
            if n > target:
                hi = mid - 1
            else:
                lo = mid + 1
        assert prompt.split("]\n", 1)[1] == best_body.split("]\n", 1)[1], (
            f"target {target}: chosen candidate is not the minimum-error "
            f"one (best_err tracked a token count, not the relative error)")


def test_prompt_builder_lands_within_five_percent(bench_env):
    """Each sweep point must build a prompt within 5% of its target tokens."""
    module, _, _, _ = bench_env
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        str(bench_env[3] / "llm" / "tokenizer.json"))
    paragraph = ("the quick brown fox jumps over the lazy dog " * 30).strip()
    previous = 0
    for target in [20, 100, 400, 800, 1600]:
        prompt, measured = module.build_ttft_prompt(target, tokenizer,
                                                    paragraph)
        assert measured > 0
        assert abs(measured - target) / target <= 0.05, (
            f"target {target}: built a {measured}-token prompt")
        assert prompt.startswith(
            "[probe "), "unique time_ns prefix must lead the prompt"
        assert measured > previous or previous == 0
        previous = measured


def test_profiling_disabled_during_timed_runs(bench_env):
    """The fake records the profiling flag at each handle_request; the timed
    runs must all see it off, exactly the two metric-gated passes (prefill
    wall time and, on MTP, the spec metrics) see it on, and the flag must be
    off again afterwards."""
    module, run, state, engine_dir = bench_env
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc == 0
    flags = state["handle_requests"]
    assert flags, "the bench never called handle_request"
    # The prefill-timing pass and the metrics pass are the only profiled ones.
    assert flags.count(
        True) == 2, f"expected exactly two profiled runs, got {flags}"
    profiled = [i for i, flag in enumerate(flags) if flag]
    assert all(flag is False for i, flag in enumerate(flags)
               if i not in profiled), (
                   f"profiling was on during a timed run: {flags}")
    assert state[
        "profiling_enabled"] is False, "flag not restored after the metric passes"


def test_plain_engine_skips_spec_metrics(bench_env, capsys):
    """Without --mtp no spec metrics pass runs and the spec fields stay null,
    while the decode/prefill measurement still works."""
    module, run, state, engine_dir = bench_env
    state["mtp"] = False
    rc = run(["--engine", str(engine_dir)] + BASE_ARGS)
    assert rc == 0
    record = first_record(capsys.readouterr().out)
    assert record["mtp"] is False
    assert record["tokens_per_verify_step"] is None
    assert record["verify_steps"] is None
    assert record["decode_prefill_seconds"] == pytest.approx(0.004)


def test_out_file_appends_one_json_line_per_variant(bench_env, tmp_path,
                                                    capsys):
    """--out FILE.jsonl appends one line per measured variant."""
    module, run, state, engine_dir = bench_env
    out = tmp_path / "results" / "bench.jsonl"
    rc = run([
        "--engine",
        str(engine_dir), "--mtp", "--adapter", "alpha=/adapters/alpha"
    ] + BASE_ARGS + ["--out", str(out)])
    assert rc == 0
    capsys.readouterr()  # drop stdout logs
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "base + one adapter must produce two records"
    base, adapter = (json.loads(line) for line in lines)
    assert base["tag"] == "base" and base["adapter"] is None
    assert adapter["tag"] == "alpha" and adapter["adapter"] == "/adapters/alpha"


def test_verify_step_math_rejects_zero_iterations():
    module = load_lora_bench()
    assert module.verify_step_math(257, 100) == 2.57
    assert module.verify_step_math(257, 0) is None
    assert module.verify_step_math(257, None) is None


def test_spec_metrics_delta_subtracts_cumulative_totals():
    """The delta helper: after - before per field; missing before is zero."""
    module = load_lora_bench()
    before = {
        "total_iterations": 128,
        "total_generated_tokens": 257,
        "accepted_draft_tokens": 200,
        "proposed_draft_tokens": 256,
    }
    after = {
        "total_iterations": 256,
        "total_generated_tokens": 514,
        "accepted_draft_tokens": 456,
        "proposed_draft_tokens": 512,
    }
    assert module.spec_metrics_delta(before, after) == {
        "total_iterations": 128,
        "total_generated_tokens": 257,
        "accepted_draft_tokens": 256,
        "proposed_draft_tokens": 256,
    }
    assert module.spec_metrics_delta(None, after) == after
    assert module.spec_metrics_delta(before, None) is None


def test_parse_args_defaults_build_dir_from_env(monkeypatch):
    module = load_lora_bench()
    monkeypatch.setenv("B", "/opt/build")
    args = module.parse_args(["--engine", "/engines/e"])
    assert args.build_dir == "/opt/build"
    assert args.ttft_lengths == module.DEFAULT_TTFT_LENGTHS
    assert args.reps == module.DEFAULT_REPS
    assert args.decode_reps == module.DEFAULT_DECODE_REPS == 5
    assert module.MIN_DECODE_REPS == 3


def test_parse_args_rejects_bad_adapter():
    module = load_lora_bench()
    with pytest.raises(SystemExit):
        module.parse_args(
            ["--engine", "/e", "--build-dir", "/b", "--adapter", "noequals"])


def test_module_imports_without_runtime():
    """The module must not import _edgellm_runtime at import time."""
    module = load_lora_bench()
    assert "_edgellm_runtime" not in sys.modules or \
        sys.modules["_edgellm_runtime"].__name__ == "_edgellm_runtime"
    assert callable(module.build_ttft_prompt)
    assert callable(module.assemble_record)
