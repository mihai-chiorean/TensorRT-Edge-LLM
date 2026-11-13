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
setup. The fake runtime models the real pybind contract: metrics accumulate
only while ``set_profiling_enabled(True)`` is active on the calling thread,
mirroring ``SpecDecodeGenerationMetrics::recordRun``'s
``getProfilingEnabled()`` gate in cpp/profiling/metrics.h.
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


class FakeRuntime:
    """LLMRuntime stand-in whose metrics advance only while profiling is on."""

    def __init__(self, state):
        self._state = state

    def handle_request(self, request):
        from types import SimpleNamespace

        profiling_on = self._state["profiling_enabled"]
        generated = self._state["gen_per_run"]
        prompt_tokens = self._state["prompt_tokens"]
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
    ContextCacheConfig, <engine>/visual and the benchmarked 2/5/12 shape."""
    module, run, state, engine_dir = bench_env
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc == 0

    args = ctor_args_dict(last_ctor_call(state), MTP_CTOR_PARAMS)
    assert args["engine_dir"] == str(engine_dir)
    assert args["multimodal_engine_dir"] == str(engine_dir / "visual")
    assert args["draft_top_k"] == module.DEFAULT_DRAFT_TOP_K == 2
    assert args["draft_step"] == module.DEFAULT_DRAFT_STEP == 5
    assert args["verify_tree_size"] == module.DEFAULT_VERIFY_TREE_SIZE == 12
    assert args["checkpoint_dir"] == ""
    assert args["draft_checkpoint_dir"] == ""
    assert isinstance(args["context_cache_config"], FakeContextCacheConfig)
    assert args["context_cache_config"].enabled is True
    assert args["dflash_block_size"] == 0
    assert args["host_embedding_tables"] is False
    assert args["host_ple_table"] is False


def test_plain_constructor_matches_pybind_signature(bench_env):
    """The plain path must call the 7-arg pybind constructor."""
    module, run, state, engine_dir = bench_env
    state["mtp"] = False
    rc = run(["--engine", str(engine_dir)] + BASE_ARGS)
    assert rc == 0

    args = ctor_args_dict(last_ctor_call(state), PLAIN_CTOR_PARAMS)
    assert args["engine_dir"] == str(engine_dir)
    assert args["multimodal_engine_dir"] == str(engine_dir / "visual")
    assert args["checkpoint_dir"] == ""
    assert isinstance(args["context_cache_config"], FakeContextCacheConfig)
    assert args["context_cache_config"].enabled is True
    assert args["host_embedding_tables"] is False
    assert args["host_ple_table"] is False


def test_visual_dir_falls_back_to_empty_when_missing(bench_env, tmp_path):
    """No <engine>/visual directory: the pybind default '' must be passed."""
    module, run, state, engine_dir = bench_env
    (engine_dir / "visual").rmdir()
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc == 0
    args = ctor_args_dict(last_ctor_call(state), MTP_CTOR_PARAMS)
    assert args["multimodal_engine_dir"] == ""


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
    assert record["verify_steps"] == state["iterations"]
    assert record["verify_steps"] > 0
    assert record["accepted_draft_tokens"] == state["accepted"]
    assert record["proposed_draft_tokens"] == state["proposed"]
    # 257 generated tokens over the recorded iterations
    assert record["tokens_per_verify_step"] == pytest.approx(
        record["decode_generated"] / record["verify_steps"])


def test_zero_iterations_on_mtp_engine_exits_nonzero(bench_env, capsys):
    """A metrics pass that records nothing must fail loudly, not write null."""
    module, run, state, engine_dir = bench_env
    state["gen_per_run"] = 0  # nothing generated -> recordRun never advances
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc != 0
    err = capsys.readouterr().err
    assert "zero spec-decode iterations" in err


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
    runs must all see it off, exactly one run (the metrics pass) sees it on,
    and the flag must be off again after that pass."""
    module, run, state, engine_dir = bench_env
    rc = run(["--engine", str(engine_dir), "--mtp"] + BASE_ARGS)
    assert rc == 0
    flags = state["handle_requests"]
    assert flags, "the bench never called handle_request"
    # The metrics pass is the only handle_request with profiling enabled.
    assert flags.count(
        True) == 1, f"expected exactly one profiled run, got {flags}"
    profiled_index = flags.index(True)
    assert all(flag is False for flag in flags[:profiled_index]), (
        f"profiling was on during a timed run before the metrics pass: {flags}"
    )
    assert all(flag is False for flag in flags[profiled_index + 1:]), (
        f"profiling stayed on after the metrics pass: {flags}")
    assert state[
        "profiling_enabled"] is False, "flag not restored after the metrics pass"


def test_plain_engine_skips_spec_metrics(bench_env, capsys):
    """Without --mtp no metrics pass runs and the spec fields stay null."""
    module, run, state, engine_dir = bench_env
    state["mtp"] = False
    rc = run(["--engine", str(engine_dir)] + BASE_ARGS)
    assert rc == 0
    record = first_record(capsys.readouterr().out)
    assert record["mtp"] is False
    assert record["tokens_per_verify_step"] is None
    assert record["verify_steps"] is None


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


def test_parse_args_defaults_build_dir_from_env(monkeypatch):
    module = load_lora_bench()
    monkeypatch.setenv("B", "/opt/build")
    args = module.parse_args(["--engine", "/engines/e"])
    assert args.build_dir == "/opt/build"
    assert args.ttft_lengths == module.DEFAULT_TTFT_LENGTHS
    assert args.reps == module.DEFAULT_REPS


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
