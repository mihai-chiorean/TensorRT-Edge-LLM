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


class FakeRtModule(types.ModuleType):
    """Stand-in for the _edgellm_runtime pybind module."""

    def __init__(self, state):
        super().__init__("_edgellm_runtime")
        self._state = state
        self.profiling_enabled = False

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

        class LLMRuntime:

            def __init__(self, *args):
                pass

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
