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
"""Runtime EOS metadata contracts without loading torch or model weights."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def checkpoint_utils(monkeypatch):
    source = Path(__file__).resolve(
    ).parents[2] / "tensorrt_edgellm" / "checkpoint" / "checkpoint_utils.py"
    spec = importlib.util.spec_from_file_location("checkpoint_eos_under_test",
                                                  source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__package__", "tensorrt_edgellm.checkpoint")
    monkeypatch.setattr(module, "build_runtime_llm_config_dict",
                        lambda _model: {})

    for name in ("torch", "tensorrt_edgellm", "tensorrt_edgellm.checkpoint",
                 "tensorrt_edgellm.vocab_reduction"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for name, functions in {
            "tensorrt_edgellm._safetensors_io": ("save_file", ),
            "tensorrt_edgellm.chat_template":
        ("process_chat_template", "write_fallback_processed_chat_template"),
            "tensorrt_edgellm.vocab_reduction.onnx_export":
        ("copy_reduced_vocab_artifacts", ),
    }.items():
        dependency = ModuleType(name)
        for function in functions:
            setattr(dependency, function, lambda *_args: None)
        monkeypatch.setitem(sys.modules, name, dependency)
    return module


def _write_runtime_config(checkpoint_utils, tmp_path, root, generation):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for filename, config in (("config.json", root), ("generation_config.json",
                                                     generation)):
        if config is not None:
            (checkpoint / filename).write_text(json.dumps(config))
    model = SimpleNamespace(config=SimpleNamespace(is_eagle3_draft=True))
    output = tmp_path / "export"
    checkpoint_utils.write_runtime_artifacts(model, str(checkpoint),
                                             str(output))
    return json.loads((output / "config.json").read_text())


@pytest.mark.parametrize(
    "root,generation,expected",
    [
        ({
            "eos_token_id": [1, 106]
        }, {
            "eos_token_id": [1, 106, 50]
        }, [1, 106, 50]),
        ({
            "eos_token_id": [1, 106]
        }, {
            "eos_token_id": 50
        }, [50]),
        ({
            "eos_token_id": 1
        }, {
            "eos_token_id": 0
        }, [0]),
        ({
            "eos_token_id": 1
        }, {
            "eos_token_id": [50, 106]
        }, [50, 106]),
        ({
            "eos_token_id": [1, 106]
        }, None, [1, 106]),
        ({
            "eos_token_id": 1
        }, None, [1]),
        ({
            "eos_token_id": 0
        }, None, [0]),
        ({
            "eos_token_id": [1, 106]
        }, {}, [1, 106]),
        ({
            "eos_token_id": [1, 106]
        }, {
            "eos_token_id": None
        }, [1, 106]),
        ({}, {
            "eos_token_id": [1, 106, 50]
        }, [1, 106, 50]),
        (None, {
            "eos_token_id": [1, 106, 50]
        }, [1, 106, 50]),
        (None, {
            "eos_token_id": 50
        }, [50]),
        ({
            "eos_token_id": None
        }, {
            "eos_token_id": 50
        }, [50]),
        ({
            "eos_token_id": True
        }, {
            "eos_token_id": 50
        }, [50]),
        ({}, {}, None),
        (None, None, None),
    ],
)
def test_writer_resolves_generation_eos_before_model_eos(
        checkpoint_utils, tmp_path, root, generation, expected):
    result = _write_runtime_config(checkpoint_utils, tmp_path, root,
                                   generation)
    if expected is None:
        assert "eos_token_id" not in result
    else:
        assert result["eos_token_id"] == expected


@pytest.mark.parametrize("invalid", [
    True, False, -1, 1.5, "50", [], {}, [1, True], [1, -1], [1, "50"],
    [1, 50.0], [1, None], [[50]]
])
def test_invalid_generation_eos_fails_without_fallback(checkpoint_utils,
                                                       tmp_path, invalid):
    with pytest.raises(
            ValueError,
            match=r"Invalid eos_token_id in generation_config\.json"):
        _write_runtime_config(checkpoint_utils, tmp_path,
                              {"eos_token_id": [1, 106]},
                              {"eos_token_id": invalid})
    assert not (tmp_path / "export" / "config.json").exists()


@pytest.mark.parametrize("invalid", [
    True, False, -1, 1.5, "50", [], {}, [1, True], [1, -1], [1, "50"],
    [1, 50.0]
])
def test_invalid_model_eos_fails_without_exporting(checkpoint_utils, tmp_path,
                                                   invalid):
    with pytest.raises(ValueError,
                       match=r"Invalid eos_token_id in config\.json"):
        _write_runtime_config(checkpoint_utils, tmp_path,
                              {"eos_token_id": invalid}, None)
    assert not (tmp_path / "export" / "config.json").exists()


@pytest.mark.parametrize("contents", ["null", "[]", "true", "50", '"text"'])
def test_nonobject_model_config_fails_explicitly(checkpoint_utils, tmp_path,
                                                 contents):
    (tmp_path / "config.json").write_text(contents)
    model = SimpleNamespace(config=SimpleNamespace(is_eagle3_draft=True))
    output = tmp_path / "export"
    with pytest.raises(ValueError,
                       match=r"config\.json must contain an object"):
        checkpoint_utils.write_runtime_artifacts(model, str(tmp_path),
                                                 str(output))
    assert not (output / "config.json").exists()


def test_writer_preserves_vision_config(checkpoint_utils, tmp_path):
    vision = {"model_type": "gemma4_vision", "num_position_embeddings": 4096}
    result = _write_runtime_config(checkpoint_utils, tmp_path, {
        "eos_token_id": [1, 106],
        "vision_config": vision
    }, {"eos_token_id": [1, 106, 50]})
    assert result["vision_config"] == vision
    assert result["eos_token_id"] == [1, 106, 50]


@pytest.mark.parametrize("contents", ["{", "null", "[]"])
def test_malformed_generation_config_fails_explicitly(checkpoint_utils,
                                                      tmp_path, contents):
    (tmp_path / "generation_config.json").write_text(contents)
    with pytest.raises(ValueError):
        checkpoint_utils._runtime_eos_token_ids(str(tmp_path),
                                                {"eos_token_id": 1})


def test_empty_model_dir_does_not_read_working_directory(
        checkpoint_utils, tmp_path, monkeypatch):
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": 50}')
    monkeypatch.chdir(tmp_path)
    assert checkpoint_utils._runtime_eos_token_ids("",
                                                   {"eos_token_id": 1}) == [1]
