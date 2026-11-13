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
"""Adapter packages bound to the deployed engine directory's identity."""

import json
import os

import numpy as np
import pytest
from safetensors.numpy import save_file

from tensorrt_edgellm import lora_package
from tensorrt_edgellm.scripts.lora_package import main

_STEMS = ["model.layers.0.mlp.up_proj", "model.layers.0.self_attn.q_proj"]


def _engine(tmp_path):
    llm = tmp_path / "llm"
    llm.mkdir()
    (llm / "llm.engine").write_bytes(b"engine-bytes")
    (llm / "tokenizer.json").write_text('{"model": "x"}')
    (llm / "config.json").write_text(
        json.dumps({"builder_config": {
            "max_lora_rank": 16
        }}))
    return llm


def _processed_adapter(engine_dir, name="tool", rank=4, record=None):
    directory = engine_dir / "lora_weights" / name
    directory.mkdir(parents=True)
    tensors = {}
    for stem in _STEMS:
        tensors[f"{stem}.lora_A.weight"] = np.ones((8, rank), np.float16)
        tensors[f"{stem}.lora_B.weight"] = np.ones((rank, 8), np.float16)
    save_file(tensors, str(directory / lora_package.WEIGHTS_FILE))
    processing = {
        "processing_schema_version": 1,
        "rank": rank,
        "lora_scale": 2.0,
        "bound_modules": list(_STEMS),
        "adapter_model_sha256": "ab" * 32,
    }
    processing.update(record or {})
    processing = {k: v for k, v in processing.items() if v is not None}
    (directory / "config.json").write_text(
        json.dumps({"edgellm_lora_processing": processing}))
    return directory


def test_base_identity_is_stable_and_ignores_adapters(tmp_path):
    llm = _engine(tmp_path)
    first = lora_package.compute_base_identity(str(llm))
    _processed_adapter(llm)

    assert lora_package.compute_base_identity(str(llm)) == first
    assert [entry["path"] for entry in first["files"]
            ] == ["config.json", "llm.engine", "tokenizer.json"]


def test_base_identity_changes_with_one_tokenizer_byte(tmp_path):
    llm = _engine(tmp_path)
    before = lora_package.compute_base_identity(str(llm))["digest"]
    (llm / "tokenizer.json").write_text('{"model": "y"}')

    assert lora_package.compute_base_identity(str(llm))["digest"] != before


def test_base_identity_refuses_symlinks(tmp_path):
    llm = _engine(tmp_path)
    os.symlink(llm / "llm.engine", llm / "alias.engine")

    with pytest.raises(ValueError, match="symlink"):
        lora_package.compute_base_identity(str(llm))


def test_read_base_identity_checks_sizes_then_hashes(tmp_path):
    llm = _engine(tmp_path)
    lora_package.write_base_identity(str(llm))
    lora_package.read_base_identity(str(llm), verify="sizes")

    (llm / "llm.engine").write_bytes(b"ENGINE-bytes")  # same size
    lora_package.read_base_identity(str(llm), verify="sizes")
    with pytest.raises(ValueError, match="llm.engine"):
        lora_package.read_base_identity(str(llm), verify="full")

    (llm / "extra.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="extra.bin is unexpected"):
        lora_package.read_base_identity(str(llm), verify="sizes")


def test_missing_base_identity_names_the_command(tmp_path):
    llm = _engine(tmp_path)
    with pytest.raises(FileNotFoundError, match="base-identity"):
        lora_package.read_base_identity(str(llm))


def test_manifest_binds_weights_to_the_base(tmp_path):
    llm = _engine(tmp_path)
    base = lora_package.write_base_identity(str(llm))
    adapter = _processed_adapter(llm)

    manifest = lora_package.write_adapter_manifest(str(adapter), str(llm),
                                                   "2026.11.12-1")

    assert manifest["base_identity"] == base["digest"]
    assert manifest["targets"] == sorted(_STEMS)
    assert manifest["rank"] == 4
    assert manifest["version"] == "2026.11.12-1"
    assert manifest["weights"]["size"] == os.path.getsize(
        adapter / lora_package.WEIGHTS_FILE)
    on_disk = json.loads((adapter / lora_package.ADAPTER_MANIFEST_FILE).read_text())
    assert on_disk == manifest


def test_manifest_refuses_a_record_that_disagrees_with_the_weights(tmp_path):
    llm = _engine(tmp_path)
    lora_package.write_base_identity(str(llm))
    adapter = _processed_adapter(llm, record={"bound_modules": [_STEMS[0]]})

    with pytest.raises(ValueError, match="bound modules"):
        lora_package.write_adapter_manifest(str(adapter), str(llm), "v1")


def test_manifest_refuses_an_old_processing_schema(tmp_path):
    llm = _engine(tmp_path)
    lora_package.write_base_identity(str(llm))
    adapter = _processed_adapter(llm,
                                 record={"processing_schema_version": None})

    with pytest.raises(ValueError, match="processing_schema_version"):
        lora_package.write_adapter_manifest(str(adapter), str(llm), "v1")


def _packaged(tmp_path):
    llm = _engine(tmp_path)
    base = lora_package.write_base_identity(str(llm))
    adapter = _processed_adapter(llm)
    lora_package.write_adapter_manifest(str(adapter), str(llm), "v1")
    return llm, base, adapter


def test_verify_adapter_accepts_a_matching_package(tmp_path):
    _, base, adapter = _packaged(tmp_path)
    manifest = lora_package.verify_adapter(
        str(adapter / lora_package.WEIGHTS_FILE), base, 16)
    assert manifest["version"] == "v1"


def test_verify_adapter_refuses_another_base(tmp_path):
    _, base, adapter = _packaged(tmp_path)
    with pytest.raises(ValueError, match="deployed base"):
        lora_package.verify_adapter(str(adapter), dict(base, digest="f" * 64),
                                    16)


def test_verify_adapter_refuses_changed_weights(tmp_path):
    _, base, adapter = _packaged(tmp_path)
    weights = adapter / lora_package.WEIGHTS_FILE
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF
    weights.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="sha256"):
        lora_package.verify_adapter(str(adapter), base, 16)


def test_verify_adapter_refuses_rank_above_the_engine(tmp_path):
    _, base, adapter = _packaged(tmp_path)
    with pytest.raises(ValueError, match="max_lora_rank"):
        lora_package.verify_adapter(str(adapter), base, 2)


def test_cli_packages_and_verifies(tmp_path, capsys):
    llm = _engine(tmp_path)
    adapter = _processed_adapter(llm)

    main(["base-identity", "--engine-dir", str(llm)])
    main([
        "adapter", "--engine-dir",
        str(llm), "--adapter-dir",
        str(adapter), "--version", "v1"
    ])
    main(["verify", "--engine-dir", str(llm), "--full"])
    assert "tool version v1 ok" in capsys.readouterr().out

    weights = adapter / lora_package.WEIGHTS_FILE
    weights.write_bytes(weights.read_bytes()[:-1] + b"\x00")
    with pytest.raises(SystemExit) as exited:
        main(["verify", "--engine-dir", str(llm)])
    assert exited.value.code == 1
