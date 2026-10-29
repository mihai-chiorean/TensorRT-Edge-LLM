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
"""Hub ids used by the built-in text calibration datasets (no network)."""

import sys
import types

import pytest

from tensorrt_edgellm.quantization.datasets import builtin


class _Value:

    def __init__(self, text):
        self._text = text

    def as_py(self):
        return self._text


class _Table:

    def column(self, name):
        assert name == "text"
        return [_Value("zh-1"), _Value("zh-2")]


@pytest.fixture
def recorded_ids(monkeypatch):
    ids = []

    def fake_load_hf_split(hf_id, **kwargs):
        ids.append(hf_id)
        return [{"article": "en-1"}, {"article": "en-2"}]

    monkeypatch.setattr(builtin, "load_hf_split", fake_load_hf_split)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda *args, **kwargs: "shard.parquet"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    fake_parquet = types.ModuleType("pyarrow.parquet")
    fake_parquet.read_table = lambda path, columns: _Table()
    fake_pyarrow = types.ModuleType("pyarrow")
    fake_pyarrow.parquet = fake_parquet
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)
    return ids


@pytest.mark.parametrize("generator",
                         [builtin.cnn_dailymail, builtin.zh_en_mixed])
def test_cnn_dailymail_uses_namespaced_hub_id(recorded_ids, generator):
    texts = list(generator())

    assert texts
    assert recorded_ids == ["abisee/cnn_dailymail"]


def test_zh_en_mixed_alternates_sources(recorded_ids):
    assert list(builtin.zh_en_mixed()) == ["zh-1", "en-1", "zh-2", "en-2"]
