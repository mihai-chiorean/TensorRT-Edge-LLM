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

import pytest

from tensorrt_edgellm import config


def _base(excluded, gptq_zero_point_offset=1):
    return config.ModelConfig(
        model_type="qwen3_5_text",
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=256,
        head_dim=64,
        rms_norm_eps=1e-6,
        vocab_size=256,
        rope_theta=10_000.0,
        max_position_embeddings=4096,
        default_attention_scale=0.125,
        mtp_num_hidden_layers=1,
        quant=config.QuantConfig(
            quant_type=config.QUANT_INT4_AWQ,
            group_size=128,
            excluded=excluded,
            gptq_zero_point_offset=gptq_zero_point_offset),
    )


@pytest.mark.parametrize("excluded, expected_lm_head_type", [
    pytest.param(["mtp*"], config.QUANT_INT4_AWQ, id="quantized-lm-head"),
    pytest.param(["mtp*", "lm_head"], config.QUANT_FP16,
                 id="excluded-lm-head"),
])
def test_borrowed_lm_head_takes_base_quantization(excluded,
                                                  expected_lm_head_type):
    base = _base(excluded)
    assert not base.quant.layer_overrides
    assert config.module_quant_type("lm_head", base) == expected_lm_head_type

    draft = config.make_mtp_draft_config(base, inherit_base_lm_head=True)

    assert config.module_quant_type("lm_head", draft) == expected_lm_head_type
    if expected_lm_head_type == config.QUANT_INT4_AWQ:
        assert config.module_quant_group_size("lm_head", draft) == 128
    assert config.module_quant_type("fc", draft) == config.QUANT_FP16
    assert config.module_quant_type("layers.0.mlp.gate_proj",
                                    draft) == config.QUANT_FP16


def test_dedicated_draft_lm_head_stays_fp16():
    draft = config.make_mtp_draft_config(_base(["mtp*"]),
                                         inherit_base_lm_head=False)
    assert config.module_quant_type("lm_head", draft) == config.QUANT_FP16


def test_borrowed_lm_head_keeps_gptq_zero_point_offset():
    draft = config.make_mtp_draft_config(_base(["mtp*"], 0),
                                         inherit_base_lm_head=True)
    assert draft.quant.gptq_zero_point_offset == 0
