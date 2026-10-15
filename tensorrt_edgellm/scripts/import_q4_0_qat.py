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
"""CLI: import a Q4_0-style QAT checkpoint on its own INT4 grid (default group 32)."""

import argparse
import logging
import sys

from tensorrt_edgellm.checkpoint.q4_0_grid import convert_q4_0_checkpoint

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(
        description="Write a half-precision QAT checkpoint whose weights sit on "
        "a symmetric INT4 group grid as a W4A16 checkpoint for "
        "tensorrt-edgellm-export, keeping the trained codes and scales.")
    parser.add_argument(
        "--model_dir",
        required=True,
        help="HF checkpoint directory (single model.safetensors)")
    parser.add_argument("--output_dir",
                        help="Output checkpoint directory; omit to only "
                        "report how well the weights fit the grid")
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--linear_prefix",
                        default="model.language_model.",
                        help="Tensor name prefix of the decoder linears")
    parser.add_argument("--keep_fp16",
                        nargs="*",
                        default=[],
                        help="Module-name globs left in fp16, e.g. "
                        "'*.mlp.down_proj'")
    parser.add_argument(
        "--int4_lm_head",
        action="store_true",
        help="Also write the tied embedding as an INT4 lm_head")
    args = parser.parse_args()

    try:
        report = convert_q4_0_checkpoint(model_dir=args.model_dir,
                                         output_dir=args.output_dir,
                                         group_size=args.group_size,
                                         linear_prefix=args.linear_prefix,
                                         keep_fp16=args.keep_fp16,
                                         int4_lm_head=args.int4_lm_head)
    except Exception:
        logger.exception("Error during Q4_0 QAT import")
        sys.exit(1)
    stats = report["int4"]
    if stats.groups and stats.groups_on_grid < stats.groups:
        logger.warning(
            "%d of %d groups are not on a group-%d grid; the output is a "
            "round-to-nearest quantization of those groups",
            stats.groups - stats.groups_on_grid, stats.groups, args.group_size)


if __name__ == "__main__":
    main()
