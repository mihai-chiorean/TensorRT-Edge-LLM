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
"""
Command-line script for processing LoRA weights.

This script provides a command-line interface for processing LoRA weights according
to specified requirements including tensor name processing, shape corrections, and
format conversions.

Usage:
    tensorrt-edgellm-process-lora --input_dir /path/to/adapter --output_dir /path/to/output \
        [--onnx_dir /path/to/onnx/llm] [--max_lora_rank 16]
"""

import argparse
import sys
import traceback

from tensorrt_edgellm.lora.lora import process_lora_weights_and_save


def main() -> None:
    """
    Main function that parses command line arguments and processes LoRA weights.

    This function sets up argument parsing for the LoRA weight processing script
    and calls the process_lora_weights_and_save function with the provided parameters.
    """
    parser = argparse.ArgumentParser(
        description="Process LoRA weights according to specifications")
    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="Directory containing input adapter files")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Directory where processed files will be saved")
    parser.add_argument(
        "--onnx_dir",
        type=str,
        default=None,
        help="Directory containing lora_model.onnx; validates the adapter "
        "against the engine's LoRA bindings and drops unused tensors")
    parser.add_argument(
        "--max_lora_rank",
        type=int,
        default=None,
        help="Engine --maxLoraRank; the adapter rank must not exceed it")

    args = parser.parse_args()

    try:
        report = process_lora_weights_and_save(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            onnx_dir=args.onnx_dir,
            max_lora_rank=args.max_lora_rank)

        print(f"Wrote {report.output_path}: {len(report.bound)} modules, "
              f"rank {report.rank}, scale {report.lora_scale:.4f}")
        if report.unused:
            print(f"Dropped {len(report.unused)} adapter module(s) the "
                  "engine does not have:")
            for stem in report.unused:
                print(f"  {stem}")
        if report.unbound_untargeted:
            print(f"{len(report.unbound_untargeted)} engine binding(s) "
                  "outside the adapter's targets stay at rank-1 zeros:")
            for stem in report.unbound_untargeted:
                print(f"  {stem}")
        print("LoRA weight processing completed successfully!")

    except Exception as e:
        print(f"Error during LoRA weight processing: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
