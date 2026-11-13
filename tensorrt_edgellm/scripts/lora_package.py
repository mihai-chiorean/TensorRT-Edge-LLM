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
"""
Package processed LoRA adapters against a deployed engine directory.

Usage:
    tensorrt-edgellm-lora-package base-identity --engine-dir <bundle>/llm
    tensorrt-edgellm-lora-package adapter --engine-dir <bundle>/llm \\
        --adapter-dir <bundle>/llm/lora_weights/tool --version 2026.11.12-1
    tensorrt-edgellm-lora-package verify --engine-dir <bundle>/llm [--full]
"""

import argparse
import os
import sys

from tensorrt_edgellm import lora_package


def _base_identity(args) -> None:
    record = lora_package.write_base_identity(args.engine_dir)
    print(f"base identity {record['digest']} ({len(record['files'])} files)")


def _adapter(args) -> None:
    manifest = lora_package.write_adapter_manifest(args.adapter_dir,
                                                   args.engine_dir,
                                                   args.version)
    print(f"{os.path.basename(os.path.normpath(args.adapter_dir))} "
          f"version {manifest['version']}: weights "
          f"{manifest['weights']['sha256'][:12]}, base "
          f"{manifest['base_identity'][:12]}")


def _verify(args) -> None:
    base = lora_package.read_base_identity(args.engine_dir,
                                           "full" if args.full else "sizes")
    max_rank = lora_package.max_lora_rank_of(args.engine_dir) or 0
    print(f"base identity {base['digest'][:12]} ok")
    for adapter_dir in lora_package.adapter_dirs(args.engine_dir):
        manifest = lora_package.verify_adapter(adapter_dir, base, max_rank)
        print(f"{os.path.basename(adapter_dir)} version {manifest['version']} ok")


def main(argv=None, *, prog="tensorrt-edgellm-lora-package") -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Bind processed LoRA adapters to a deployed engine directory")
    sub = parser.add_subparsers(dest="command", required=True)
    base = sub.add_parser("base-identity",
                          help="write edgellm_base_identity.json")
    base.add_argument("--engine-dir", required=True)
    base.set_defaults(func=_base_identity)
    adapter = sub.add_parser("adapter",
                             help="write an adapter's edgellm_adapter_manifest.json")
    adapter.add_argument("--engine-dir", required=True)
    adapter.add_argument("--adapter-dir", required=True)
    adapter.add_argument("--version", required=True)
    adapter.set_defaults(func=_adapter)
    verify = sub.add_parser(
        "verify", help="check the base identity and every adapter under lora_weights/")
    verify.add_argument("--engine-dir", required=True)
    verify.add_argument("--full",
                        action="store_true",
                        help="rehash every base file instead of checking sizes")
    verify.set_defaults(func=_verify)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
