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
"""ldd-style dependency report built from `readelf -d`, without loading anything.

For every ELF file under the given trees, lists SONAME, RUNPATH/RPATH and
NEEDED entries, resolves each NEEDED name the way the dynamic loader would
(RUNPATH, then the ld.so cache, then the CUDA target lib dir that the
toolkit packages do not register with ldconfig) and names the dpkg package
owning the resolved file. The ELF objects are never executed or dlopen'd,
so this is safe to run in a container without a GPU driver.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path


def readelf_dynamic(path: Path) -> dict:
    out = subprocess.run(["readelf", "-d", str(path)],
                         capture_output=True,
                         text=True).stdout
    info = {"needed": [], "soname": None, "runpath": [], "rpath": []}
    for line in out.splitlines():
        if "(NEEDED)" in line:
            info["needed"].append(line.split("[", 1)[1].rstrip("]"))
        elif "(SONAME)" in line:
            info["soname"] = line.split("[", 1)[1].rstrip("]")
        elif "(RUNPATH)" in line:
            info["runpath"] = line.split("[", 1)[1].rstrip("]").split(":")
        elif "(RPATH)" in line:
            info["rpath"] = line.split("[", 1)[1].rstrip("]").split(":")
    return info


def ldconfig_cache() -> dict:
    cache = {}
    out = subprocess.run(["ldconfig", "-p"], capture_output=True,
                         text=True).stdout
    for line in out.splitlines()[1:]:
        name, _, target = line.strip().partition(" (")
        if "=> " in target:
            cache.setdefault(name, target.split("=> ", 1)[1])
    return cache


def dpkg_owner(path: str) -> str:
    real = os.path.realpath(path)
    for candidate in (path, real):
        r = subprocess.run(["dpkg", "-S", candidate],
                           capture_output=True,
                           text=True)
        if r.returncode == 0:
            return r.stdout.split(":", 1)[0].strip()
    return "(no package)"


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trees", nargs="+")
    ap.add_argument("--cuda-lib-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    cache = ldconfig_cache()
    report = []
    for tree in args.trees:
        for path in sorted(Path(tree).rglob("*")):
            if path.is_symlink() or not path.is_file() or not is_elf(path):
                continue
            dyn = readelf_dynamic(path)
            resolved = []
            for name in dyn["needed"]:
                found = None
                for d in dyn["runpath"] + dyn["rpath"]:
                    if (Path(d) / name).exists():
                        found = str(Path(d) / name)
                        break
                if not found and name in cache:
                    found = cache[name]
                if not found and (Path(args.cuda_lib_dir) / name).exists():
                    found = str(Path(args.cuda_lib_dir) / name)
                if not found and (Path(args.cuda_lib_dir) / "stubs" /
                                  name).exists():
                    found = "(stub only: " + str(
                        Path(args.cuda_lib_dir) / "stubs" / name) + ")"
                entry = {"needed": name, "resolved": found}
                if found and not found.startswith("("):
                    entry["package"] = dpkg_owner(found)
                    entry["realpath"] = os.path.realpath(found)
                resolved.append(entry)
            report.append({
                "file": str(path),
                "soname": dyn["soname"],
                "runpath": dyn["runpath"],
                "rpath": dyn["rpath"],
                "needed": resolved,
            })

    with open(args.json, "w") as f:
        json.dump(report, f, indent=2)
    with open(args.text, "w") as f:
        for item in report:
            f.write(f"{item['file']}\n")
            if item["soname"]:
                f.write(f"  SONAME  {item['soname']}\n")
            if item["runpath"] or item["rpath"]:
                f.write(
                    f"  RUNPATH {':'.join(item['runpath'] + item['rpath'])}\n")
            for dep in item["needed"]:
                target = dep["resolved"] or "not found in container"
                pkg = f"  [{dep['package']}]" if "package" in dep else ""
                f.write(f"  NEEDED  {dep['needed']:<32} => {target}{pkg}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
