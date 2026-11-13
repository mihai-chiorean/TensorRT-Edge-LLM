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
"""Adapter packages bound to the identity of the deployed engine directory.

An engine directory (the bundle's ``llm/``) carries ``edgellm_base_identity.json``:
the size and SHA-256 of every file that makes up the effective base (engines,
embedding tables, tokenizer, chat template, configs), excluding ``lora_weights/``.
Each processed adapter directory carries ``edgellm_adapter_manifest.json`` naming
that identity's digest and its own weights' hash, so the server can refuse an
adapter processed for a different base. Stdlib only: the server imports it.
"""

import datetime
import hashlib
import json
import os
import re
import struct
from typing import Dict, List, Optional, Tuple

BASE_IDENTITY_FILE = "edgellm_base_identity.json"
ADAPTER_MANIFEST_FILE = "edgellm_adapter_manifest.json"
BASE_IDENTITY_SCHEMA = 1
ADAPTER_MANIFEST_SCHEMA = 1
SUPPORTED_PROCESSING_SCHEMAS = frozenset({1})

WEIGHTS_FILE = "processed_adapter_model.safetensors"
LORA_WEIGHTS_DIR = "lora_weights"
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")
_FACTOR_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
_CHUNK = 16 * 1024 * 1024
_MAX_HEADER = 100_000_000


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: str, value: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")
    os.replace(tmp, path)


def _identity_files(engine_dir: str) -> Dict[str, int]:
    """``{posix relpath: size}`` of the files an identity covers."""
    if not os.path.isdir(engine_dir):
        raise FileNotFoundError(f"engine directory not found: {engine_dir}")
    files: Dict[str, int] = {}
    for root, dirs, names in os.walk(engine_dir, followlinks=False):
        rel_root = os.path.relpath(root, engine_dir)
        if rel_root == ".":
            dirs[:] = [d for d in dirs if d != LORA_WEIGHTS_DIR]
        for entry in list(dirs) + names:
            path = os.path.join(root, entry)
            if os.path.islink(path):
                raise ValueError(
                    f"symlink in engine directory: {path}; the base identity "
                    "must describe real files")
        for name in names:
            if rel_root == "." and name in (BASE_IDENTITY_FILE,
                                            BASE_IDENTITY_FILE + ".tmp"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, engine_dir).replace(os.sep, "/")
            files[rel] = os.path.getsize(path)
    if not files:
        raise ValueError(f"engine directory {engine_dir} holds no files")
    return files


def _digest(files: List[dict]) -> str:
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_base_identity(engine_dir: str) -> dict:
    """Size and SHA-256 of every file of the effective base, plus their digest."""
    files = [{
        "path": rel,
        "size": size,
        "sha256": _sha256(os.path.join(engine_dir, rel))
    } for rel, size in sorted(_identity_files(engine_dir).items())]
    return {
        "schema": BASE_IDENTITY_SCHEMA,
        "files": files,
        "digest": _digest(files)
    }


def write_base_identity(engine_dir: str) -> dict:
    record = compute_base_identity(engine_dir)
    _write_json_atomic(os.path.join(engine_dir, BASE_IDENTITY_FILE), record)
    return record


def read_base_identity(engine_dir: str, verify: str = "sizes") -> dict:
    """Read the identity file and check the tree against it.

    ``verify="sizes"`` checks the file set and sizes; ``"full"`` also rehashes
    every file.
    """
    if verify not in ("sizes", "full"):
        raise ValueError(f"verify must be 'sizes' or 'full', got {verify!r}")
    path = os.path.join(engine_dir, BASE_IDENTITY_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found; run tensorrt-edgellm-lora-package "
            f"base-identity --engine-dir {engine_dir}")
    with open(path) as stream:
        record = json.load(stream)
    if not isinstance(record, dict) or record.get("schema") != BASE_IDENTITY_SCHEMA:
        raise ValueError(f"{path}: unsupported base identity schema")
    files = record.get("files")
    if not isinstance(files, list) or _digest(files) != record.get("digest"):
        raise ValueError(f"{path}: digest does not match its file list")
    recorded = {entry["path"]: entry for entry in files}
    present = _identity_files(engine_dir)
    for rel in sorted(set(recorded) | set(present)):
        if rel not in present:
            raise ValueError(f"base identity mismatch: {rel} is missing")
        if rel not in recorded:
            raise ValueError(f"base identity mismatch: {rel} is unexpected")
        if present[rel] != recorded[rel]["size"]:
            raise ValueError(
                f"base identity mismatch: {rel} size {present[rel]} != "
                f"recorded {recorded[rel]['size']}")
        if verify == "full" and _sha256(os.path.join(
                engine_dir, rel)) != recorded[rel]["sha256"]:
            raise ValueError(f"base identity mismatch: {rel} sha256 differs")
    return record


def read_safetensors_tensor_names(path: str) -> Dict[str, Tuple[str, List[int]]]:
    """``{name: (dtype, shape)}`` from a safetensors header, without the payload."""
    with open(path, "rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= _MAX_HEADER:
            raise ValueError(f"{path}: invalid safetensors header length")
        header = json.loads(stream.read(length))
    return {
        name: (info["dtype"], list(info["shape"]))
        for name, info in header.items() if name != "__metadata__"
    }


def _adapter_targets(weights_path: str) -> Tuple[List[str], Dict[str, list]]:
    """Module stems of a processed adapter and its ``lora_A`` shapes."""
    tensors = read_safetensors_tensor_names(weights_path)
    factors: Dict[str, set] = {}
    a_shapes: Dict[str, list] = {}
    for name, (_, shape) in tensors.items():
        suffix = next((s for s in _FACTOR_SUFFIXES if name.endswith(s)), None)
        if suffix is None:
            raise ValueError(f"{weights_path}: tensor {name!r} is not a "
                             "lora_A/lora_B weight")
        stem = name[:-len(suffix)]
        factors.setdefault(stem, set()).add(suffix)
        if suffix == ".lora_A.weight":
            a_shapes[stem] = shape
    unpaired = sorted(stem for stem, found in factors.items()
                      if found != set(_FACTOR_SUFFIXES))
    if unpaired:
        raise ValueError(f"{weights_path}: modules without both factors: "
                         f"{unpaired[:5]}")
    return sorted(factors), a_shapes


def _processing_record(adapter_dir: str) -> dict:
    with open(os.path.join(adapter_dir, "config.json")) as stream:
        config = json.load(stream)
    record = config.get("edgellm_lora_processing") if isinstance(
        config, dict) else None
    if not isinstance(record, dict):
        raise ValueError(f"{adapter_dir}/config.json has no "
                         "edgellm_lora_processing record; reprocess the adapter")
    if record.get("processing_schema_version") not in SUPPORTED_PROCESSING_SCHEMAS:
        raise ValueError(
            f"{adapter_dir}: processing_schema_version "
            f"{record.get('processing_schema_version')!r} is not supported "
            f"({sorted(SUPPORTED_PROCESSING_SCHEMAS)}); reprocess the adapter "
            "with tensorrt-edgellm-process-lora")
    return record


def write_adapter_manifest(adapter_dir: str, engine_dir: str,
                           version: str) -> dict:
    """Bind a processed adapter to the engine directory's base identity."""
    if not _VERSION_RE.fullmatch(version or ""):
        raise ValueError(f"invalid adapter version {version!r}")
    weights_path = os.path.join(adapter_dir, WEIGHTS_FILE)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"{weights_path} not found")
    record = _processing_record(adapter_dir)
    base = read_base_identity(engine_dir, verify="full")
    targets, a_shapes = _adapter_targets(weights_path)
    if targets != sorted(record.get("bound_modules") or []):
        raise ValueError(
            f"{adapter_dir}: the weights file and its processing record "
            "disagree on the bound modules; reprocess the adapter")
    rank = record.get("rank")
    if any(shape[-1] != rank for shape in a_shapes.values()):
        raise ValueError(f"{adapter_dir}: lora_A rank differs from the "
                         f"recorded rank {rank}")
    manifest = {
        "schema": ADAPTER_MANIFEST_SCHEMA,
        "version": version,
        "base_identity": base["digest"],
        "weights": {
            "file": WEIGHTS_FILE,
            "size": os.path.getsize(weights_path),
            "sha256": _sha256(weights_path),
        },
        "rank": rank,
        "lora_scale": record.get("lora_scale"),
        "processing_schema_version": record["processing_schema_version"],
        "targets": targets,
        "source_adapter_sha256": record.get("adapter_model_sha256"),
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _write_json_atomic(os.path.join(adapter_dir, ADAPTER_MANIFEST_FILE),
                       manifest)
    return manifest


def verify_adapter(adapter_path: str, base_identity: dict,
                   max_lora_rank: int) -> dict:
    """Check an adapter directory (or its weights file) against the deployed base."""
    adapter_dir = (adapter_path
                   if os.path.isdir(adapter_path) else os.path.dirname(adapter_path))
    path = os.path.join(adapter_dir, ADAPTER_MANIFEST_FILE)
    if not os.path.isfile(path):
        raise ValueError(
            f"{adapter_dir}: no {ADAPTER_MANIFEST_FILE}; run "
            "tensorrt-edgellm-lora-package adapter --engine-dir <llm dir> "
            f"--adapter-dir {adapter_dir} --version <version>")
    with open(path) as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("schema") != ADAPTER_MANIFEST_SCHEMA:
        raise ValueError(f"{adapter_dir}: unsupported manifest schema")
    if manifest.get("processing_schema_version") not in SUPPORTED_PROCESSING_SCHEMAS:
        raise ValueError(f"{adapter_dir}: unsupported processing_schema_version "
                         f"{manifest.get('processing_schema_version')!r}")
    if manifest.get("base_identity") != base_identity["digest"]:
        raise ValueError(
            f"{adapter_dir}: packaged for base "
            f"{str(manifest.get('base_identity'))[:12]}, deployed base is "
            f"{base_identity['digest'][:12]}")
    weights = manifest.get("weights") or {}
    weights_path = os.path.join(adapter_dir, weights.get("file") or WEIGHTS_FILE)
    if not os.path.isfile(weights_path):
        raise ValueError(f"{adapter_dir}: weights file missing")
    if os.path.getsize(weights_path) != weights.get("size"):
        raise ValueError(f"{adapter_dir}: weights size differs from manifest")
    if _sha256(weights_path) != weights.get("sha256"):
        raise ValueError(f"{adapter_dir}: weights sha256 differs from manifest")
    targets, _ = _adapter_targets(weights_path)
    if targets != manifest.get("targets"):
        raise ValueError(f"{adapter_dir}: weights modules differ from the "
                         "manifest targets")
    rank = manifest.get("rank")
    if not isinstance(rank, int) or rank > max_lora_rank:
        raise ValueError(f"{adapter_dir}: rank {rank!r} exceeds the engine's "
                         f"max_lora_rank={max_lora_rank}")
    return manifest


def adapter_dirs(engine_dir: str) -> List[str]:
    """Visible adapter directories under ``<engine_dir>/lora_weights/``."""
    root = os.path.join(engine_dir, LORA_WEIGHTS_DIR)
    if not os.path.isdir(root):
        return []
    return [
        os.path.join(root, entry) for entry in sorted(os.listdir(root))
        if not entry.startswith(".") and os.path.isdir(os.path.join(root, entry))
    ]


def max_lora_rank_of(engine_dir: str) -> Optional[int]:
    for name in ("config.json", "base_config.json"):
        path = os.path.join(engine_dir, name)
        if os.path.isfile(path):
            with open(path) as stream:
                return int(json.load(stream).get("builder_config",
                                                 {}).get("max_lora_rank", 0))
    return None
