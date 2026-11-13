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
import logging
import os
import re
import struct
import time
from concurrent.futures import ThreadPoolExecutor
import tempfile
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
    """Write through a uniquely named, exclusively created temporary file."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".edgellm-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _identity_files(engine_dir: str) -> Dict[str, int]:
    """``{posix relpath: size}`` of the files an identity covers."""
    if os.path.islink(os.path.normpath(engine_dir)):
        raise ValueError(f"engine directory {engine_dir} is a symlink; the "
                         "base identity must describe real files")
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
            if rel_root == "." and (name == BASE_IDENTITY_FILE or
                                    (name.startswith(".edgellm-")
                                     and name.endswith(".tmp"))):
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


# A file modified this recently is hashed but not stamped: on filesystems with
# coarse timestamps a same-size rewrite within one tick could keep the stamp.
_STAMP_SETTLE_NS = 2_000_000_000


def _stamp_of(st: os.stat_result) -> List[int]:
    """Identity of a file's inode and contents as the filesystem sees it;
    replacing or rewriting the file changes it (ctime cannot be set)."""
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def _file_stamp(path: str) -> List[int]:
    return _stamp_of(os.stat(path))


def _hash_and_stamp(path: str) -> Tuple[str, List[int]]:
    """SHA-256 of the file and the stamp of the exact inode that was read;
    refuses a file that changed or was replaced while it was hashed."""
    with open(path, "rb") as stream:
        before = _stamp_of(os.fstat(stream.fileno()))
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
        after = _stamp_of(os.fstat(stream.fileno()))
    if before != after or _file_stamp(path) != after:
        raise ValueError(f"base identity mismatch: {os.path.basename(path)} "
                         "changed while it was being verified")
    return digest.hexdigest(), after


def _trusted_cache_file(path: str) -> bool:
    """Only a stamp cache that other users cannot replace may skip hashing:
    a regular file and directory owned by this user and not group- or
    world-writable, neither of them a symlink."""
    for candidate in (path, os.path.dirname(os.path.abspath(path))):
        try:
            st = os.lstat(candidate)
        except OSError:
            return False
        if (os.path.islink(candidate) or st.st_uid != os.geteuid()
                or st.st_mode & 0o022):
            return False
    return os.path.isfile(path)


def _load_stamps(stamp_path: Optional[str], digest: str) -> Dict[str, list]:
    if not stamp_path or not os.path.isfile(stamp_path):
        return {}
    if not _trusted_cache_file(stamp_path):
        logger.warning("Ignoring base identity stamp cache %s: it is not a "
                       "private file of this user", stamp_path)
        return {}
    try:
        with open(stamp_path) as stream:
            cached = json.load(stream)
    except (OSError, ValueError):
        return {}
    if not isinstance(cached, dict) or cached.get("digest") != digest:
        return {}
    stamps = cached.get("stamps")
    return stamps if isinstance(stamps, dict) else {}


def read_base_identity(engine_dir: str,
                       verify: str = "full",
                       stamp_path: Optional[str] = None) -> dict:
    """Read the identity file and check the tree against it.

    ``verify="full"`` rehashes every file; ``"sizes"`` only checks the file set
    and sizes, which a same-size engine rebuild passes. With ``stamp_path``,
    full verification skips files whose inode stamp (device, inode, size,
    mtime, ctime) matches the one recorded after their last successful hash
    against this identity, and records the stamps after it succeeds.

    The returned record carries ``verification``: mode, files hashed, files
    trusted from stamps, and seconds taken.
    """
    started = time.monotonic()
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
    if not all(
            isinstance(entry, dict) and isinstance(entry.get("path"), str)
            and isinstance(entry.get("size"), int)
            and isinstance(entry.get("sha256"), str) for entry in files):
        raise ValueError(f"{path}: malformed file entry")
    paths = [entry["path"] for entry in files]
    if paths != sorted(set(paths)):
        raise ValueError(f"{path}: file entries must be unique and sorted")
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
    hashed = trusted = 0
    if verify == "full":
        cached = _load_stamps(stamp_path, record["digest"])
        stamps = {rel: _file_stamp(os.path.join(engine_dir, rel)) for rel in paths}
        pending = [rel for rel in paths if cached.get(rel) != stamps[rel]]
        trusted = len(paths) - len(pending)
        with ThreadPoolExecutor(max_workers=max(1, min(4, len(pending)))) as pool:
            results = dict(
                zip(pending,
                    pool.map(
                        lambda rel: _hash_and_stamp(
                            os.path.join(engine_dir, rel)), pending)))
        for rel in pending:
            digest, stamps[rel] = results[rel]
            if digest != recorded[rel]["sha256"]:
                raise ValueError(f"base identity mismatch: {rel} sha256 differs")
        for rel in paths:
            if rel not in results and _file_stamp(os.path.join(
                    engine_dir, rel)) != stamps[rel]:
                raise ValueError(f"base identity mismatch: {rel} changed "
                                 "while the identity was being verified")
        hashed = len(pending)
        if stamp_path:
            settle = time.time_ns() - _STAMP_SETTLE_NS
            settled = {
                rel: stamp
                for rel, stamp in stamps.items()
                if stamp[3] < settle and stamp[4] < settle
            }
            try:
                directory = os.path.dirname(os.path.abspath(stamp_path))
                os.makedirs(directory, mode=0o700, exist_ok=True)
                _write_json_atomic(stamp_path, {
                    "digest": record["digest"],
                    "stamps": settled
                })
            except OSError as error:
                # Verification succeeded; only the next start's shortcut is lost.
                logger.warning("Could not record base identity stamps in %s: "
                               "%s", stamp_path, error)
    record["verification"] = {
        "mode": verify,
        "hashed_files": hashed,
        "trusted_files": trusted,
        "seconds": round(time.monotonic() - started, 3),
    }
    return record


_DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "I8": 1, "U8": 1}


def read_safetensors_tensor_names(path: str) -> Dict[str, Tuple[str, List[int]]]:
    """``{name: (dtype, shape)}`` from a safetensors header, without reading the
    payload; refuses headers whose byte ranges the file cannot hold."""
    file_size = os.path.getsize(path)
    with open(path, "rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= min(_MAX_HEADER, file_size - 8):
            raise ValueError(f"{path}: invalid safetensors header length")
        blob = stream.read(length)
    if len(blob) != length:
        raise ValueError(f"{path}: truncated safetensors header")
    try:
        header = json.loads(blob)
    except ValueError as error:
        raise ValueError(f"{path}: invalid safetensors header: {error}")
    if not isinstance(header, dict):
        raise ValueError(f"{path}: invalid safetensors header")
    payload = file_size - 8 - length
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype, shape = info["dtype"], [int(d) for d in info["shape"]]
            begin, end = (int(v) for v in info["data_offsets"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{path}: malformed header entry {name!r}")
        count = 1
        for dim in shape:
            count *= dim
        if (dtype not in _DTYPE_BYTES or not 0 <= begin <= end <= payload
                or end - begin != count * _DTYPE_BYTES[dtype]):
            raise ValueError(f"{path}: tensor {name!r} does not fit the file")
        tensors[name] = (dtype, shape)
    return tensors


def _adapter_targets(weights_path: str) -> Tuple[List[str], set]:
    """Module stems of a processed adapter and the set of ranks its factors carry."""
    tensors = read_safetensors_tensor_names(weights_path)
    scales = {name for name in tensors if name.endswith("_scale")}
    for scale in scales:
        factor = scale[:-len("_scale")]
        if factor not in tensors or tensors[factor][0] != "I8":
            raise ValueError(f"{weights_path}: scale {scale!r} has no INT8 "
                             "factor")
    for name, (dtype, _) in tensors.items():
        if dtype == "I8" and name + "_scale" not in tensors:
            raise ValueError(f"{weights_path}: INT8 factor {name!r} has no "
                             "scale")
    factors: Dict[str, set] = {}
    ranks = set()
    for name, (_, shape) in tensors.items():
        if name in scales:
            continue
        suffix = next((s for s in _FACTOR_SUFFIXES if name.endswith(s)), None)
        if suffix is None:
            raise ValueError(f"{weights_path}: tensor {name!r} is not a "
                             "lora_A/lora_B weight")
        if len(shape) != 2:
            raise ValueError(f"{weights_path}: tensor {name!r} is not 2-D")
        stem = name[:-len(suffix)]
        factors.setdefault(stem, set()).add(suffix)
        ranks.add(shape[1] if suffix == ".lora_A.weight" else shape[0])
    unpaired = sorted(stem for stem, found in factors.items()
                      if found != set(_FACTOR_SUFFIXES))
    if unpaired:
        raise ValueError(f"{weights_path}: modules without both factors: "
                         f"{unpaired[:5]}")
    return sorted(factors), ranks


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
    targets, ranks = _adapter_targets(weights_path)
    if targets != sorted(record.get("bound_modules") or []):
        raise ValueError(
            f"{adapter_dir}: the weights file and its processing record "
            "disagree on the bound modules; reprocess the adapter")
    rank = record.get("rank")
    if ranks != {rank}:
        raise ValueError(f"{adapter_dir}: factor ranks {sorted(ranks)} differ "
                         f"from the recorded rank {rank}")
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


def _manifest_fields(manifest, where: str) -> None:
    """Refuse a manifest whose fields have the wrong type or are missing."""
    if not isinstance(manifest, dict) or manifest.get("schema") != ADAPTER_MANIFEST_SCHEMA:
        raise ValueError(f"{where}: unsupported manifest schema")
    weights = manifest.get("weights")
    checks = {
        "version": isinstance(manifest.get("version"), str)
        and bool(_VERSION_RE.fullmatch(manifest["version"])),
        "base_identity": isinstance(manifest.get("base_identity"), str),
        "weights": isinstance(weights, dict)
        and isinstance(weights.get("file"), str)
        and isinstance(weights.get("size"), int)
        and isinstance(weights.get("sha256"), str),
        "rank": isinstance(manifest.get("rank"), int)
        and not isinstance(manifest.get("rank"), bool),
        "targets": isinstance(manifest.get("targets"), list)
        and all(isinstance(t, str) for t in manifest["targets"]),
    }
    bad = sorted(field for field, ok in checks.items() if not ok)
    if bad:
        raise ValueError(f"{where}: manifest fields missing or malformed: {bad}")


def verify_adapter(weights_path: str, base_identity: dict,
                   max_lora_rank: int) -> dict:
    """Check the exact weights file the runtime will load against its manifest
    (in the same directory) and the deployed base."""
    adapter_dir = os.path.dirname(os.path.abspath(weights_path))
    path = os.path.join(adapter_dir, ADAPTER_MANIFEST_FILE)
    if not os.path.isfile(path):
        raise ValueError(
            f"{adapter_dir}: no {ADAPTER_MANIFEST_FILE}; run "
            "tensorrt-edgellm-lora-package adapter --engine-dir <llm dir> "
            f"--adapter-dir {adapter_dir} --version <version>")
    with open(path) as stream:
        try:
            manifest = json.load(stream)
        except ValueError as error:
            raise ValueError(f"{path}: invalid JSON: {error}")
    _manifest_fields(manifest, adapter_dir)
    if manifest.get("processing_schema_version") not in SUPPORTED_PROCESSING_SCHEMAS:
        raise ValueError(f"{adapter_dir}: unsupported processing_schema_version "
                         f"{manifest.get('processing_schema_version')!r}")
    if manifest["base_identity"] != base_identity["digest"]:
        raise ValueError(
            f"{adapter_dir}: packaged for base "
            f"{manifest['base_identity'][:12]}, deployed base is "
            f"{base_identity['digest'][:12]}")
    weights = manifest["weights"]
    if weights["file"] != os.path.basename(weights_path):
        raise ValueError(
            f"{adapter_dir}: manifest describes {weights['file']!r}, the "
            f"runtime loads {os.path.basename(weights_path)!r}")
    if not os.path.isfile(weights_path):
        raise ValueError(f"{weights_path}: weights file missing")
    if os.path.getsize(weights_path) != weights["size"]:
        raise ValueError(f"{adapter_dir}: weights size differs from manifest")
    if _sha256(weights_path) != weights["sha256"]:
        raise ValueError(f"{adapter_dir}: weights sha256 differs from manifest")
    targets, ranks = _adapter_targets(weights_path)
    if targets != manifest["targets"]:
        raise ValueError(f"{adapter_dir}: weights modules differ from the "
                         "manifest targets")
    rank = manifest["rank"]
    if ranks != {rank}:
        raise ValueError(f"{adapter_dir}: factor ranks {sorted(ranks)} differ "
                         f"from the manifest rank {rank}")
    if rank > max_lora_rank:
        raise ValueError(f"{adapter_dir}: rank {rank} exceeds the engine's "
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
