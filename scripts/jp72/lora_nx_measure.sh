#!/bin/bash
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
#
# LoRA/MTP bench measurement harness for an Orin NX 16 GB on JetPack 7.2.
#
# Runs scripts/jp72/lora_bench.py over a list of engines and records the
# platform state around each run (nvpmodel, jetson_clocks, nvidia-smi,
# tegrastats, a memory sampler). Everything lands in --out DIR:
#   bench.jsonl         one JSON line per (engine, adapter) record
#   platform.txt        hostname/model/serial/MemTotal/nv_tegra_release + clock state
#   tegrastats-<tag>.log  tegrastats samples for each engine run
#   mem-<tag>.csv       1 s memory samples for each engine run
#
# Safety: refuses to run unless --expect-serial matches the device serial
# number, and always refuses on hostname orin-nx-vqplnc (that device is a live
# robot). Engines are passed as tag:engine_dir:mtp|plain[:adapter=path].
#
# Example:
#   sudo -E ./lora_nx_measure.sh --expect-serial 04216190xxxxx --out /data/bench/run1 \
#       --build-dir "$B" -- \
#       base:/data/engines/gemma4:plain \
#       mtp:/data/engines/gemma4-mtp:mtp:adapterA=/data/adapters/a
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${SCRIPT_DIR}/lora_bench.py"

PROTECTED_HOSTNAME="orin-nx-vqplnc"
TEGRASTATS_PID=""
SAMPLER_PID=""

usage() {
    cat <<'USAGE'
Usage: lora_nx_measure.sh --expect-serial SERIAL --out DIR [--build-dir DIR]
                          [--reps N] [--ttft-lengths LIST] [--skip-outputs]
                          [--lock FILE] [-- ENGINE SPEC...]

  --expect-serial SERIAL   Refuse to run unless it matches
                           /sys/firmware/devicetree/base/serial-number.
  --out DIR                Output directory (created; must not already hold a
                           bench.jsonl from an earlier interrupted pass).
  --build-dir DIR          EdgeLLM build tree passed to lora_bench.py
                           --build-dir (default: $B).
  --reps N                 Passed to lora_bench.py --reps.
  --ttft-lengths LIST      Passed to lora_bench.py --ttft-lengths.
  --skip-outputs           Passed to lora_bench.py --skip-outputs.
  --lock FILE              flock(1) lock file; serializes runs against other
                           GPU users of the device.

  ENGINE SPEC: tag:engine_dir:mtp|plain[:adapter=path]
    tag          short name used for the log files
    engine_dir   directory passed to lora_bench.py --engine
    mtp|plain    whether --mtp is passed
    adapter=path optional NAME=PATH pair; may repeat after further colons
                 (e.g. tag:dir:mtp:a=/x:b=/y); every adapter=path field is
                 passed as its own --adapter to lora_bench.py.

  The bench measures the base engine and each adapter; pass engine specs
  after '--' so paths with '=' or ':' stay unambiguous.
USAGE
}

die() {
    echo "lora_nx_measure.sh: ERROR: $*" >&2
    exit 1
}

on_exit() {
    stop_tegrastats
    stop_sampler
}

# ---------------------------------------------------------------- tegrastats

start_tegrastats() {
    local tag="$1" out_dir="$2"
    if ! command -v tegrastats >/dev/null 2>&1; then
        echo "lora_nx_measure.sh: WARNING: tegrastats not found; skipping" >&2
        return 0
    fi
    tegrastats --interval 100 --logfile "${out_dir}/tegrastats-${tag}.log" &
    TEGRASTATS_PID=$!
}

stop_tegrastats() {
    if [ -n "${TEGRASTATS_PID}" ] && kill -0 "${TEGRASTATS_PID}" 2>/dev/null; then
        kill "${TEGRASTATS_PID}" 2>/dev/null || true
        wait "${TEGRASTATS_PID}" 2>/dev/null || true
    fi
    TEGRASTATS_PID=""
}

# ------------------------------------------------------------ memory sampler

# One CSV line per second: timestamp, MemAvailable, SwapFree, cgroup
# memory.current, cgroup memory.peak. cgroup v2 path from /proc/self/cgroup.
sampler_loop() {
    local csv="$1" cgroup="$2"
    local mem_avail swap_free mem_cur mem_peak ts
    : >"${csv}"
    printf 'timestamp_s,mem_available_kb,swap_free_kb,cgroup_memory_current_bytes,cgroup_memory_peak_bytes\n' >>"${csv}"
    while true; do
        ts="$(date +%s)"
        mem_avail="$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null || echo '')"
        swap_free="$(awk '/SwapFree/ {print $2}' /proc/meminfo 2>/dev/null || echo '')"
        mem_cur=""
        mem_peak=""
        if [ -n "${cgroup}" ]; then
            mem_cur="$(cat "${cgroup}/memory.current" 2>/dev/null || echo '')"
            mem_peak="$(cat "${cgroup}/memory.peak" 2>/dev/null || echo '')"
        fi
        printf '%s,%s,%s,%s,%s\n' "${ts}" "${mem_avail}" "${swap_free}" "${mem_cur}" "${mem_peak}" >>"${csv}"
        sleep 1
    done
}

start_sampler() {
    local tag="$1" out_dir="$2" cgroup="$3"
    sampler_loop "${out_dir}/mem-${tag}.csv" "${cgroup}" &
    SAMPLER_PID=$!
}

stop_sampler() {
    if [ -n "${SAMPLER_PID}" ] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
        kill "${SAMPLER_PID}" 2>/dev/null || true
        wait "${SAMPLER_PID}" 2>/dev/null || true
    fi
    SAMPLER_PID=""
}

min_mem_available_kb() {
    local csv="$1"
    # Skip the header; empty fields (failed reads) are ignored by sort.
    tail -n +2 "${csv}" | cut -d, -f2 | grep -v '^$' | sort -n | head -1
}

# ------------------------------------------------------------------- preflight

device_serial() {
    # Jetson eFuse serial; NUL-separated on some releases.
    tr -d '\0' </sys/firmware/devicetree/base/serial-number 2>/dev/null || true
}

record_platform() {
    local out_dir="$1" phase="$2"
    {
        echo "=== ${phase} $(date -Is) ==="
        echo "hostname: $(hostname)"
        echo "model: $(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo unknown)"
        echo "serial: $(device_serial)"
        echo "memtotal_kb: $(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo unknown)"
        echo "nv_tegra_release:"
        cat /etc/nv_tegra_release 2>/dev/null || echo "(missing)"
        echo "nvpmodel:"
        nvpmodel -q 2>&1 || true
        echo "jetson_clocks:"
        if command -v jetson_clocks >/dev/null 2>&1; then
            jetson_clocks --show 2>&1 || echo "(jetson_clocks --show failed: needs root?)"
        else
            echo "(jetson_clocks not found)"
        fi
        echo "nvidia-smi:"
        nvidia-smi 2>&1 || echo "(nvidia-smi failed)"
        echo
    } >>"${out_dir}/platform.txt"
}

preflight() {
    local expected_serial="$1" out_dir="$2"

    local host
    host="$(hostname)"
    if [ "${host}" = "${PROTECTED_HOSTNAME}" ]; then
        die "hostname ${PROTECTED_HOSTNAME} is a live robot; this harness must not run there"
    fi

    local serial
    serial="$(device_serial)"
    if [ -z "${serial}" ]; then
        die "cannot read /sys/firmware/devicetree/base/serial-number"
    fi
    if [ "${serial}" != "${expected_serial}" ]; then
        die "serial mismatch: device serial '${serial}' != --expect-serial '${expected_serial}'"
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        die "nvidia-smi not found; is this a Jetson with the driver stack installed?"
    fi
    if ! nvidia-smi >/dev/null 2>&1; then
        die "nvidia-smi failed; GPU driver not healthy?"
    fi

    if [ ! -f "${BENCH}" ]; then
        die "lora_bench.py not found next to this script: ${BENCH}"
    fi
    if [ ! -d "${out_dir}" ]; then
        mkdir -p "${out_dir}"
    fi
    if [ -e "${out_dir}/bench.jsonl" ]; then
        die "${out_dir}/bench.jsonl already exists; use a fresh --out directory"
    fi
}

# ------------------------------------------------------------------ engine run

parse_engine_spec() {
    # tag:engine_dir:mtp|plain[:adapter=path...] -> globals SPEC_*.
    local spec="$1"
    local -a fields
    IFS=':' read -r -a fields <<<"${spec}"
    if [ "${#fields[@]}" -lt 3 ]; then
        die "engine spec '${spec}' must be tag:engine_dir:mtp|plain[:adapter=path]"
    fi
    SPEC_TAG="${fields[0]}"
    SPEC_ENGINE="${fields[1]}"
    SPEC_MODE="${fields[2]}"
    SPEC_ADAPTERS=()
    local i
    for i in "${fields[@]:3}"; do
        if [[ "${i}" != *=* ]]; then
            die "engine spec '${spec}': extra field '${i}' must be adapter=path"
        fi
        SPEC_ADAPTERS+=("${i}")
    done
    if [ -z "${SPEC_TAG}" ] || [ -z "${SPEC_ENGINE}" ]; then
        die "engine spec '${spec}': tag and engine_dir must be non-empty"
    fi
    case "${SPEC_MODE}" in
        mtp | plain) ;;
        *) die "engine spec '${spec}': mode must be 'mtp' or 'plain', got '${SPEC_MODE}'" ;;
    esac
}

run_engine() {
    local spec="$1" out_dir="$2" build_dir="$3" reps_arg="$4" lengths_arg="$5" skip_outputs="$6" lock_file="$7"
    parse_engine_spec "${spec}"

    local -a bench_args=(--engine "${SPEC_ENGINE}" --build-dir "${build_dir}")
    if [ "${SPEC_MODE}" = "mtp" ]; then
        bench_args+=(--mtp)
    fi
    local adapter
    for adapter in "${SPEC_ADAPTERS[@]:-}"; do
        [ -n "${adapter}" ] && bench_args+=(--adapter "${adapter}")
    done
    if [ -n "${reps_arg}" ]; then
        bench_args+=(--reps "${reps_arg}")
    fi
    if [ -n "${lengths_arg}" ]; then
        bench_args+=(--ttft-lengths "${lengths_arg}")
    fi
    if [ "${skip_outputs}" = "1" ]; then
        bench_args+=(--skip-outputs)
    fi
    bench_args+=(--out "${out_dir}/bench.jsonl")

    echo "=== engine ${SPEC_TAG}: ${SPEC_ENGINE} (${SPEC_MODE}) ==="
    start_tegrastats "${SPEC_TAG}" "${out_dir}"
    start_sampler "${SPEC_TAG}" "${out_dir}" "$(cgroup_v2_path)"
    local failed=0
    if [ -n "${lock_file}" ]; then
        if ! flock "${lock_file}" python3 "${BENCH}" "${bench_args[@]}"; then
            failed=1
        fi
    else
        if ! python3 "${BENCH}" "${bench_args[@]}"; then
            failed=1
        fi
    fi
    stop_sampler
    stop_tegrastats
    local min_avail
    min_avail="$(min_mem_available_kb "${out_dir}/mem-${SPEC_TAG}.csv")"
    echo "engine ${SPEC_TAG}: min MemAvailable = ${min_avail:-unknown} kB"
    if [ "${failed}" = "1" ]; then
        echo "lora_nx_measure.sh: WARNING: bench failed for engine ${SPEC_TAG}" >&2
        return 1
    fi
    return 0
}

cgroup_v2_path() {
    # '::' entry of /proc/self/cgroup is the v2 path relative to the cgroup
    # mount; on a stock Jetson root cgroup it is usually '/'.
    local v2
    v2="$(awk '$1 == "0::" {print $2}' /proc/self/cgroup 2>/dev/null || true)"
    if [ -z "${v2}" ]; then
        echo ""
        return
    fi
    if [ -d "/sys/fs/cgroup${v2}" ]; then
        echo "/sys/fs/cgroup${v2}"
    else
        echo ""
    fi
}

# ----------------------------------------------------------------------- main

main() {
    local expect_serial="" out_dir="" build_dir="${B:-}" reps="" lengths="" skip_outputs=0 lock_file=""
    local -a engines=()
    local after_ddash=0

    while [ $# -gt 0 ]; do
        case "$1" in
            --) after_ddash=1; shift ;;
            --expect-serial)
                [ -n "${2:-}" ] || die "--expect-serial needs a value"
                expect_serial="$2"; shift 2 ;;
            --out)
                [ -n "${2:-}" ] || die "--out needs a value"
                out_dir="$2"; shift 2 ;;
            --build-dir)
                [ -n "${2:-}" ] || die "--build-dir needs a value"
                build_dir="$2"; shift 2 ;;
            --reps)
                [ -n "${2:-}" ] || die "--reps needs a value"
                reps="$2"; shift 2 ;;
            --ttft-lengths)
                [ -n "${2:-}" ] || die "--ttft-lengths needs a value"
                lengths="$2"; shift 2 ;;
            --skip-outputs) skip_outputs=1; shift ;;
            --lock)
                [ -n "${2:-}" ] || die "--lock needs a value"
                lock_file="$2"; shift 2 ;;
            -h | --help) usage; exit 0 ;;
            -*) die "unknown option: $1" ;;
            *)
                if [ "${after_ddash}" = "1" ]; then
                    engines+=("$1")
                else
                    die "engine specs must come after '--' (got '$1')"
                fi
                shift ;;
        esac
    done

    [ -n "${expect_serial}" ] || die "--expect-serial is required"
    [ -n "${out_dir}" ] || die "--out is required"
    [ -n "${build_dir}" ] || die "--build-dir is required (or set \$B)"
    [ "${#engines[@]}" -gt 0 ] || die "no engine specs given (after '--')"

    preflight "${expect_serial}" "${out_dir}"
    trap on_exit EXIT

    record_platform "${out_dir}" "pre-run"
    : >"${out_dir}/bench.jsonl"

    local spec failures=0
    for spec in "${engines[@]}"; do
        if ! run_engine "${spec}" "${out_dir}" "${build_dir}" "${reps}" "${lengths}" "${skip_outputs}" "${lock_file}"; then
            failures=$((failures + 1))
        fi
    done

    record_platform "${out_dir}" "post-run"

    if [ "${failures}" -gt 0 ]; then
        echo "lora_nx_measure.sh: ${failures} engine run(s) failed; see logs in ${out_dir}" >&2
        exit 1
    fi
    echo "lora_nx_measure.sh: done; results in ${out_dir}"
}

main "$@"
