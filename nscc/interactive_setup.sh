#!/usr/bin/env bash

# interactive_session.sh normally sources this automatically after qsub -I
# reaches a compute node. Manual recovery remains available with:
#   cd /home/users/nus/depeng/State-Predictive-Information-Bottleneck
#   source nscc/interactive_setup.sh
# It defines spib_run, which gives every debug execution its own persistent
# log and result directory while reusing the current GPU allocation.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this script instead of executing it:" >&2
    echo "  source nscc/interactive_setup.sh" >&2
    exit 2
fi

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "interactive_setup.sh must be sourced inside a PBS allocation." >&2
    return 1
fi

SPIB_INTERACTIVE_CODE_ROOT="${NSCC_RUNTIME_CODE_ROOT:-/home/users/nus/depeng/State-Predictive-Information-Bottleneck}"
SPIB_INTERACTIVE_TARGET="${NSCC_TARGET:-h100}"
if [[ "${SPIB_INTERACTIVE_TARGET}" == a100 ]]; then
    SPIB_PROJECT_DEFAULT=/home/project/11004454/depeng
else
    SPIB_PROJECT_DEFAULT=/data/projects/11014454/depeng
fi
SPIB_INTERACTIVE_PROJECT_ROOT="${NSCC_PROJECT_ROOT:-${SPIB_PROJECT_DEFAULT}}"
SPIB_INTERACTIVE_ENV_PREFIX="${NSCC_ENV_PREFIX:-${SPIB_INTERACTIVE_PROJECT_ROOT}/envs/spib}"

[[ -d "${SPIB_INTERACTIVE_CODE_ROOT}" ]] || {
    echo "Missing NSCC runtime checkout: ${SPIB_INTERACTIVE_CODE_ROOT}" >&2
    return 1
}
[[ -x "${SPIB_INTERACTIVE_ENV_PREFIX}/bin/python" ]] || {
    echo "Missing SPIB environment: ${SPIB_INTERACTIVE_ENV_PREFIX}" >&2
    return 1
}

if [[ "${SPIB_INTERACTIVE_TARGET}" == a100 ]]; then
    SPIB_MINIFORGE_DEFAULT=miniforge3
else
    SPIB_MINIFORGE_DEFAULT=miniforge3/24.3.0
fi
module load "${NSCC_MINIFORGE_MODULE:-${SPIB_MINIFORGE_DEFAULT}}" || return
source "$(conda info --base)/etc/profile.d/conda.sh" || return
conda activate "${SPIB_INTERACTIVE_ENV_PREFIX}" || return
cd "${SPIB_INTERACTIVE_CODE_ROOT}" || return
export PYTHONPATH="${SPIB_INTERACTIVE_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NSCC_PROJECT_ROOT="${SPIB_INTERACTIVE_PROJECT_ROOT}"

_spib_link_trpcage_data() {
    local src="${SPIB_INTERACTIVE_PROJECT_ROOT}/data/trpcage"
    local dest="${SPIB_INTERACTIVE_CODE_ROOT}/trpcage"
    local f
    [[ -d "${src}" ]] || return 0
    mkdir -p "${dest}"
    for f in traj_data.npy data_mean.npy data_std.npy \
        init_label_tica_kmeans200_lag200.npy tica_projection_lag200.npy; do
        if [[ -e "${src}/${f}" ]]; then
            ln -sfn "${src}/${f}" "${dest}/${f}"
        fi
    done
}

_spib_link_trpcage_data

spib_run() {
    local config="${1:-examples/Four_Well_hsic_config.ini}"
    local run_stamp run_id output_root log_file status code_version base_commit tracked_state

    [[ "${config}" != /* && "${config}" != .. && "${config}" != ../* && "${config}" != */../* ]] || {
        echo "Config must be a repository-relative path." >&2
        return 2
    }
    [[ -f "${SPIB_INTERACTIVE_CODE_ROOT}/${config}" ]] || {
        echo "Missing config: ${SPIB_INTERACTIVE_CODE_ROOT}/${config}" >&2
        return 1
    }

    _spib_link_trpcage_data

    code_version="$(tr -d '[:space:]' <"${SPIB_INTERACTIVE_CODE_ROOT}/VERSION" 2>/dev/null || true)"
    [[ -n "${code_version}" ]] || code_version=unversioned
    base_commit="$(sed -n 's/^base_commit=//p' "${SPIB_INTERACTIVE_CODE_ROOT}/.nscc-source-info" 2>/dev/null | head -1)"
    [[ -n "${base_commit}" ]] || base_commit="$(git -C "${SPIB_INTERACTIVE_CODE_ROOT}" rev-parse HEAD^{commit} 2>/dev/null || true)"
    tracked_state="$(sed -n 's/^tracked_state=//p' "${SPIB_INTERACTIVE_CODE_ROOT}/.nscc-source-info" 2>/dev/null | head -1)"
    [[ -n "${tracked_state}" ]] || tracked_state=unknown

    run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    run_id="interactive-${PBS_JOBID}-${code_version//[^0-9A-Za-z._-]/_}-${run_stamp}"
    output_root="${SPIB_INTERACTIVE_PROJECT_ROOT}/results/spib/${run_id}"
    log_file="${SPIB_INTERACTIVE_PROJECT_ROOT}/logs/spib/${run_id}.log"
    mkdir -p "${output_root}/SPIB" "${output_root}/fig" "${SPIB_INTERACTIVE_PROJECT_ROOT}/logs/spib"
    cp -- "${SPIB_INTERACTIVE_CODE_ROOT}/${config}" "${output_root}/submitted-config.ini"
    for provenance_file in VERSION CHANGELOG.md .nscc-source-info .nscc-source-manifest.sha256; do
        if [[ -f "${SPIB_INTERACTIVE_CODE_ROOT}/${provenance_file}" ]]; then
            cp -- "${SPIB_INTERACTIVE_CODE_ROOT}/${provenance_file}" "${output_root}/${provenance_file}"
        fi
    done

    export SPIB_OUTPUT_DIR="${output_root}/SPIB"
    export SPIB_FIG_DIR="${output_root}/fig"

    _spib_write_last_run() {
        local state="$1"
        mkdir -p "${SPIB_INTERACTIVE_PROJECT_ROOT}/results/spib"
        {
            echo "state=${state}"
            echo "run_id=${run_id}"
            echo "fig_dir=${output_root}/fig"
            echo "output=${output_root}"
            echo "log=${log_file}"
            echo "config=${config}"
            echo "exit_status=${status:-}"
            echo "updated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } >"${SPIB_INTERACTIVE_PROJECT_ROOT}/results/spib/.nscc-last-run"
    }
    status=""
    _spib_write_last_run running

    {
        echo "===== SPIB NSCC interactive run ====="
        echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "job=${PBS_JOBID}"
        echo "target=${SPIB_INTERACTIVE_TARGET}"
        echo "host=$(hostname)"
        echo "code_root=${SPIB_INTERACTIVE_CODE_ROOT}"
        echo "config=${config}"
        echo "env=${SPIB_INTERACTIVE_ENV_PREFIX}"
        echo "output=${output_root}"
        echo "source_mode=working-interactive"
        echo "code_version=${code_version}"
        echo "base_commit=${base_commit:-unknown}"
        echo "tracked_state=${tracked_state}"
    } > >(tee "${log_file}") 2>&1
    if python -u "${SPIB_INTERACTIVE_CODE_ROOT}/test_model_advanced.py" \
        -config "${SPIB_INTERACTIVE_CODE_ROOT}/${config}" \
        > >(tee -a "${log_file}") 2>&1; then
        status=0
    else
        status=$?
    fi
    {
        echo "exit_status=${status}"
        echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } | tee -a "${log_file}"
    echo "interactive_run=${run_id}"
    echo "log=${log_file}"
    echo "output=${output_root}"
    _spib_write_last_run finished
    return "${status}"
}

spib_gpu_check() {
python - <<'PY'
import torch
print("SPIB interactive environment ready")
print("torch=", torch.__version__)
print("torch_cuda_build=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the interactive GPU job")
print("gpu=", torch.cuda.get_device_name(0))
PY
}
echo "SPIB interactive environment ready"
echo "code_root=${SPIB_INTERACTIVE_CODE_ROOT}"
echo "env=${SPIB_INTERACTIVE_ENV_PREFIX}"
echo "Run: spib_run examples/Four_Well_hsic_config.ini"
echo "Optional GPU check: spib_gpu_check"
