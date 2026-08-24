#!/usr/bin/env bash

# interactive_session.sh sources this after gpu-interactive reaches a GPU node.
# Manual recovery:
#   cd /userhome/cs3/lidepeng/TS/State-Predictive-Information-Bottleneck
#   source hku/interactive_setup.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this script instead of executing it:" >&2
    echo "  source hku/interactive_setup.sh" >&2
    exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "interactive_setup.sh must be sourced inside gpu-interactive / a GPU allocation." >&2
    return 1
fi

SPIB_INTERACTIVE_CODE_ROOT="${HKU_RUNTIME_CODE_ROOT:-/userhome/cs3/lidepeng/TS/State-Predictive-Information-Bottleneck}"
SPIB_INTERACTIVE_PROJECT_ROOT="${HKU_PROJECT_ROOT:-${SPIB_INTERACTIVE_CODE_ROOT}}"
SPIB_CONDA_BASE="${HKU_CONDA_BASE:-/userhome/cs3/lidepeng/anaconda3}"
SPIB_INTERACTIVE_ENV_PREFIX="${HKU_ENV_PREFIX:-${SPIB_CONDA_BASE}}"

[[ -d "${SPIB_INTERACTIVE_CODE_ROOT}" ]] || {
    echo "Missing HKU runtime checkout: ${SPIB_INTERACTIVE_CODE_ROOT}" >&2
    return 1
}
[[ -x "${SPIB_INTERACTIVE_ENV_PREFIX}/bin/python" ]] || {
    echo "Missing SPIB environment: ${SPIB_INTERACTIVE_ENV_PREFIX}" >&2
    return 1
}

source "${SPIB_CONDA_BASE}/etc/profile.d/conda.sh" || return
if [[ "${SPIB_INTERACTIVE_ENV_PREFIX}" == "${SPIB_CONDA_BASE}" ]]; then
    conda activate base || return
else
    conda activate "${SPIB_INTERACTIVE_ENV_PREFIX}" || return
fi
cd "${SPIB_INTERACTIVE_CODE_ROOT}" || return
export PYTHONPATH="${SPIB_INTERACTIVE_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HKU_PROJECT_ROOT="${SPIB_INTERACTIVE_PROJECT_ROOT}"
unset NSCC_PROJECT_ROOT

spib_run() {
    local config="${1:-examples/Four_Well_hsic_hku_config.ini}"
    local run_stamp run_id output_root log_file status code_version base_commit tracked_state job_id

    [[ "${config}" != /* && "${config}" != .. && "${config}" != ../* && "${config}" != */../* ]] || {
        echo "Config must be a repository-relative path." >&2
        return 2
    }
    [[ -f "${SPIB_INTERACTIVE_CODE_ROOT}/${config}" ]] || {
        echo "Missing config: ${SPIB_INTERACTIVE_CODE_ROOT}/${config}" >&2
        return 1
    }

    code_version="$(tr -d '[:space:]' <"${SPIB_INTERACTIVE_CODE_ROOT}/VERSION" 2>/dev/null || true)"
    [[ -n "${code_version}" ]] || code_version=unversioned
    base_commit="$(sed -n 's/^base_commit=//p' "${SPIB_INTERACTIVE_CODE_ROOT}/.hku-source-info" 2>/dev/null | head -1)"
    [[ -n "${base_commit}" ]] || base_commit="$(git -C "${SPIB_INTERACTIVE_CODE_ROOT}" rev-parse HEAD^{commit} 2>/dev/null || true)"
    tracked_state="$(sed -n 's/^tracked_state=//p' "${SPIB_INTERACTIVE_CODE_ROOT}/.hku-source-info" 2>/dev/null | head -1)"
    [[ -n "${tracked_state}" ]] || tracked_state=unknown
    job_id="${SLURM_JOB_ID:-nogpujob}"

    run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    run_id="interactive-${job_id}-${code_version//[^0-9A-Za-z._-]/_}-${run_stamp}"
    output_root="${SPIB_INTERACTIVE_PROJECT_ROOT}/results/spib/${run_id}"
    log_file="${SPIB_INTERACTIVE_PROJECT_ROOT}/logs/spib/${run_id}.log"
    mkdir -p "${output_root}/SPIB" "${output_root}/fig" "${SPIB_INTERACTIVE_PROJECT_ROOT}/logs/spib"
    cp -- "${SPIB_INTERACTIVE_CODE_ROOT}/${config}" "${output_root}/submitted-config.ini"
    for provenance_file in VERSION CHANGELOG.md .hku-source-info .hku-source-manifest.sha256; do
        if [[ -f "${SPIB_INTERACTIVE_CODE_ROOT}/${provenance_file}" ]]; then
            cp -- "${SPIB_INTERACTIVE_CODE_ROOT}/${provenance_file}" "${output_root}/${provenance_file}"
        fi
    done

    export SPIB_OUTPUT_DIR="${output_root}/SPIB"
    export SPIB_FIG_DIR="${output_root}/fig"

    _hku_write_last_run() {
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
        } >"${SPIB_INTERACTIVE_PROJECT_ROOT}/results/spib/.hku-last-run"
    }
    status=""
    _hku_write_last_run running

    {
        echo "===== SPIB HKU interactive run ====="
        echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "job=${job_id}"
        echo "target=hku"
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
    _hku_write_last_run finished
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
echo "Run: spib_run examples/Four_Well_hsic_hku_config.ini"
echo "Optional GPU check: spib_gpu_check"
