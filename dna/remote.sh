#!/usr/bin/env bash

set -euo pipefail

SSH_HOST="${DNA_SSH_HOST:-lidp@zlogin1.ddns.comp.nus.edu.sg}"
REMOTE_CODE_ROOT="${DNA_CODE_ROOT:-/mnt/rna01/lidp/State-Predictive-Information-Bottleneck}"
PROJECT_ROOT="${DNA_PROJECT_ROOT:-/mnt/rna01/lidp/spib-project}"
LOCAL_ROOT="$(git rev-parse --show-toplevel)"

usage() {
    cat <<'EOF'
Usage: dna/remote.sh COMMAND [ARGS]

Commands:
  probe                              Show DNA GPU partitions and your jobs
  update                             Fast-forward DNA to pushed hsic-spib HEAD
  install-env                        Submit the spib-cu118 environment job
  probe-gpu GPUA100|GPUA40           Submit a short PyTorch/GPU check
  submit GPUA100|GPUA40 [config.ini] Submit a commit-bound SPIB run
  status                             Show your DNA Slurm jobs
  tail JOB_ID                        Follow the persistent SPIB log
  cancel JOB_ID                      Cancel one of your jobs
EOF
}

quote_args() {
    local arg
    printf '%q ' "$@"
}

require_job_id() {
    [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "A numeric Slurm job ID is required" >&2; exit 2; }
}

command="${1:-}"
case "${command}" in
    probe)
        ssh "${SSH_HOST}" 'sinfo -p GPUA100,GPUA40 -o "%P %a %l %D %G %C"; squeue -u "$USER" -o "%.18i %.9P %.20j %.2t %.10M %R"'
        ;;
    update)
        cd "${LOCAL_ROOT}"
        [[ "$(git symbolic-ref --quiet --short HEAD || true)" == "hsic-spib" ]] || {
            echo "Local checkout must be on hsic-spib" >&2
            exit 1
        }
        if ! git diff --quiet || ! git diff --cached --quiet; then
            echo "Local checkout has tracked changes; commit them before DNA update." >&2
            exit 1
        fi
        local_commit="$(git rev-parse HEAD^{commit})"
        git fetch personal hsic-spib
        remote_commit="$(git rev-parse personal/hsic-spib^{commit})"
        [[ "${local_commit}" == "${remote_commit}" ]] || {
            echo "Local HEAD is not the pushed personal/hsic-spib tip." >&2
            echo "local=${local_commit}" >&2
            echo "pushed=${remote_commit}" >&2
            exit 1
        }
        remote_command="cd $(quote_args "${REMOTE_CODE_ROOT}")&& DNA_CODE_ROOT=$(quote_args "${REMOTE_CODE_ROOT}")dna/update_code.sh"
        dna_commit="$(ssh "${SSH_HOST}" "${remote_command}" | tail -1)"
        [[ "${dna_commit}" == "${local_commit}" ]] || {
            echo "DNA commit mismatch: ${dna_commit}" >&2
            exit 1
        }
        echo "DNA now uses ${dna_commit}"
        ;;
    install-env)
        remote_command="mkdir -p $(quote_args "${PROJECT_ROOT}/logs/slurm")&& cd $(quote_args "${REMOTE_CODE_ROOT}")&& sbatch --parsable --output=$(quote_args "${PROJECT_ROOT}/logs/slurm/spib-env-%j.out")dna/install_spib_env.slurm"
        ssh "${SSH_HOST}" "${remote_command}"
        ;;
    probe-gpu)
        partition="${2:-}"
        case "${partition}" in GPUA100|GPUA40) ;; *) usage; exit 2 ;; esac
        remote_command="mkdir -p $(quote_args "${PROJECT_ROOT}/logs/gpu-probe")&& cd $(quote_args "${REMOTE_CODE_ROOT}")&& sbatch --parsable --partition=$(quote_args "${partition}")--output=$(quote_args "${PROJECT_ROOT}/logs/gpu-probe/${partition}-%j.out")dna/probe_gpu.slurm"
        ssh "${SSH_HOST}" "${remote_command}"
        ;;
    submit)
        partition="${2:-}"
        config="${3:-examples/Four_Well_hsic_config.ini}"
        case "${partition}" in GPUA100|GPUA40) ;; *) usage; exit 2 ;; esac
        [[ "${config}" != *,* && "${config}" != *$'\n'* ]] || { echo "Unsafe config path" >&2; exit 2; }
        remote_command="cd $(quote_args "${REMOTE_CODE_ROOT}")&& dna/submit_spib.sh $(quote_args "${partition}" "${config}")"
        ssh "${SSH_HOST}" "${remote_command}"
        ;;
    status)
        ssh "${SSH_HOST}" 'squeue -u "$USER" -o "%.18i %.9P %.20j %.2t %.10M %R"'
        ;;
    tail)
        require_job_id "${2:-}"
        ssh -t "${SSH_HOST}" "tail -n 100 -f $(quote_args "${PROJECT_ROOT}/logs/spib/${2}.log")"
        ;;
    cancel)
        require_job_id "${2:-}"
        ssh "${SSH_HOST}" "scancel $(quote_args "${2}")"
        ;;
    *) usage; exit 2 ;;
esac
