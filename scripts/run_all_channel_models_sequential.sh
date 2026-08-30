#!/usr/bin/env bash
# Run all-channel preprocessing and model evaluation sequentially in an
# already allocated GPU node. Safe to restart: verified TFRs and completed
# folds are reused unless the corresponding overwrite variable is set to 1.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/beegfs/home/t.samsonov/notebooks/Pirogov/NeuronDeCo}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-/beegfs/home/t.samsonov/notebooks/Pirogov/PreprocessedData}"
DATA_ROOT="${DATA_ROOT:-/beegfs/home/t.samsonov/notebooks/Pirogov/PirogovDATA}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_PATH="${CONFIG_PATH:-${PREPROCESSED_ROOT}/config.py}"
STUDY_DEF_PATH="${STUDY_DEF_PATH:-${PREPROCESSED_ROOT}/study_definition_open_vs_all}"
OPTUNA_CONFIG="${OPTUNA_CONFIG:-${PROJECT_ROOT}/configs/optuna_sources.yaml}"
TFR_ROOT="${TFR_ROOT:-${PREPROCESSED_ROOT}/all_channels_tfr}"
WORK_ROOT="${WORK_ROOT:-${PREPROCESSED_ROOT}/all_channel_model_benchmark_work}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PREPROCESSED_ROOT}/all_channel_model_benchmark}"

TFR_JOBS="${TFR_JOBS:-2}"
TFR_BATCH_SIZE="${TFR_BATCH_SIZE:-16}"
PATIENTS_TO_RUN="${PATIENTS_TO_RUN:-s09 s12}"

# Keep BLAS/joblib from creating hidden CPU worker pools and duplicating large
# arrays. Neural models still use the single allocated GPU.
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PRESETS=(
  allch_100ms_2000ms
  allch_100ms_1000ms
  allch_0ms_1000ms_trim100_400
)
MODELS=(svm alexnet transformer)
read -r -a PATIENTS <<< "${PATIENTS_TO_RUN}"

on_error() {
  local exit_code=$?
  echo "[FAILED] line=${BASH_LINENO[0]} exit_code=${exit_code} time=$(date --iso-8601=seconds)" >&2
  exit "${exit_code}"
}
trap on_error ERR

condition_has_all_markers() {
  local condition_root=$1
  local patient_id=$2
  local model_name fold_id marker_path
  for model_name in "${MODELS[@]}"; do
    for fold_id in 0 1 2 3 4; do
      marker_path="${condition_root}/${patient_id}/${model_name}/fold_${fold_id}/COMPLETED.json"
      [[ -s "${marker_path}" ]] || return 1
    done
  done
  return 0
}

for required_path in \
  "${PROJECT_ROOT}/scripts/prepare_all_channel_tfr.py" \
  "${PROJECT_ROOT}/scripts/run_confirmatory_analysis.py" \
  "${CONFIG_PATH}" \
  "${STUDY_DEF_PATH}" \
  "${OPTUNA_CONFIG}" \
  "${DATA_ROOT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${TFR_ROOT}" "${WORK_ROOT}" "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

for patient in "${PATIENTS[@]}"; do
  echo "[PATIENT START] patient=${patient} time=$(date --iso-8601=seconds)"

  prepare_args=(
    --patients "${patient}"
    --config-path "${CONFIG_PATH}"
    --study-def-path "${STUDY_DEF_PATH}"
    --data-root "${DATA_ROOT}"
    --output-root "${TFR_ROOT}"
    --work-root "${WORK_ROOT}"
    --tfr-jobs "${TFR_JOBS}"
    --tfr-batch-size "${TFR_BATCH_SIZE}"
  )
  if [[ "${FORCE_REBUILD_TFR:-0}" == "1" ]]; then
    prepare_args+=(--overwrite)
  fi

  if ! "${PYTHON_BIN}" scripts/prepare_all_channel_tfr.py "${prepare_args[@]}"; then
    echo "[TFR RETRY] patient=${patient} with n_jobs=1 batch_size=4" >&2
    "${PYTHON_BIN}" scripts/prepare_all_channel_tfr.py \
      --patients "${patient}" \
      --config-path "${CONFIG_PATH}" \
      --study-def-path "${STUDY_DEF_PATH}" \
      --data-root "${DATA_ROOT}" \
      --output-root "${TFR_ROOT}" \
      --work-root "${WORK_ROOT}" \
      --tfr-jobs 1 \
      --tfr-batch-size 4 \
      --overwrite
  fi

  tfr_path="${TFR_ROOT}/tfr_${patient}.fif"
  audit_path="${WORK_ROOT}/${patient}/tfr_preparation.json"
  if [[ ! -s "${tfr_path}" || ! -s "${audit_path}" ]]; then
    echo "TFR verification failed for ${patient}: ${tfr_path}, ${audit_path}" >&2
    exit 3
  fi
  "${PYTHON_BIN}" -c \
    'import json, pathlib, sys; audit=json.loads(pathlib.Path(sys.argv[1]).read_text()); counts=audit.get("labels", {}).get("class_counts", {}); assert int(counts.get("0", 0)) > 0 and int(counts.get("1", 0)) > 0, counts; assert pathlib.Path(sys.argv[2]).is_file()' \
    "${audit_path}" "${tfr_path}"
  echo "[TFR VERIFIED] patient=${patient} file=${tfr_path}"

  for preset in "${PRESETS[@]}"; do
    condition_output="${OUTPUT_ROOT}/${patient}/${preset}"
    if [[ "${OVERWRITE_RESULTS:-0}" != "1" ]] \
      && condition_has_all_markers "${condition_output}" "${patient}"; then
      echo "[CONDITION SKIP] verified 15 folds patient=${patient} preset=${preset}"
      continue
    fi
    run_mode=(--resume)
    if [[ "${OVERWRITE_RESULTS:-0}" == "1" ]]; then
      run_mode=(--overwrite)
    fi

    echo "[CONDITION START] patient=${patient} preset=${preset} time=$(date --iso-8601=seconds)"
    run_args=(
      --config "${PROJECT_ROOT}/configs/confirmatory.yaml"
      --optuna-config "${OPTUNA_CONFIG}"
      --data-root "${TFR_ROOT}"
      --output-root "${condition_output}"
      --patients "${patient}"
      --models "${MODELS[@]}"
      --input-preset "${preset}"
      --device cuda
      --seed 42
      --n-jobs 1
      --num-workers 0
    )
    if ! "${PYTHON_BIN}" scripts/run_confirmatory_analysis.py \
      "${run_args[@]}" "${run_mode[@]}"; then
      echo "[CONDITION RETRY] patient=${patient} preset=${preset} resume completed folds" >&2
      "${PYTHON_BIN}" scripts/run_confirmatory_analysis.py \
        "${run_args[@]}" --resume
    fi

    if ! condition_has_all_markers "${condition_output}" "${patient}"; then
      echo "Condition did not produce all 15 completion markers: ${condition_output}" >&2
      exit 4
    fi
    echo "[CONDITION COMPLETE] patient=${patient} preset=${preset} time=$(date --iso-8601=seconds)"
  done

  echo "[PATIENT COMPLETE] patient=${patient} time=$(date --iso-8601=seconds)"
done

echo "[ALL COMPLETE] patients=${PATIENTS_TO_RUN} time=$(date --iso-8601=seconds)"
