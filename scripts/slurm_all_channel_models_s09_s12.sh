#!/usr/bin/env bash
#SBATCH --job-name=all_ch_models
#SBATCH --array=0-1
#SBATCH --output=all_ch_models_%A_%a.out
#SBATCH --error=all_ch_models_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

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

PATIENTS=(s09 s12)
PRESETS=(
  allch_100ms_2000ms
  allch_100ms_1000ms
  allch_0ms_1000ms_trim100_400
)

task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id < 0 || task_id >= ${#PATIENTS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected 0 or 1" >&2
  exit 2
fi
patient="${PATIENTS[task_id]}"

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

"${PYTHON_BIN}" scripts/prepare_all_channel_tfr.py \
  --patients "${patient}" \
  --config-path "${CONFIG_PATH}" \
  --study-def-path "${STUDY_DEF_PATH}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${TFR_ROOT}" \
  --work-root "${WORK_ROOT}" \
  --tfr-jobs 4

for preset in "${PRESETS[@]}"; do
  condition_output="${OUTPUT_ROOT}/${patient}/${preset}"
  echo "patient=${patient} preset=${preset} models=svm,alexnet,transformer"
  "${PYTHON_BIN}" scripts/run_confirmatory_analysis.py \
    --config "${PROJECT_ROOT}/configs/confirmatory.yaml" \
    --optuna-config "${OPTUNA_CONFIG}" \
    --data-root "${TFR_ROOT}" \
    --output-root "${condition_output}" \
    --patients "${patient}" \
    --models svm alexnet transformer \
    --input-preset "${preset}" \
    --device cuda \
    --seed 42 \
    --num-workers 0 \
    --resume
done
