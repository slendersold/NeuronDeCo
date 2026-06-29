#!/bin/bash
#SBATCH --job-name=confirmatory_s11
#SBATCH --output=logs/confirmatory_s11_%j.out
#SBATCH --error=logs/confirmatory_s11_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT="/ABSOLUTE/PATH/TO/NeuronDeCo"
CONFIG="${PROJECT_ROOT}/configs/confirmatory.yaml"

cd "${PROJECT_ROOT}"
mkdir -p logs

python scripts/run_confirmatory_analysis.py \
  --config "${CONFIG}" \
  --patients s11 \
  --models svm alexnet transformer \
  --device cuda \
  --seed 42 \
  --resume

python scripts/plot_confirmatory_results.py \
  --input-root "$(python - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/confirmatory.yaml").read_text())
print(cfg["output"]["root"])
PY
)" \
  --output-dir "$(python - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/confirmatory.yaml").read_text())
print(str(Path(cfg["output"]["root"]) / "figures"))
PY
)"
