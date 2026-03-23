#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from lib import MODEL_REGISTRY, MODE_REGISTRY, PipelineRunner, RunContext


def main() -> None:
    # Minimal fake data for skeleton demonstration only.
    X = np.random.randn(8, 7, 20, 30)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    model = MODEL_REGISTRY["alexnet"]()
    mode = MODE_REGISTRY["offline_epoch"]()
    context = RunContext(run_id="example-001", device="cpu", seed=42)

    runner = PipelineRunner(mode=mode, model=model, context=context)
    result = runner.run(X, y)
    print(result)


if __name__ == "__main__":
    main()

