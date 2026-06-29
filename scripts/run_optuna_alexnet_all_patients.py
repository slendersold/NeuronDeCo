#!/usr/bin/env python3
"""
Optuna hyperparameter search: AlexNetTFR only, all subjects in a list.

Same layout as ``run_optuna_transformer_all_patients.py`` (PreprocessedData,
specs_with_car, cumulative loss metric, TPESampler, patience=max_epochs).

Default **10 trials** per subject. For small trial counts, ``n_startup_trials``
is capped so TPE runs after the startup phase (see ``make_tpe_sampler``).

Run:

    cd /path/to/NeuronDeCo && python scripts/run_optuna_alexnet_all_patients.py \\
        --preprocessed-root /path/to/Pirogov/PreprocessedData \\
        --out-dir /path/to/Pirogov/PreprocessedData/2026-04-01
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mne  # noqa: E402

from lib.data.normalisation import normalize_tfr_robust  # noqa: E402
from lib.data.tfr_dataset import TFRDataset  # noqa: E402
from lib.models.alexnet import AlexNetTFR  # noqa: E402
from lib.optuna import (  # noqa: E402
    attrs_fn,
    cumulative_loss_metric_factory,
    make_objective_engine,
    make_splits_fn_factory,
    objectives_fn,
    params_fn_factory_alexnet,
    run_fold_fn_factory,
)
from lib.training.epochs import eval_one_epoch_f1_macro, train_one_epoch  # noqa: E402


DEFAULT_SUBJECTS = [
    "s02",
    "s03",
    "s04",
    "s05",
    "s06",
    "s07",
    "s09",
    "s10",
    "s11",
    "s12",
    "s13",
    "s15",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str, code: int = 2) -> None:
    log("FATAL: " + msg)
    sys.exit(code)


def fmt_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def save_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def make_tpe_sampler(*, seed: int, n_trials: int) -> optuna.samplers.TPESampler:
    """TPESampler for multi-objective study (Optuna 4.6); startup capped by n_trials."""

    def _gamma(n_completed: int) -> int:
        if n_completed <= 0:
            return 1
        return max(1, min(int(math.ceil(0.3 * n_completed)), n_completed))

    # With only 10 trials, n_startup_trials=40 would mean purely random search.
    n_startup = min(40, max(1, n_trials // 2))
    if n_startup >= n_trials:
        n_startup = max(1, n_trials - 1)

    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=n_startup,
        multivariate=True,
        constant_liar=True,
        gamma=_gamma,
        prior_weight=2.0,
        consider_magic_clip=True,
        consider_endpoints=True,
    )


def resolve_tfr_path(
    subject_id: str,
    *,
    preprocessed_root: Path,
    project_root: Path,
) -> Path:
    candidates = [
        preprocessed_root / "specs_with_car" / f"tfr_{subject_id}.fif",
        project_root / "specs_with_car" / f"tfr_{subject_id}.fif",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "TFR not found for subject "
        f"{subject_id!r}. Checked: " + ", ".join(str(p) for p in candidates)
    )


def load_xy(
    tfr_path: Path,
    *,
    event_pos_code: int,
    crop_tmin: float,
    crop_tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    tfr_list = mne.time_frequency.read_tfrs(str(tfr_path))
    if not tfr_list:
        raise ValueError("read_tfrs returned empty list")
    tfr = tfr_list[0]
    if getattr(tfr, "events", None) is None:
        raise ValueError("TFR has no events attribute")
    y = np.where(tfr.events[:, 2] == event_pos_code, 1, 0).astype(np.int64)
    tfr = tfr.crop(tmin=crop_tmin, tmax=crop_tmax)
    X_full = tfr.data
    if X_full.ndim != 4:
        raise ValueError(f"Unexpected tfr.data shape: {X_full.shape}")
    X = normalize_tfr_robust(X_full[:, :, :-50, :]).astype(np.float32)
    del tfr, tfr_list, X_full
    gc.collect()
    return X, y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna TFR AlexNet search for all subjects.")
    p.add_argument(
        "--preprocessed-root",
        type=Path,
        default=None,
        help="Path to PreprocessedData. Default: <parent of NeuronDeCo>/PreprocessedData.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for SQLite studies.",
    )
    p.add_argument(
        "--subjects",
        type=str,
        default=",".join(DEFAULT_SUBJECTS),
        help="Comma-separated subject ids.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv", action="store_true")
    p.add_argument("--cv-aggregate", default="median", choices=["mean", "median"])
    p.add_argument("--event-pos-code", type=int, default=9)
    p.add_argument("--crop-tmin", type=float, default=0.0)
    p.add_argument("--crop-tmax", type=float, default=1.0)
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    p.add_argument("--cumulative-up-weight", type=float, default=1.1)
    p.add_argument("--cumulative-down-weight", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_root = _PROJECT_ROOT
    if args.preprocessed_root is not None:
        preprocessed_root = args.preprocessed_root.expanduser().resolve()
    else:
        guess = project_root.parent / "PreprocessedData"
        if not guess.is_dir():
            die(f"Set --preprocessed-root explicitly (default {guess} not found).")
        preprocessed_root = guess

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not subjects:
        die("No subjects after parsing --subjects")

    loss_metric = cumulative_loss_metric_factory(
        up_weight=args.cumulative_up_weight,
        down_weight=args.cumulative_down_weight,
    )
    patience_disabled = int(args.max_epochs)

    errors: dict[str, str] = {}
    saved: dict[str, str] = {}

    for subject_id in subjects:
        log(f"=== Subject {subject_id} (AlexNet) ===")
        try:
            tfr_path = resolve_tfr_path(
                subject_id, preprocessed_root=preprocessed_root, project_root=project_root
            )
            X, y = load_xy(
                tfr_path,
                event_pos_code=args.event_pos_code,
                crop_tmin=args.crop_tmin,
                crop_tmax=args.crop_tmax,
            )
            _n, c, _f, _t = X.shape
            num_classes = int(np.unique(y).shape[0])

            study_db = out_dir / f"{tfr_path.stem}_alexnet.db"
            storage_url = f"sqlite:///{study_db}"

            sampler = make_tpe_sampler(seed=args.seed, n_trials=args.n_trials)

            study = optuna.create_study(
                directions=["maximize", "minimize"],
                sampler=sampler,
                storage=storage_url,
                study_name=f"{tfr_path.stem}_alexnet",
                load_if_exists=True,
            )

            objective = make_objective_engine(
                X=X,
                y=y,
                make_splits_fn=make_splits_fn_factory(
                    test_size=args.test_size,
                    seed=args.seed,
                    cv=args.cv,
                ),
                run_fold_fn=run_fold_fn_factory(
                    ModelCls=AlexNetTFR,
                    device=device,
                    max_epochs=args.max_epochs,
                    patience=patience_disabled,
                    TFRDataset=TFRDataset,
                    DataLoader=DataLoader,
                    train_one_epoch=train_one_epoch,
                    eval_one_epoch_f1_macro=eval_one_epoch_f1_macro,
                    loss_metric=loss_metric,
                ),
                aggregate_mode=args.cv_aggregate,
                params_fn=params_fn_factory_alexnet(
                    in_channels=c,
                    num_classes=num_classes,
                ),
                objectives_fn=objectives_fn,
                attrs_fn=attrs_fn,
            )

            study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
            saved[subject_id] = str(study_db)
            log(f"OK {subject_id} -> {study_db}")

            del X, y, study, objective
            gc.collect()

        except Exception as e:
            errors[subject_id] = fmt_exc(e)
            log(f"ERROR {subject_id}: {errors[subject_id]}")
            err_txt = out_dir / f"tfr_{subject_id}_alexnet.error.txt"
            try:
                with open(err_txt, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
            except OSError:
                pass

    report = {
        "date": date.today().isoformat(),
        "model": "AlexNetTFR",
        "preprocessed_root": str(preprocessed_root),
        "out_dir": str(out_dir),
        "subjects": subjects,
        "n_trials": args.n_trials,
        "max_epochs": args.max_epochs,
        "patience": patience_disabled,
        "early_stopping": "disabled (patience >= max_epochs)",
        "sampler": "TPESampler (multivariate, constant_liar, n_startup capped by n_trials)",
        "saved": saved,
        "errors": errors,
    }
    save_json(out_dir / "_ALEXNET_RUN_REPORT.json", report)
    log(f"Report: {out_dir / '_ALEXNET_RUN_REPORT.json'}")


if __name__ == "__main__":
    main()
