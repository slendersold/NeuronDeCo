#!/usr/bin/env python3
"""
Эмуляция пайплайна ``run_optuna_transformer_all_patients.py`` без Optuna и без долгого обучения:

* те же субъекты, ``load_xy``, сплиты ``make_splits_fn_factory`` (holdout или ``--cv``),
  тот же ``run_fold_fn_factory`` + ``objectives_fn`` для агрегации по фолдам;
* **один** фиксированный набор гиперпараметров (плоский JSON как ``FrozenTrial.params``
  у transformer-студии), подставляются ``num_classes`` и ``seq_len`` с каждого субъекта;
* по умолчанию **1 эпоха** на фолд (можно увеличить ``--max-epochs``).

Выход: JSON-отчёт в ``--out-dir``, без SQLite.

Режимы параметров (ровно один):

* ``--params-json PATH`` — один JSON на всех субъектов (плоские ключи как ``trial.params``).
* ``--params-run-dir DIR`` — каталог run Optuna / экспорта ноутбука: для каждого субъекта читается
  ``DIR/tfr_<subject>_transformer_best_params.json`` (как в ``export_transformer_params_json.ipynb``).

Пример (общий JSON)::

    python scripts/run_transformer_fixed_params_all_patients.py \\
      --preprocessed-root /path/to/PreprocessedData \\
      --out-dir /path/to/out_fixed \\
      --params-json /path/to/best_trial_params.json \\
      --max-epochs 1

Пример (по пациенту, как после экспорта из SQLite)::

    python scripts/run_transformer_fixed_params_all_patients.py \\
      --preprocessed-root /path/to/PreprocessedData \\
      --out-dir /path/to/out_fixed \\
      --params-run-dir /path/to/PreprocessedData/2026-05-11 \\
      --max-epochs 1
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import mne  # noqa: E402

from lib.data.normalisation import normalize_tfr_robust  # noqa: E402
from lib.data.tfr_dataset import TFRDataset  # noqa: E402
from lib.models.tfr_transformer import TFRTransformerWrapper  # noqa: E402
from lib.models.tfr_transformer.preprocess import PREPROCESS_BUILDERS, SeqPool  # noqa: E402
from lib.optuna import (  # noqa: E402
    cumulative_loss_metric_factory,
    make_splits_fn_factory,
    objectives_fn,
    run_fold_fn_factory,
)
from lib.optuna.types import FoldResult, Params  # noqa: E402
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

_TRIAL_KEYS = frozenset(
    {
        "embed_dim",
        "nhead",
        "dim_fc",
        "num_layers",
        "dropout",
        "preprocess",
        "pooling",
        "batch_size",
        "encoder_dropout",
        "mlp_dropout",
        "use_conv",
        "conv_kernel_size",
        "conv_dropout",
        "lr",
        "weight_decay",
    }
)


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


def trial_flat_to_params(
    trial_params: Mapping[str, Any],
    *,
    num_classes: int,
    seq_len: int,
) -> Params:
    missing = _TRIAL_KEYS - frozenset(trial_params.keys())
    if missing:
        raise KeyError("params JSON misses keys: " + ", ".join(sorted(missing)))

    pre_name = str(trial_params["preprocess"])
    if pre_name not in PREPROCESS_BUILDERS:
        raise KeyError(f"Unknown preprocess={pre_name!r}")

    bs = int(trial_params["batch_size"])
    use_conv = bool(trial_params["use_conv"])

    model_kw: dict[str, Any] = {
        "num_classes": int(num_classes),
        "seq_len": int(seq_len),
        "embed_dim": int(trial_params["embed_dim"]),
        "nhead": int(trial_params["nhead"]),
        "dim_fc": int(trial_params["dim_fc"]),
        "num_layers": int(trial_params["num_layers"]),
        "dropout": float(trial_params["dropout"]),
        "encoder_dropout": float(trial_params["encoder_dropout"]),
        "mlp_dropout": float(trial_params["mlp_dropout"]),
        "use_conv": use_conv,
        "conv_kernel_size": int(trial_params["conv_kernel_size"]),
        "conv_dropout": float(trial_params["conv_dropout"]),
        "pooling": SeqPool(mode=str(trial_params["pooling"])),
        "preprocess": PREPROCESS_BUILDERS[pre_name](),
    }

    return {
        "model": model_kw,
        "optimizer": {
            "lr": float(trial_params["lr"]),
            "weight_decay": float(trial_params["weight_decay"]),
        },
        "tr_dataset": {"time_crop": None},
        "tr_loader": {"batch_size": bs, "shuffle": True},
        "vl_dataset": {"time_crop": None},
        "vl_loader": {"batch_size": bs, "shuffle": False},
    }


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


def fold_result_to_json(fr: FoldResult) -> dict[str, Any]:
    return {
        "split": fr.split,
        "best_f1": float(fr.best_f1),
        "loss_metric": float(fr.loss_metric),
        "curves": fr.curves,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="All-patients transformer run with fixed Optuna-style params, no search (default 1 epoch)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help="Single JSON for all subjects (flat keys like FrozenTrial.params).",
    )
    g.add_argument(
        "--params-run-dir",
        type=Path,
        default=None,
        help="Directory with per-subject exports: tfr_<subject>_transformer_best_params.json "
        "(same folder as Optuna *.db).",
    )
    p.add_argument("--preprocessed-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--subjects", type=str, default=",".join(DEFAULT_SUBJECTS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=1, help="Default 1 (fast dry-run); set 100 to match full training.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv", action="store_true")
    p.add_argument("--cv-aggregate", default="median", choices=["mean", "median"])
    p.add_argument("--event-pos-code", type=int, default=9)
    p.add_argument("--crop-tmin", type=float, default=0.0)
    p.add_argument("--crop-tmax", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--cumulative-up-weight", type=float, default=1.1)
    p.add_argument("--cumulative-down-weight", type=float, default=1.0)
    return p.parse_args()


def _load_flat_params(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        raise ValueError("JSON root must be an object")
    return dict(obj)


def params_json_path_for_subject(run_dir: Path, subject_id: str) -> Path:
    """Имя как у ноутбука export_transformer_params_json рядом с ``tfr_<id>_transformer.db``."""
    stem = f"tfr_{subject_id}_transformer"
    return (run_dir / f"{stem}_best_params.json").resolve()


def main() -> None:
    args = parse_args()
    project_root = _PROJECT_ROOT

    params_run_dir: Path | None = None
    shared_flat: dict[str, Any] | None = None
    params_path: Path | None = None

    if args.params_run_dir is not None:
        params_run_dir = args.params_run_dir.expanduser().resolve()
        if not params_run_dir.is_dir():
            die(f"--params-run-dir is not a directory: {params_run_dir}")
    else:
        assert args.params_json is not None
        params_path = args.params_json.expanduser().resolve()
        try:
            shared_flat = _load_flat_params(params_path)
        except (FileNotFoundError, ValueError) as e:
            die(f"--params-json: {e}")

    if args.preprocessed_root is not None:
        preprocessed_root = args.preprocessed_root.expanduser().resolve()
    else:
        guess = project_root.parent / "PreprocessedData"
        if not guess.is_dir():
            die(f"Set --preprocessed-root (default {guess} missing)")
        preprocessed_root = guess

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not subjects:
        die("No subjects")

    loss_metric_fn = cumulative_loss_metric_factory(
        up_weight=args.cumulative_up_weight,
        down_weight=args.cumulative_down_weight,
    )
    patience = int(args.max_epochs)

    make_splits = make_splits_fn_factory(
        test_size=args.test_size,
        seed=args.seed,
        cv=args.cv,
    )

    run_fold = run_fold_fn_factory(
        ModelCls=TFRTransformerWrapper,
        device=device,
        max_epochs=args.max_epochs,
        patience=patience,
        TFRDataset=TFRDataset,
        DataLoader=DataLoader,
        train_one_epoch=train_one_epoch,
        eval_one_epoch_f1_macro=eval_one_epoch_f1_macro,
        loss_metric=loss_metric_fn,
    )

    errors: dict[str, str] = {}
    per_subject: dict[str, Any] = {}

    for subject_id in subjects:
        log(f"=== Subject {subject_id} ===")
        try:
            if params_run_dir is not None:
                pj = params_json_path_for_subject(params_run_dir, subject_id)
                trial_like = _load_flat_params(pj)
                params_used = str(pj)
            else:
                assert shared_flat is not None
                assert params_path is not None
                trial_like = shared_flat
                params_used = str(params_path)

            tfr_path = resolve_tfr_path(
                subject_id, preprocessed_root=preprocessed_root, project_root=project_root
            )
            X, y = load_xy(
                tfr_path,
                event_pos_code=args.event_pos_code,
                crop_tmin=args.crop_tmin,
                crop_tmax=args.crop_tmax,
            )
            _n, _c, _f, t_bins = X.shape
            num_classes = int(np.unique(y).shape[0])

            fold_bundle = trial_flat_to_params(
                trial_like,
                num_classes=num_classes,
                seq_len=int(t_bins),
            )

            splits = make_splits(np.asarray(X), np.asarray(y))
            fold_results: list[FoldResult] = []
            for sp in splits:
                fold_results.append(run_fold(sp, fold_bundle))

            f1_agg, loss_agg = objectives_fn(fold_results, args.cv_aggregate)

            per_subject[subject_id] = {
                "params_json_used": params_used,
                "tfr_path": str(tfr_path),
                "num_classes": num_classes,
                "seq_len": int(t_bins),
                "aggregate_mode": args.cv_aggregate,
                "f1_aggregate": float(f1_agg),
                "loss_metric_aggregate": float(loss_agg),
                "folds": [fold_result_to_json(f) for f in fold_results],
            }
            log(
                f"OK {subject_id}: f1_{args.cv_aggregate}={f1_agg:.6f} "
                f"loss_metric_{args.cv_aggregate}={loss_agg:.6f}"
            )

            del X, y
            gc.collect()

        except Exception as e:
            errors[subject_id] = fmt_exc(e)
            log(f"ERROR {subject_id}: {errors[subject_id]}")
            err_txt = out_dir / f"tfr_{subject_id}_transformer_fixed.error.txt"
            try:
                with open(err_txt, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
            except OSError:
                pass

    report = {
        "date": date.today().isoformat(),
        "script": "run_transformer_fixed_params_all_patients",
        "notes": (
            "Emulates objective stack of run_optuna_transformer_all_patients.py: "
            "same splits/run_fold/objectives_fn, fixed flat params from JSON, no Optuna."
        ),
        "params_mode": "per_subject_run_dir" if params_run_dir else "single_json",
        "params_run_dir": str(params_run_dir) if params_run_dir else None,
        "params_json_single": str(params_path) if params_path is not None else None,
        "preprocessed_root": str(preprocessed_root),
        "out_dir": str(out_dir),
        "params_flat_snapshot": shared_flat if shared_flat is not None else None,
        "subjects": subjects,
        "max_epochs": args.max_epochs,
        "patience": patience,
        "test_size": args.test_size,
        "cv": bool(args.cv),
        "cv_aggregate": args.cv_aggregate,
        "device": str(device),
        "per_subject": per_subject,
        "errors": errors,
    }
    save_json(out_dir / "_TRANSFORMER_FIXED_PARAMS_REPORT.json", report)
    log(f"Report: {out_dir / '_TRANSFORMER_FIXED_PARAMS_REPORT.json'}")


if __name__ == "__main__":
    main()
