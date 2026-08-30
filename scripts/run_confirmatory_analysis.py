#!/usr/bin/env python3
"""
Confirmatory cross-validation with frozen Optuna hyperparameters.

Re-trains SVM, AlexNet, and Transformer on fixed outer folds without re-running Optuna.

Example (pilot dry-run)::

    python scripts/run_confirmatory_analysis.py \\
        --config configs/confirmatory.yaml \\
        --patients s11 \\
        --models svm alexnet transformer \\
        --dry-run

Example (pilot run)::

    python scripts/run_confirmatory_analysis.py \\
        --config configs/confirmatory.yaml \\
        --patients s11 \\
        --models svm alexnet transformer \\
        --device cuda \\
        --seed 42 \\
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

sys.path.insert(0, str(_SCRIPT_DIR))
import confirmatory_support as cs  # noqa: E402

collect_fold_metrics = cs.collect_fold_metrics
collect_oof_predictions = cs.collect_oof_predictions
patient_model_summary = cs.patient_model_summary
statistical_summary = cs.statistical_summary
ConfirmatoryConfig = cs.ConfirmatoryConfig
load_confirmatory_config = cs.load_confirmatory_config
paths_checklist = cs.paths_checklist
build_fold_assignment_table = cs.build_fold_assignment_table
load_tfr_xy_metadata = cs.load_tfr_xy_metadata
make_outer_folds = cs.make_outer_folds
normalize_train_test = cs.normalize_train_test
fold_dir = cs.fold_dir
is_fold_completed = cs.is_fold_completed
log_error = cs.log_error
save_json_atomic = cs.save_json_atomic
update_progress = cs.update_progress
write_completed_marker = cs.write_completed_marker
predictions_to_dataframe = cs.predictions_to_dataframe
audit_study_cv_mode = cs.audit_study_cv_mode
discover_study_path = cs.discover_study_path
load_selected_trial = cs.load_selected_trial
scan_run_dir = cs.scan_run_dir
build_params_for_model = cs.build_params_for_model
capture_environment = cs.capture_environment
set_global_seeds = cs.set_global_seeds
train_eval_nn_outer_fold = cs.train_eval_nn_outer_fold
train_eval_svm_outer_fold = cs.train_eval_svm_outer_fold
from lib.models.alexnet import AlexNetTFR  # noqa: E402
from lib.models.tfr_svm import TfrParadigmSvmClassifier  # noqa: E402
from lib.models.tfr_transformer import TFRTransformerWrapper  # noqa: E402

MODEL_CLASSES = {
    "alexnet": AlexNetTFR,
    "transformer": TFRTransformerWrapper,
    "svm": TfrParadigmSvmClassifier,
}


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Confirmatory CV analysis with frozen Optuna hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config (paths, CV, training).",
    )
    p.add_argument("--patients", nargs="+", default=None, help="Patient ids.")
    p.add_argument(
        "--models",
        nargs="+",
        choices=["svm", "alexnet", "transformer"],
        default=None,
    )
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None, help="Alias for preprocessed_root.")
    p.add_argument("--optuna-config", type=Path, default=None, help="Optional extra YAML with studies.")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--folds", nargs="*", type=int, default=None, help="Subset of fold ids, e.g. 0 1 2")
    p.add_argument(
        "--input-preset",
        choices=sorted(cs.INPUT_PRESETS),
        default=None,
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def merge_config(args: argparse.Namespace) -> ConfirmatoryConfig:
    cfg = load_confirmatory_config(args.config.expanduser().resolve())
    if args.patients:
        cfg.patients = args.patients
    if args.models:
        cfg.models = args.models
    if args.output_root:
        cfg.output_root = args.output_root.expanduser().resolve()
    if args.data_root:
        cfg.preprocessed_root = args.data_root.expanduser().resolve()
    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.n_jobs is not None:
        cfg.n_jobs = args.n_jobs
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.input_preset:
        cfg.input_preset = args.input_preset
    if args.optuna_config and args.optuna_config.is_file():
        extra = load_confirmatory_config(args.optuna_config.expanduser().resolve())
        if extra.studies:
            cfg.studies.extend(extra.studies)
        if extra.optuna_run_dir and cfg.optuna_run_dir is None:
            cfg.optuna_run_dir = extra.optuna_run_dir
    return cfg


def validate_config(cfg: ConfirmatoryConfig, *, dry_run: bool) -> list[str]:
    issues: list[str] = []
    for label, path, _desc in paths_checklist(cfg):
        if path is None:
            if label != "data.sample_manifest":
                issues.append(f"Missing required path: {label}")
        elif label != "data.sample_manifest" and not dry_run:
            if label.startswith("optuna") and not path.exists():
                issues.append(f"Path does not exist: {label} -> {path}")
            if label.startswith("data.preprocessed") and not path.exists():
                issues.append(f"Path does not exist: {label} -> {path}")
            if label.startswith("output") and path is not None:
                path.mkdir(parents=True, exist_ok=True)
    return issues


def run_dry_run(
    cfg: ConfirmatoryConfig,
    project_root: Path,
    *,
    fold_subset: list[int] | None,
) -> int:
    logging.info("=== DRY RUN ===")
    issues = validate_config(cfg, dry_run=True)

    print("\n## Path checklist")
    for label, path, desc in paths_checklist(cfg):
        status = "OK" if path and path.exists() else ("MISSING" if path else "NOT SET")
        print(f"  [{status}] {label}: {path}\n      {desc}")

    if issues:
        print("\n## Configuration issues")
        for issue in issues:
            print(f"  - {issue}")

    if cfg.optuna_run_dir and cfg.optuna_run_dir.is_dir():
        found = scan_run_dir(cfg.optuna_run_dir)
        print(f"\n## Studies discovered under {cfg.optuna_run_dir}: {len(found)}")
        for ref in found:
            print(f"  {ref.patient:5s} {ref.model:12s} {ref.storage.name}")

    print("\n## Selected trials (planned)")
    selected_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for patient in cfg.patients:
        for model in cfg.models:
            try:
                ref = discover_study_path(
                    cfg.optuna_run_dir or Path("."),
                    patient,
                    model,
                    explicit_studies=cfg.studies,
                )
                audit = audit_study_cv_mode(ref)
                audit_rows.append(audit.__dict__)
                if ref.storage.is_file():
                    trial = load_selected_trial(ref, selector=cfg.trial_selector)
                    selected_rows.append(
                        {
                            "patient": trial.patient,
                            "model": trial.model,
                            "storage": str(trial.storage),
                            "study_name": trial.study_name,
                            "trial_number": trial.trial_number,
                            "objective_f1": trial.objective_f1,
                            "objective_loss": trial.objective_loss,
                        }
                    )
                    print(
                        f"  {patient}/{model}: trial={trial.trial_number} "
                        f"f1={trial.objective_f1:.4f} db={ref.storage.name}"
                    )
                else:
                    print(f"  {patient}/{model}: DB MISSING -> {ref.storage}")
            except Exception as exc:
                print(f"  {patient}/{model}: ERROR {exc}")

        try:
            tfr_path = cfg.resolve_tfr_path(patient, project_root)
            preset = cfg.get_input_preset()
            loaded = load_tfr_xy_metadata(
                tfr_path,
                patient=patient,
                event_pos_code=cfg.event_pos_code,
                preset=preset,
                sample_manifest=cfg.sample_manifest,
            )
            splits = make_outer_folds(
                loaded.y,
                n_splits=cfg.n_folds,
                seed=cfg.cv_seed,
                fold_subset=fold_subset,
            )
            print(
                f"\n## TFR {patient}: path={tfr_path.name} "
                f"shape=({loaded.X_raw.shape}) preset={preset.name}"
            )
            print(f"   folds={len(splits)} seed={cfg.cv_seed}")
        except Exception as exc:
            print(f"\n## TFR {patient}: ERROR {exc}")

    n_runs = len(cfg.patients) * len(cfg.models) * cfg.n_folds
    print(f"\n## Planned fold runs: {n_runs}")
    print("Dry-run complete. No training performed.")
    if audit_rows:
        print("\n## Optuna CV audit (from available metadata)")
        audit_df = pd.DataFrame(audit_rows)
        print(audit_df.to_string(index=False))
    return 1 if issues else 0


def save_run_artifacts(
    output_root: Path,
    *,
    cfg: ConfirmatoryConfig,
    args: argparse.Namespace,
    project_root: Path,
) -> None:
    run_cfg = {
        "patients": cfg.patients,
        "models": cfg.models,
        "input_preset": cfg.input_preset,
        "n_folds": cfg.n_folds,
        "cv_seed": cfg.cv_seed,
        "trial_selector": cfg.trial_selector,
        "command_line": " ".join(sys.argv),
        "config_path": str(args.config),
    }
    save_json_atomic(output_root / "run_config.json", run_cfg)
    save_json_atomic(output_root / "environment.json", capture_environment(project_root))


def run_fold_job(
    *,
    cfg: ConfirmatoryConfig,
    patient: str,
    model: str,
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    loaded,
    selected_trial,
    output_root: Path,
    device: str,
    resume: bool,
    overwrite: bool,
) -> None:
    fdir = fold_dir(output_root, patient, model, fold)
    if resume and not overwrite and is_fold_completed(fdir):
        logging.info("Skip completed %s/%s fold %s", patient, model, fold)
        return

    fdir.mkdir(parents=True, exist_ok=True)
    X_train_raw = loaded.X_raw[train_idx]
    X_test_raw = loaded.X_raw[test_idx]
    y_train = loaded.y[train_idx]
    y_test = loaded.y[test_idx]

    X_train, X_test, norm_meta = normalize_train_test(X_train_raw, X_test_raw)
    num_classes = int(np.unique(loaded.y).shape[0])
    params = build_params_for_model(
        model=model,
        user_attrs=selected_trial.user_attrs,
        flat_params=selected_trial.params_flat,
        in_channels=loaded.n_channels,
        num_classes=num_classes,
        seq_len=loaded.n_time,
    )

    ModelCls = MODEL_CLASSES[model]
    if model == "svm":
        result = train_eval_svm_outer_fold(
            ModelCls=ModelCls,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            device=device,
        )
        chosen_epoch = None
        inner_best_f1 = None
        train_time = result.training_time_s
        infer_time = result.inference_time_s
        metrics = result.metrics
        y_pred = result.y_pred
        decision = result.decision_score
        prob_open = result.probability_open_hand
    else:
        from sklearn.model_selection import train_test_split

        inner_tr_rel, inner_va_rel = train_test_split(
            np.arange(len(y_train)),
            test_size=cfg.inner_val_fraction,
            random_state=cfg.inner_val_seed + fold,
            stratify=y_train,
        )
        nn_result = train_eval_nn_outer_fold(
            ModelCls=ModelCls,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            inner_train_idx=inner_tr_rel,
            inner_val_idx=inner_va_rel,
            device=device,
            max_epochs=cfg.max_epochs,
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
            seed=cfg.seed,
        )
        chosen_epoch = nn_result.chosen_epoch
        inner_best_f1 = nn_result.inner_best_f1
        train_time = nn_result.training_time_s
        infer_time = nn_result.inference_time_s
        metrics = nn_result.metrics
        y_pred = nn_result.y_pred
        decision = nn_result.decision_score
        prob_open = nn_result.probability_open_hand

    metrics_row = {
        "patient": patient,
        "model": model,
        "fold": fold,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "best_trial_number": selected_trial.trial_number,
        "best_trial_objective": selected_trial.objective_f1,
        "best_params": selected_trial.params_flat,
        "chosen_epoch": chosen_epoch,
        "inner_val_best_f1": inner_best_f1,
        "seed": cfg.seed,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "roc_auc": metrics["roc_auc"],
        "confusion_matrix": metrics["confusion_matrix"],
        "training_time": train_time,
        "inference_time": infer_time,
    }
    save_json_atomic(fdir / "metrics.json", metrics_row)

    preds_df = predictions_to_dataframe(
        loaded.metadata,
        test_idx,
        patient=patient,
        model=model,
        fold=fold,
        y_true=y_test,
        y_pred=y_pred,
        decision_score=decision,
        probability_open_hand=prob_open,
    )
    preds_df.to_csv(fdir / "predictions.csv", index=False)

    save_json_atomic(
        fdir / "model_metadata.json",
        {
            "normalization": norm_meta,
            "optuna_storage": str(selected_trial.storage),
            "study_name": selected_trial.study_name,
            "trial_number": selected_trial.trial_number,
            "objective_f1": selected_trial.objective_f1,
            "objective_loss": selected_trial.objective_loss,
            "input_preset": cfg.input_preset,
        },
    )

    write_completed_marker(
        fdir,
        patient=patient,
        model=model,
        fold=fold,
        extra={"macro_f1": metrics["macro_f1"]},
    )
    logging.info(
        "Done %s/%s fold %s macro_f1=%.4f",
        patient,
        model,
        fold,
        metrics["macro_f1"],
    )


def aggregate_all(output_root: Path) -> None:
    fold_metrics = collect_fold_metrics(output_root)
    if fold_metrics.empty:
        return
    fold_metrics.to_csv(output_root / "all_fold_metrics.csv", index=False)
    oof = collect_oof_predictions(output_root)
    if not oof.empty:
        oof.to_csv(output_root / "all_oof_predictions.csv", index=False)
    summary = patient_model_summary(fold_metrics)
    summary.to_csv(output_root / "patient_model_summary.csv", index=False)
    stats = statistical_summary(summary)
    save_json_atomic(output_root / "statistical_summary.json", stats)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    cfg = merge_config(args)

    if cfg.output_root is None:
        logging.error("output.root must be set in config or via --output-root")
        return 2

    output_root = cfg.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        return run_dry_run(cfg, _PROJECT_ROOT, fold_subset=args.folds)

    issues = validate_config(cfg, dry_run=False)
    if issues:
        for issue in issues:
            logging.error(issue)
        return 2

    set_global_seeds(cfg.seed)
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA unavailable; falling back to CPU")
        device = "cpu"

    save_run_artifacts(output_root, cfg=cfg, args=args, project_root=_PROJECT_ROOT)

    progress_path = output_root / "progress.json"
    progress: dict[str, Any] = {"completed": [], "failed": []}
    if progress_path.is_file():
        with open(progress_path, encoding="utf-8") as fh:
            progress = json.load(fh)

    had_failures = False
    selected_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for patient in cfg.patients:
        try:
            tfr_path = cfg.resolve_tfr_path(patient, _PROJECT_ROOT)
            preset = cfg.get_input_preset()
            loaded = load_tfr_xy_metadata(
                tfr_path,
                patient=patient,
                event_pos_code=cfg.event_pos_code,
                preset=preset,
                sample_manifest=cfg.sample_manifest,
            )
            splits = make_outer_folds(
                loaded.y,
                n_splits=cfg.n_folds,
                seed=cfg.cv_seed,
                fold_subset=args.folds,
            )
            fold_table = build_fold_assignment_table(loaded.metadata, splits)
            patient_fold_path = output_root / patient / "fold_assignments.csv"
            patient_fold_path.parent.mkdir(parents=True, exist_ok=True)
            fold_table.to_csv(patient_fold_path, index=False)
            root_fold_path = output_root / "fold_assignments.csv"
            if root_fold_path.is_file():
                existing = pd.read_csv(root_fold_path)
                if "patient" in existing.columns:
                    existing = existing[existing["patient"].astype(str) != patient]
                fold_table = pd.concat([existing, fold_table], ignore_index=True)
            fold_table.to_csv(root_fold_path, index=False)

            for model in cfg.models:
                ref = discover_study_path(
                    cfg.optuna_run_dir or Path("."),
                    patient,
                    model,
                    explicit_studies=cfg.studies,
                )
                audit_rows.append(audit_study_cv_mode(ref).__dict__)
                selected = load_selected_trial(ref, selector=cfg.trial_selector)
                selected_rows.append(
                    {
                        "patient": selected.patient,
                        "model": selected.model,
                        "storage": str(selected.storage),
                        "study_name": selected.study_name,
                        "trial_number": selected.trial_number,
                        "objective_f1": selected.objective_f1,
                        "objective_loss": selected.objective_loss,
                    }
                )

                for fold_id, (train_idx, test_idx) in enumerate(splits):
                    try:
                        run_fold_job(
                            cfg=cfg,
                            patient=patient,
                            model=model,
                            fold=fold_id,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            loaded=loaded,
                            selected_trial=selected,
                            output_root=output_root,
                            device=device,
                            resume=args.resume,
                            overwrite=args.overwrite,
                        )
                        progress["completed"].append(f"{patient}/{model}/fold_{fold_id}")
                    except Exception as exc:
                        had_failures = True
                        progress["failed"].append(f"{patient}/{model}/fold_{fold_id}")
                        err_path = log_error(
                            output_root / "errors",
                            patient=patient,
                            model=model,
                            fold=fold_id,
                            exc=exc,
                        )
                        logging.exception(
                            "Failed %s/%s fold %s (logged %s)",
                            patient,
                            model,
                            fold_id,
                            err_path,
                        )
                    update_progress(progress_path, progress)

        except Exception as exc:
            had_failures = True
            log_error(
                output_root / "errors",
                patient=patient,
                model="all",
                fold=None,
                exc=exc,
            )
            logging.exception("Failed patient %s", patient)

    if selected_rows:
        pd.DataFrame(selected_rows).to_csv(output_root / "selected_trials.csv", index=False)
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(output_root / "optuna_cv_audit.csv", index=False)

    aggregate_all(output_root)
    return 1 if had_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
