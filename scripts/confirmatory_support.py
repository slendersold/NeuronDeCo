"""Helpers for scripts/run_confirmatory_analysis.py (local to scripts/, not lib/)."""

from __future__ import annotations

import gc
import json
import platform
import random
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Type

import mne
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import yaml
from optuna.trial import FrozenTrial, TrialState
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from lib.data.normalisation import apply_tfr_robust_norm, fit_tfr_robust_stats
from lib.data.tfr_dataset import TFRDataset
from lib.models.tfr_transformer.preprocess import PREPROCESS_BUILDERS, SeqPool
from lib.optuna.study_io import load_study_sqlite
from lib.optuna.types import Params
from lib.training.epochs import eval_one_epoch_f1_macro, train_one_epoch
from lib.training.svm_epochs import train_one_epoch_svm

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None


DEFAULT_SUBJECTS = [
    "s02", "s03", "s04", "s05", "s06", "s07", "s09", "s10", "s11", "s12", "s13", "s15",
]

MODEL_DB_SUFFIX = {
    "alexnet": "alexnet",
    "transformer": "transformer",
    "svm": "tfr_svm",
}


@dataclass(frozen=True)
class InputPresetSpec:
    """TFR loading parameters for a named input representation."""

    name: str
    crop_tmin: float
    crop_tmax: float
    freq_slice: str
    description: str


INPUT_PRESETS: dict[str, InputPresetSpec] = {
    "optuna_original": InputPresetSpec(
        name="optuna_original",
        crop_tmin=0.0,
        crop_tmax=1.0,
        freq_slice="drop_last_50",
        description="Marker-relative crop [0,1]s; drop last 50 frequency bins (Optuna default).",
    ),
    "compact_ablation": InputPresetSpec(
        name="compact_ablation",
        crop_tmin=0.0,
        crop_tmax=1.0,
        freq_slice="band_0p1_30",
        description="Optional sensitivity preset: marker [0,1]s; keep 0.1-30 Hz only.",
    ),
}


@dataclass
class ConfirmatoryConfig:
    """Runtime configuration merged from YAML and CLI overrides."""

    optuna_run_dir: Path | None = None
    preprocessed_root: Path | None = None
    tfr_subdir: str = "specs_with_car"
    tfr_pattern: str = "tfr_{patient}.fif"
    external_config: Path | None = None
    sample_manifest: Path | None = None
    output_root: Path | None = None

    patients: list[str] = field(default_factory=lambda: list(DEFAULT_SUBJECTS))
    models: list[str] = field(default_factory=lambda: ["svm", "alexnet", "transformer"])

    input_preset: str = "optuna_original"
    event_pos_code: int = 9
    n_folds: int = 5
    cv_seed: int = 42

    inner_val_fraction: float = 0.15
    inner_val_seed: int = 42
    max_epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    device: str = "cuda"
    num_workers: int = 0
    n_jobs: int = 1
    seed: int = 42

    trial_selector: str = "top5_f1_then_min_loss"
    studies: list[dict[str, Any]] = field(default_factory=list)

    def resolve_tfr_path(self, patient: str, project_root: Path) -> Path:
        """Resolve ``tfr_{patient}.fif`` under preprocessed root."""
        if self.preprocessed_root is None:
            raise ValueError("preprocessed_root is not set")
        name = self.tfr_pattern.format(patient=patient)
        candidates = [
            self.preprocessed_root / self.tfr_subdir / name,
            self.preprocessed_root / name,
            project_root / self.tfr_subdir / name,
        ]
        for path in candidates:
            if path.exists():
                return path.resolve()
        raise FileNotFoundError(
            f"TFR not found for {patient!r}. Checked: "
            + ", ".join(str(p) for p in candidates)
        )

    def get_input_preset(self) -> InputPresetSpec:
        if self.input_preset not in INPUT_PRESETS:
            known = ", ".join(sorted(INPUT_PRESETS))
            raise ValueError(f"Unknown input_preset={self.input_preset!r}. Known: {known}")
        return INPUT_PRESETS[self.input_preset]


def _expand_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value)).expanduser()


def load_confirmatory_config(path: Path) -> ConfirmatoryConfig:
    """Load YAML config into :class:`ConfirmatoryConfig`."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    optuna = raw.get("optuna", {}) or {}
    data = raw.get("data", {}) or {}
    output = raw.get("output", {}) or {}
    cv = raw.get("cv", {}) or {}
    training = raw.get("training", {}) or {}
    runtime = raw.get("runtime", {}) or {}
    trial = raw.get("trial_selection", {}) or {}

    cfg = ConfirmatoryConfig(
        optuna_run_dir=_expand_path(optuna.get("run_dir")),
        preprocessed_root=_expand_path(data.get("preprocessed_root")),
        tfr_subdir=str(data.get("tfr_subdir", "specs_with_car")),
        tfr_pattern=str(data.get("tfr_pattern", "tfr_{patient}.fif")),
        external_config=_expand_path(data.get("external_config")),
        sample_manifest=_expand_path(data.get("sample_manifest")),
        output_root=_expand_path(output.get("root")),
        input_preset=str(raw.get("input_preset", "optuna_original")),
        event_pos_code=int(raw.get("event_pos_code", 9)),
        n_folds=int(cv.get("n_splits", 5)),
        cv_seed=int(cv.get("seed", 42)),
        inner_val_fraction=float(training.get("inner_val_fraction", 0.15)),
        inner_val_seed=int(training.get("inner_val_seed", 42)),
        max_epochs=int(training.get("max_epochs", 100)),
        early_stopping_patience=int(training.get("early_stopping_patience", 10)),
        early_stopping_min_delta=float(training.get("early_stopping_min_delta", 0.001)),
        device=str(runtime.get("device", "cuda")),
        num_workers=int(runtime.get("num_workers", 0)),
        n_jobs=int(runtime.get("n_jobs", 1)),
        seed=int(runtime.get("seed", 42)),
        trial_selector=str(trial.get("selector", "top5_f1_then_min_loss")),
        studies=list(raw.get("studies", []) or []),
    )

    if "patients" in raw:
        cfg.patients = [str(p) for p in raw["patients"]]
    if "models" in raw:
        cfg.models = [str(m) for m in raw["models"]]

    return cfg


def paths_checklist(cfg: ConfirmatoryConfig) -> list[tuple[str, Path | None, str]]:
    """Human-readable list of paths the user should fill before running."""
    items: list[tuple[str, Path | None, str]] = [
        ("optuna.run_dir", cfg.optuna_run_dir, "Directory with tfr_<patient>_<model>.db files"),
        (
            "data.preprocessed_root",
            cfg.preprocessed_root,
            "Root containing specs_with_car/tfr_<patient>.fif",
        ),
        ("data.external_config", cfg.external_config, "External config.py with best_ch_by_power"),
        (
            "data.sample_manifest",
            cfg.sample_manifest,
            "Optional sample_manifest.csv for session/epoch_id enrichment",
        ),
        ("output.root", cfg.output_root, "Confirmatory results directory"),
    ]
    return items

# ----- data.py -----


@dataclass(frozen=True)
class TfrLoadResult:
    """Loaded TFR tensors and per-epoch metadata before normalization."""

    X_raw: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    tfr_path: Path
    patient: str
    n_channels: int
    n_freqs: int
    n_time: int
    freq_hz: np.ndarray | None
    time_s: np.ndarray | None


def _apply_freq_slice(X: np.ndarray, preset: InputPresetSpec, freqs: np.ndarray | None) -> np.ndarray:
    if preset.freq_slice == "drop_last_50":
        if X.shape[2] <= 50:
            raise ValueError(f"Cannot drop last 50 freq bins from F={X.shape[2]}")
        return X[:, :, :-50, :]
    if preset.freq_slice == "band_0p1_30":
        if freqs is None:
            raise ValueError("compact_ablation preset requires frequency axis from TFR")
        mask = (freqs >= 0.1) & (freqs <= 30.0)
        if not np.any(mask):
            raise ValueError("No frequency bins in [0.1, 30] Hz")
        return X[:, :, mask, :]
    raise ValueError(f"Unsupported freq_slice={preset.freq_slice!r}")


def load_tfr_xy_metadata(
    tfr_path: Path,
    *,
    patient: str,
    event_pos_code: int,
    preset: InputPresetSpec,
    sample_manifest: Path | None = None,
) -> TfrLoadResult:
    """
    Load TFR epochs in the same order as Optuna ``load_xy``, with metadata.

  Order of epochs follows ``tfr.events`` row order after MNE ``read_tfrs`` /
  ``crop`` — identical to ``load_xy`` when using the same preset.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*does not conform to MNE naming conventions.*",
            category=RuntimeWarning,
        )
        tfr_list = mne.time_frequency.read_tfrs(str(tfr_path))
    if not tfr_list:
        raise ValueError("read_tfrs returned empty list")
    tfr = tfr_list[0]
    if getattr(tfr, "events", None) is None:
        raise ValueError("TFR has no events attribute")

    events = np.asarray(tfr.events)
    y = np.where(events[:, 2] == event_pos_code, 1, 0).astype(np.int64)

    freqs = np.asarray(tfr.freqs, dtype=float) if hasattr(tfr, "freqs") else None
    times = np.asarray(tfr.times, dtype=float) if hasattr(tfr, "times") else None

    tfr = tfr.crop(tmin=preset.crop_tmin, tmax=preset.crop_tmax)
    X_full = np.asarray(tfr.data, dtype=np.float32)
    if X_full.ndim != 4:
        raise ValueError(f"Unexpected tfr.data shape: {X_full.shape}")

    if freqs is not None and len(freqs) != X_full.shape[2]:
        freqs = freqs[: X_full.shape[2]]
    if times is not None and len(times) != X_full.shape[3]:
        times = times[: X_full.shape[3]]

    X_raw = _apply_freq_slice(X_full, preset, freqs).astype(np.float32)
    n_epochs = X_raw.shape[0]

    meta = pd.DataFrame(
        {
            "patient": patient,
            "sample_index": np.arange(n_epochs, dtype=int),
            "event_code": events[:n_epochs, 2].astype(int),
            "mne_sample": events[:n_epochs, 0].astype(int),
            "class_label": y,
            "stable_epoch_id": [
                f"{patient}::idx{idx:05d}::sample{int(events[idx, 0])}::code{int(events[idx, 2])}"
                for idx in range(n_epochs)
            ],
            "session": "",
        }
    )

    if sample_manifest is not None and sample_manifest.is_file():
        manifest = pd.read_csv(sample_manifest)
        if "epoch_id" in manifest.columns and len(manifest) == n_epochs:
            meta["manifest_epoch_id"] = manifest["epoch_id"].astype(str).values
            meta["stable_epoch_id"] = meta["manifest_epoch_id"]
        if "session" in manifest.columns and len(manifest) == n_epochs:
            meta["session"] = manifest["session"].astype(str).values

    del tfr, tfr_list, X_full
    gc.collect()

    _n, c, f, t = X_raw.shape
    return TfrLoadResult(
        X_raw=X_raw,
        y=y,
        metadata=meta,
        tfr_path=tfr_path.resolve(),
        patient=patient,
        n_channels=c,
        n_freqs=f,
        n_time=t,
        freq_hz=freqs,
        time_s=times,
    )


def normalize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit robust stats on train only; transform train and test."""
    median, iqr = fit_tfr_robust_stats(X_train, eps=eps)
    X_tr = apply_tfr_robust_norm(X_train, median, iqr).astype(np.float32)
    X_te = apply_tfr_robust_norm(X_test, median, iqr).astype(np.float32)
    meta = {
        "normalization": "robust_median_iqr",
        "fit_on": "outer_train",
        "eps": eps,
    }
    return X_tr, X_te, meta


def make_outer_folds(
    y: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    fold_subset: list[int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold outer splits as (train_idx, test_idx) arrays."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(y)), y))
    if fold_subset is not None:
        allowed = set(fold_subset)
        splits = [s for i, s in enumerate(splits) if i in allowed]
    return splits


def make_inner_split(
    train_idx: np.ndarray,
    y: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified inner train/validation split inside outer-train indices."""
    y_sub = y[train_idx]
    inner_train_rel, inner_val_rel = train_test_split(
        np.arange(len(train_idx)),
        test_size=val_fraction,
        random_state=seed,
        stratify=y_sub,
    )
    return train_idx[inner_train_rel], train_idx[inner_val_rel]


def build_fold_assignment_table(
    metadata: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Assign each sample_index to exactly one outer validation fold."""
    out = metadata.copy()
    fold_col = np.full(len(out), -1, dtype=int)
    for fold_id, (_tr, te) in enumerate(splits):
        fold_col[te] = fold_id
    if np.any(fold_col < 0):
        raise RuntimeError("Incomplete fold assignment")
    out["fold"] = fold_col
    return out

# ----- optuna_trials.py -----


@dataclass(frozen=True)
class StudyRef:
    patient: str
    model: str
    storage: Path
    study_name: str


@dataclass(frozen=True)
class SelectedTrial:
    patient: str
    model: str
    storage: Path
    study_name: str
    trial_number: int
    objective_f1: float
    objective_loss: float
    params_flat: dict[str, Any]
    user_attrs: dict[str, Any]


@dataclass(frozen=True)
class CvModeAuditRow:
    patient: str
    model: str
    storage: str
    study_name: str
    cv_mode: str
    number_of_folds: int | None
    evidence_source: str


def model_to_db_suffix(model: str) -> str:
    if model not in MODEL_DB_SUFFIX:
        raise ValueError(f"Unknown model={model!r}. Expected one of {sorted(MODEL_DB_SUFFIX)}")
    return MODEL_DB_SUFFIX[model]


def discover_study_path(
    run_dir: Path,
    patient: str,
    model: str,
    *,
    explicit_studies: list[dict[str, Any]] | None = None,
) -> StudyRef:
    """Resolve SQLite path and study name for patient × model."""
    if explicit_studies:
        for row in explicit_studies:
            if row.get("patient") == patient and row.get("model") == model:
                storage = Path(str(row["storage"])).expanduser().resolve()
                study_name = str(row.get("study_name", storage.stem))
                return StudyRef(patient, model, storage, study_name)

    suffix = model_to_db_suffix(model)
    stem = f"tfr_{patient}_{suffix}"
    storage = (run_dir / f"{stem}.db").resolve()
    return StudyRef(patient, model, storage, stem)


def select_best_trial(study: optuna.Study, *, selector: str = "top5_f1_then_min_loss") -> FrozenTrial | None:
    """Select best completed trial from a multi-objective study."""
    complete = [
        t
        for t in study.get_trials(deepcopy=False)
        if t.state == TrialState.COMPLETE and t.values is not None and len(t.values) >= 2
    ]
    if not complete:
        return None

    if selector == "top5_f1_then_min_loss":
        ranked = sorted(complete, key=lambda t: float(t.values[0]), reverse=True)
        top5 = ranked[:5]
        return min(top5, key=lambda t: float(t.values[1]))

    if selector == "max_f1":
        return max(complete, key=lambda t: float(t.values[0]))

    raise ValueError(f"Unknown trial selector: {selector!r}")


def load_selected_trial(ref: StudyRef, *, selector: str) -> SelectedTrial:
    """Load study and return selected trial metadata."""
    if not ref.storage.is_file():
        raise FileNotFoundError(f"Optuna database not found: {ref.storage}")
    study = load_study_sqlite(db_path=ref.storage, study_name=ref.study_name)
    trial = select_best_trial(study, selector=selector)
    if trial is None:
        raise RuntimeError(f"No complete trials in {ref.storage}")

    return SelectedTrial(
        patient=ref.patient,
        model=ref.model,
        storage=ref.storage,
        study_name=ref.study_name,
        trial_number=int(trial.number),
        objective_f1=float(trial.values[0]),
        objective_loss=float(trial.values[1]),
        params_flat=dict(trial.params),
        user_attrs=dict(trial.user_attrs),
    )


def _fold_count_from_trial(trial: FrozenTrial) -> int | None:
    curves = trial.user_attrs.get("fold_curves")
    if isinstance(curves, list) and curves:
        return len(curves)
    mode = trial.user_attrs.get("cv_mode")
    if mode == "holdout":
        return 1
    if mode == "kfold" and isinstance(curves, list):
        return len(curves)
    return None


def audit_study_cv_mode(ref: StudyRef) -> CvModeAuditRow:
    """
    Infer historical CV mode from trial user_attrs and run reports.

    Returns ``cv_mode=unknown`` when evidence is insufficient.
    """
    evidence: list[str] = []
    cv_modes: list[str] = []
    fold_counts: list[int] = []

    if ref.storage.is_file():
        study = load_study_sqlite(db_path=ref.storage, study_name=ref.study_name)
        for trial in study.get_trials(deepcopy=False):
            if trial.state != TrialState.COMPLETE:
                continue
            mode = trial.user_attrs.get("cv_mode")
            if isinstance(mode, str):
                cv_modes.append(mode)
                evidence.append("trial.user_attrs.cv_mode")
            fc = _fold_count_from_trial(trial)
            if fc is not None:
                fold_counts.append(fc)
                evidence.append("trial.user_attrs.fold_curves")

    run_dir = ref.storage.parent
    for report_name in (
        "_ALEXNET_RUN_REPORT.json",
        "_TRANSFORMER_RUN_REPORT.json",
        "_TFR_SVM_RUN_REPORT.json",
    ):
        report_path = run_dir / report_name
        if report_path.is_file():
            try:
                with open(report_path, encoding="utf-8") as fh:
                    report = json.load(fh)
                if "cv" in report:
                    cv_modes.append("kfold" if report["cv"] else "holdout")
                    evidence.append(report_name)
            except (json.JSONDecodeError, OSError):
                pass

    if cv_modes:
        mode_counter = Counter(cv_modes)
        cv_mode = mode_counter.most_common(1)[0][0]
    else:
        cv_mode = "unknown"

    n_folds: int | None = None
    if fold_counts:
        fold_counter = Counter(fold_counts)
        n_folds = fold_counter.most_common(1)[0][0]
    elif cv_mode == "holdout":
        n_folds = 1

    return CvModeAuditRow(
        patient=ref.patient,
        model=ref.model,
        storage=str(ref.storage),
        study_name=ref.study_name,
        cv_mode=cv_mode,
        number_of_folds=n_folds,
        evidence_source="; ".join(sorted(set(evidence))) or "none",
    )


def scan_run_dir(run_dir: Path) -> list[StudyRef]:
    """List all ``tfr_<patient>_<method>.db`` studies under a run directory."""
    refs: list[StudyRef] = []
    if not run_dir.is_dir():
        return refs
    for db_path in sorted(run_dir.glob("tfr_*_*.db")):
        stem = db_path.stem
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        patient = parts[1]
        method_suffix = "_".join(parts[2:])
        model = next(
            (m for m, suf in MODEL_DB_SUFFIX.items() if suf == method_suffix),
            method_suffix,
        )
        refs.append(StudyRef(patient, model, db_path.resolve(), stem))
    return refs

# ----- params.py -----


def _rebuild_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "type" in value:
            t = value["type"]
            if t == "SeqPool":
                return SeqPool(mode=str(value["mode"]))
            if t == "TFRToSeqFlatten":
                return PREPROCESS_BUILDERS["flatten"]()
            if t == "TFRToSeqChannelConvCollapse":
                from lib.models.tfr_transformer.preprocess import TFRToSeqChannelConvCollapse

                return TFRToSeqChannelConvCollapse(bias=bool(value.get("bias", True)))
            if t == "TFRToSeqFTPlaneConvCollapse":
                from lib.models.tfr_transformer.preprocess import TFRToSeqFTPlaneConvCollapse

                return TFRToSeqFTPlaneConvCollapse(
                    kernel_freq=int(value.get("kernel_freq", 3)),
                    kernel_time=int(value.get("kernel_time", 3)),
                    bias=bool(value.get("bias", True)),
                )
            if t == "TFRToSeqPixelWeightCollapse":
                return PREPROCESS_BUILDERS["pixel_weight"]()
        return {k: _rebuild_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rebuild_json_value(v) for v in value]
    return value


def _finalize_model_kwargs(
    model_name: str,
    model: dict[str, Any],
    *,
    in_channels: int,
    num_classes: int,
    seq_len: int,
) -> dict[str, Any]:
    """Keep only constructor kwargs valid for the target model class."""
    out = dict(model)
    out["num_classes"] = int(num_classes)

    if model_name == "svm":
        out.pop("in_channels", None)
        out.pop("seq_len", None)
        if "probability" not in out:
            out["probability"] = False
        allowed = {
            "num_classes",
            "preprocess",
            "pooling",
            "svm_C",
            "kernel",
            "svm_gamma",
            "probability",
        }
        return {k: v for k, v in out.items() if k in allowed}

    if model_name == "alexnet":
        out["in_channels"] = int(in_channels)
        out.pop("seq_len", None)
        allowed = {"in_channels", "num_classes", "dropout"}
        return {k: v for k, v in out.items() if k in allowed}

    if model_name == "transformer":
        out["seq_len"] = int(seq_len)
        out.pop("in_channels", None)
        return out

    raise ValueError(f"Unsupported model: {model_name!r}")


def params_from_user_attrs(
    user_attrs: Mapping[str, Any],
    *,
    model_name: str,
    in_channels: int,
    num_classes: int,
    seq_len: int,
) -> Params | None:
    """Deserialize nested ``user_attrs['params']`` saved by Optuna ``attrs_fn``."""
    raw = user_attrs.get("params")
    if not isinstance(raw, dict):
        return None
    bundle = _rebuild_json_value(raw)
    if not isinstance(bundle, dict):
        return None

    model = _finalize_model_kwargs(
        model_name,
        dict(bundle.get("model", {})),
        in_channels=in_channels,
        num_classes=num_classes,
        seq_len=seq_len,
    )

    return {
        "model": model,
        "optimizer": dict(bundle.get("optimizer", {})),
        "tr_dataset": dict(bundle.get("tr_dataset", {"time_crop": None})),
        "vl_dataset": dict(bundle.get("vl_dataset", {"time_crop": None})),
        "tr_loader": dict(bundle.get("tr_loader", {})),
        "vl_loader": dict(bundle.get("vl_loader", {})),
    }


def params_from_flat_alexnet(
    flat: Mapping[str, Any],
    *,
    in_channels: int,
    num_classes: int,
) -> Params:
    bs = int(flat["batch_size"])
    return {
        "model": {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "dropout": float(flat["dropout"]),
        },
        "optimizer": {
            "lr": float(flat["lr"]),
            "weight_decay": float(flat["weight_decay"]),
        },
        "tr_dataset": {"time_crop": None},
        "tr_loader": {"batch_size": bs, "shuffle": True},
        "vl_dataset": {"time_crop": None},
        "vl_loader": {"batch_size": bs, "shuffle": False},
    }


def params_from_flat_transformer(
    flat: Mapping[str, Any],
    *,
    num_classes: int,
    seq_len: int,
) -> Params:
    pre_name = str(flat["preprocess"])
    bs = int(flat["batch_size"])
    return {
        "model": {
            "num_classes": num_classes,
            "seq_len": seq_len,
            "embed_dim": int(flat["embed_dim"]),
            "nhead": int(flat["nhead"]),
            "dim_fc": int(flat["dim_fc"]),
            "num_layers": int(flat["num_layers"]),
            "dropout": float(flat["dropout"]),
            "encoder_dropout": float(flat["encoder_dropout"]),
            "mlp_dropout": float(flat["mlp_dropout"]),
            "use_conv": bool(flat["use_conv"]),
            "conv_kernel_size": int(flat["conv_kernel_size"]),
            "conv_dropout": float(flat["conv_dropout"]),
            "pooling": SeqPool(mode=str(flat["pooling"])),
            "preprocess": PREPROCESS_BUILDERS[pre_name](),
        },
        "optimizer": {
            "lr": float(flat["lr"]),
            "weight_decay": float(flat["weight_decay"]),
        },
        "tr_dataset": {"time_crop": None},
        "tr_loader": {"batch_size": bs, "shuffle": True},
        "vl_dataset": {"time_crop": None},
        "vl_loader": {"batch_size": bs, "shuffle": False},
    }


def params_from_flat_svm(
    flat: Mapping[str, Any],
    *,
    num_classes: int,
) -> Params:
    pre_name = str(flat["preprocess"])
    kernel = str(flat["kernel"])
    bs = int(flat["batch_size"])
    svm_gamma: float | str
    if kernel == "rbf":
        svm_gamma = float(flat["svm_gamma"])
    else:
        svm_gamma = "scale"
    return {
        "model": {
            "num_classes": num_classes,
            "preprocess": PREPROCESS_BUILDERS[pre_name](),
            "pooling": SeqPool(mode=str(flat["pooling"])),
            "kernel": kernel,
            "svm_C": float(flat["svm_C"]),
            "svm_gamma": svm_gamma,
            "probability": False,
        },
        "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
        "tr_dataset": {"time_crop": None},
        "tr_loader": {"batch_size": bs, "shuffle": True},
        "vl_dataset": {"time_crop": None},
        "vl_loader": {"batch_size": bs, "shuffle": False},
    }


def build_params_for_model(
    *,
    model: str,
    user_attrs: Mapping[str, Any],
    flat_params: Mapping[str, Any],
    in_channels: int,
    num_classes: int,
    seq_len: int,
) -> Params:
    """Prefer deserialized user_attrs; fall back to flat trial.params on failure."""
    try:
        from_attrs = params_from_user_attrs(
            user_attrs,
            model_name=model,
            in_channels=in_channels,
            num_classes=num_classes,
            seq_len=seq_len,
        )
        if from_attrs is not None:
            return from_attrs
    except (TypeError, ValueError, KeyError):
        pass

    if model == "alexnet":
        return params_from_flat_alexnet(
            flat_params, in_channels=in_channels, num_classes=num_classes
        )
    if model == "transformer":
        return params_from_flat_transformer(
            flat_params, num_classes=num_classes, seq_len=seq_len
        )
    if model == "svm":
        return params_from_flat_svm(flat_params, num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model!r}")

# ----- metrics.py -----


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_score: np.ndarray,
) -> dict[str, Any]:
    """Compute standard binary classification metrics."""
    try:
        auc = float(roc_auc_score(y_true, decision_score))
    except ValueError:
        auc = float("nan")

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


@torch.no_grad()
def predict_torch_classifier(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    device: torch.device | str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run batched inference; return y_pred, decision_score, probability_open_hand.

    ``decision_score`` is logit margin for class 1 when logits are available.
    """
    model.eval()
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.zeros(len(X), dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    probs: list[np.ndarray] = []

    for xb, _ in loader:
        xb = xb.to(device)
        out = model(xb)
        if out.shape[-1] == 1:
            logits = out.squeeze(-1)
            prob_pos = torch.sigmoid(logits)
            pred = (prob_pos >= 0.5).long()
            score = logits
        else:
            logits = out
            prob = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            score = logits[:, 1] - logits[:, 0]
            prob_pos = prob[:, 1]
        preds.append(pred.cpu().numpy())
        scores.append(score.cpu().numpy())
        probs.append(prob_pos.cpu().numpy())

    y_pred = np.concatenate(preds).astype(int)
    decision = np.concatenate(scores).astype(float)
    prob_open = np.concatenate(probs).astype(float)
    return y_pred, decision, prob_open


def predictions_to_dataframe(
    metadata: pd.DataFrame,
    test_idx: np.ndarray,
    *,
    patient: str,
    model: str,
    fold: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_score: np.ndarray,
    probability_open_hand: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build OOF prediction rows aligned across models."""
    sub = metadata.iloc[test_idx].copy().reset_index(drop=True)
    out = pd.DataFrame(
        {
            "patient": patient,
            "model": model,
            "fold": fold,
            "epoch_id": sub["stable_epoch_id"].astype(str).values,
            "sample_index": sub["sample_index"].astype(int).values,
            "session": sub.get("session", "").astype(str).values,
            "true_label": y_true.astype(int),
            "predicted_label": y_pred.astype(int),
            "decision_score": decision_score.astype(float),
            "probability_open_hand": (
                probability_open_hand.astype(float)
                if probability_open_hand is not None
                else np.nan
            ),
        }
    )
    return out

# ----- train_nn.py -----


@dataclass(frozen=True)
class NnFoldResult:
    chosen_epoch: int
    inner_best_f1: float
    metrics: dict[str, Any]
    y_pred: np.ndarray
    decision_score: np.ndarray
    probability_open_hand: np.ndarray
    training_time_s: float
    inference_time_s: float
    inner_curves: dict[str, list[float]]


def _run_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    max_epochs: int,
) -> tuple[list[float], list[float], list[float]]:
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_f1s: list[float] = []
    for _ in range(max_epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_f1 = eval_one_epoch_f1_macro(model, val_loader, device)
        train_losses.append(float(tr_loss))
        val_losses.append(float(va_loss))
        val_f1s.append(float(va_f1))
    return train_losses, val_losses, val_f1s


def _early_stop_epoch(
    val_f1s: list[float],
    *,
    patience: int,
    min_delta: float,
) -> int:
    """Return 1-based epoch count to train (at least 1)."""
    best_f1 = -1.0
    best_epoch = 1
    bad = 0
    for i, f1 in enumerate(val_f1s, start=1):
        if f1 > best_f1 + min_delta:
            best_f1 = float(f1)
            best_epoch = i
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return max(1, best_epoch)


def train_eval_nn_outer_fold(
    *,
    ModelCls: Type[nn.Module],
    params: Params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    inner_train_idx: np.ndarray,
    inner_val_idx: np.ndarray,
    device: torch.device | str,
    max_epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
) -> NnFoldResult:
    """
    Inner-val early stopping, retrain on full outer-train, evaluate once on outer-test.
    """
    del seed  # deterministic flags set at run level
    p_model = params["model"]
    p_optim = params["optimizer"]
    p_tr_ds = params["tr_dataset"]
    p_vl_ds = params["vl_dataset"]
    p_tr_ld = params["tr_loader"]
    p_vl_ld = params["vl_loader"]
    batch_size = int(p_tr_ld["batch_size"])

    t0 = time.perf_counter()

    # Phase 1 — model selection on inner split
    model_es = ModelCls(**p_model).to(device)
    opt_es = torch.optim.AdamW(model_es.parameters(), **p_optim)
    inner_tr = DataLoader(
        TFRDataset(X_train[inner_train_idx], y_train[inner_train_idx], **p_tr_ds),
        **p_tr_ld,
    )
    inner_va = DataLoader(
        TFRDataset(X_train[inner_val_idx], y_train[inner_val_idx], **p_vl_ds),
        **p_vl_ld,
    )
    tr_losses, va_losses, va_f1s = _run_epochs(
        model_es,
        inner_tr,
        inner_va,
        opt_es,
        device,
        max_epochs=max_epochs,
    )
    chosen_epoch = _early_stop_epoch(va_f1s, patience=patience, min_delta=min_delta)
    inner_best_f1 = float(max(va_f1s[:chosen_epoch]))

    # Phase 2 — retrain on full outer-train
    model_final = ModelCls(**p_model).to(device)
    opt_final = torch.optim.AdamW(model_final.parameters(), **p_optim)
    outer_tr = DataLoader(
        TFRDataset(X_train, y_train, **p_tr_ds),
        **p_tr_ld,
    )
    for _ in range(chosen_epoch):
        train_one_epoch(model_final, outer_tr, opt_final, device)

    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred, decision, prob_open = predict_torch_classifier(
        model_final,
        X_test,
        device=device,
        batch_size=batch_size,
    )
    infer_time = time.perf_counter() - t1

    
    metrics = compute_classification_metrics(y_test, y_pred, decision)
    metrics["inner_val_best_f1"] = inner_best_f1

    return NnFoldResult(
        chosen_epoch=chosen_epoch,
        inner_best_f1=inner_best_f1,
        metrics=metrics,
        y_pred=y_pred,
        decision_score=decision,
        probability_open_hand=prob_open,
        training_time_s=float(train_time),
        inference_time_s=float(infer_time),
        inner_curves={
            "train_losses": tr_losses,
            "val_losses": va_losses,
            "val_f1s": va_f1s,
        },
    )

# ----- train_svm.py -----


@dataclass(frozen=True)
class SvmFoldResult:
    metrics: dict[str, Any]
    y_pred: np.ndarray
    decision_score: np.ndarray
    probability_open_hand: np.ndarray
    training_time_s: float
    inference_time_s: float


def train_eval_svm_outer_fold(
    *,
    ModelCls: Type[nn.Module],
    params: Params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device | str,
) -> SvmFoldResult:
    """Fit TfrParadigmSvmClassifier on outer-train; evaluate on outer-test."""
    p_model = params["model"]
    p_tr_ds = params["tr_dataset"]
    p_tr_ld = params["tr_loader"]

    model = ModelCls(**p_model).to(device)

    train_loader = DataLoader(
        TFRDataset(X_train, y_train, **p_tr_ds),
        **p_tr_ld,
    )

    t0 = time.perf_counter()
    train_one_epoch_svm(model, train_loader, None, device)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    xs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        eval_loader = DataLoader(
            TFRDataset(X_test, y_test, **params["vl_dataset"]),
            **{**params["vl_loader"], "shuffle": False},
        )
        for xb, _ in eval_loader:
            feats = model._features(xb.to(device))
            xs.append(feats.cpu().numpy())
    X_feat = np.vstack(xs)
    pipeline = model._pipeline
    if pipeline is None:
        raise RuntimeError("SVM pipeline was not fitted")
    decision = pipeline.decision_function(X_feat)
    if decision.ndim > 1:
        decision = decision[:, 0]
    y_pred = (decision >= 0.0).astype(int)
    prob_open = 1.0 / (1.0 + np.exp(-decision))

    infer_time = time.perf_counter() - t1

    metrics = compute_classification_metrics(y_test, y_pred, decision)
    return SvmFoldResult(
        metrics=metrics,
        y_pred=y_pred,
        decision_score=decision,
        probability_open_hand=prob_open,
        training_time_s=float(train_time),
        inference_time_s=float(infer_time),
    )

# ----- io.py -----


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def fold_dir(output_root: Path, patient: str, model: str, fold: int) -> Path:
    return output_root / patient / model / f"fold_{fold}"


def completed_marker_path(fold_path: Path) -> Path:
    return fold_path / "COMPLETED.json"


def is_fold_completed(fold_path: Path) -> bool:
    marker = completed_marker_path(fold_path)
    if not marker.is_file():
        return False
    try:
        data = load_json(marker)
    except (json.JSONDecodeError, OSError):
        return False
    required = {"patient", "model", "fold", "status", "completed_at"}
    if not required.issubset(data.keys()):
        return False
    if data.get("status") != "completed":
        return False
    metrics_path = fold_path / "metrics.json"
    preds_path = fold_path / "predictions.csv"
    return metrics_path.is_file() and preds_path.is_file()


def write_completed_marker(
    fold_path: Path,
    *,
    patient: str,
    model: str,
    fold: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "patient": patient,
        "model": model,
        "fold": fold,
        "status": "completed",
        "completed_at": utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    save_json_atomic(completed_marker_path(fold_path), payload)


def log_error(
    errors_dir: Path,
    *,
    patient: str,
    model: str,
    fold: int | None,
    exc: BaseException,
) -> Path:
    errors_dir.mkdir(parents=True, exist_ok=True)
    fold_part = "all" if fold is None else f"fold_{fold}"
    path = errors_dir / f"{patient}__{model}__{fold_part}.log"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(traceback.format_exc())
    return path


def update_progress(path: Path, progress: dict[str, Any]) -> None:
    progress["updated_at"] = utc_now_iso()
    save_json_atomic(path, progress)


# ----- aggregate.py -----


def collect_fold_metrics(output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(output_root.glob("*/*/fold_*/metrics.json")):
        with open(metrics_path, encoding="utf-8") as fh:
            row = json.load(fh)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def collect_oof_predictions(output_root: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for pred_path in sorted(output_root.glob("*/*/fold_*/predictions.csv")):
        parts.append(pd.read_csv(pred_path))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def patient_model_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    agg = (
        fold_metrics.groupby(["patient", "model"], as_index=False)
        .agg(
            macro_f1_median=("macro_f1", "median"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_q25=("macro_f1", lambda s: float(np.percentile(s, 25))),
            macro_f1_q75=("macro_f1", lambda s: float(np.percentile(s, 75))),
            accuracy_median=("accuracy", "median"),
            balanced_accuracy_median=("balanced_accuracy", "median"),
            roc_auc_median=("roc_auc", "median"),
            n_folds=("fold", "count"),
        )
        .sort_values(["patient", "model"])
    )
    return agg


def statistical_summary(
    summary: pd.DataFrame,
    *,
    old_results: dict[str, float] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Patient-level comparisons across models."""
    out: dict[str, Any] = {"models": sorted(summary["model"].unique().tolist())}

    pivot = summary.pivot(index="patient", columns="model", values="macro_f1_median")
    out["patient_macro_f1_median"] = pivot.reset_index().to_dict(orient="records")

    rng = np.random.default_rng(seed)
    global_stats: dict[str, Any] = {}
    for model in pivot.columns:
        vals = pivot[model].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        boot = [
            float(np.median(rng.choice(vals, size=len(vals), replace=True)))
            for _ in range(n_bootstrap)
        ]
        global_stats[model] = {
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "bootstrap_ci95": [
                float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5)),
            ],
        }
    out["global"] = global_stats

    if stats is not None and pivot.shape[1] >= 2:
        pairwise: list[dict[str, Any]] = []
        models = list(pivot.columns)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                paired = pivot[[a, b]].dropna()
                if len(paired) < 2:
                    continue
                try:
                    stat, p = stats.wilcoxon(paired[a], paired[b])
                except ValueError:
                    stat, p = float("nan"), float("nan")
                pairwise.append(
                    {
                        "model_a": a,
                        "model_b": b,
                        "n_patients": int(len(paired)),
                        "median_diff_a_minus_b": float(np.median(paired[a] - paired[b])),
                        "wilcoxon_stat": float(stat),
                        "p_value": float(p),
                    }
                )
        out["pairwise_wilcoxon"] = pairwise

        if len(models) >= 3:
            complete = pivot.dropna()
            if len(complete) >= 2:
                try:
                    f_stat, f_p = stats.friedmanchisquare(
                        *[complete[m].to_numpy() for m in models]
                    )
                    out["friedman"] = {
                        "statistic": float(f_stat),
                        "p_value": float(f_p),
                        "n_patients": int(len(complete)),
                    }
                except ValueError:
                    pass

    if old_results:
        comparison = []
        for model, old_val in old_results.items():
            if model not in global_stats:
                continue
            new_val = global_stats[model]["median"]
            comparison.append(
                {
                    "model": model,
                    "old_macro_f1_median": float(old_val),
                    "confirmatory_macro_f1_median": new_val,
                    "difference": new_val - float(old_val),
                }
            )
        out["old_vs_confirmatory"] = comparison

    return out

# ----- reproducibility.py -----


def git_commit_hash(project_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def capture_environment(project_root: Path) -> dict[str, Any]:
    """Collect versions and runtime metadata."""
    import sklearn

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "git_commit": git_commit_hash(project_root),
    }


def set_global_seeds(seed: int) -> None:
    """Set Python/NumPy/PyTorch seeds and deterministic flags where possible."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False