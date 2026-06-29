#!/usr/bin/env python3
"""
Stage 3: fixed SVM ablation for one patient.

The script compares two predefined channel groups:
    selected = best_ch_by_power[patient]
    rejected = ch_to_keep[patient] - selected

For every channel group, time window and frequency band it uses:
- the same accepted epochs;
- the same preprocessing and CAR reference;
- the same StratifiedKFold splits;
- the same fixed SVM hyperparameters.

Long runs are resumable:
- time-pooled TFR features are cached;
- each condition is checkpointed immediately after completion;
- an interrupted run can be restarted with the same command.

Python 3.10+

python stage3_fixed_svm_ablation.py \
  --patient s11 \
  --config-path "/trinity/home/t.samsonov/notebooks/Pirogov/PreprocessedData/config.py" \
  --study-def-path "/trinity/home/t.samsonov/notebooks/Pirogov/PreprocessedData/study_definition_open_vs_all" \
  --data-root "/trinity/home/t.samsonov/notebooks/Pirogov/PirogovDATA" \
  --output-storage "/trinity/home/t.samsonov/notebooks/Pirogov/PreprocessedData/stage3_fixed_svm_ablation" \
  --classification-jobs 2 \
  --tfr-jobs 2 \
  --svm-cache-mb 2048
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_TIME_WINDOWS: dict[str, tuple[float, float]] = {
    "T0_marker_0_1": (0.0, 1.0),
    "T1_robot_0_1": (0.1, 1.1),
    "T2_robot_0_0p5": (0.1, 0.6),
    "T3_robot_0_1p5": (0.1, 1.6),
    "T4_robot_0_2": (0.1, 2.1),
}

DEFAULT_FREQUENCY_BANDS: dict[str, tuple[float, float]] = {
    "F0_current_0p1_59p4": (0.1, 59.4),
    "F1_low_0p1_30": (0.1, 30.0),
    "F2_alpha_beta_8_30": (8.0, 30.0),
    "F3_low_gamma_30_59p4": (30.0, 59.4),
    "F4_full_0p1_120": (0.1, 120.0),
    "F5_upper_59p4_120": (59.4, 120.0),
}


@dataclass(frozen=True)
class Condition:
    channel_group: str
    time_id: str
    freq_id: str
    channels: tuple[str, ...]
    time_start: float
    time_end: float
    freq_start: float
    freq_end: float

    @property
    def identifier(self) -> str:
        return "__".join(
            [
                sanitize(self.channel_group),
                sanitize(self.time_id),
                sanitize(self.freq_id),
            ]
        )


@dataclass(frozen=True)
class Settings:
    patient_id: str
    config_path: Path
    study_def_path: Path
    data_root: Path
    output_storage: Path
    sessions: tuple[str, ...] | None

    svm_kernel: str
    svm_c: float
    svm_gamma: str
    svm_class_weight: str | None
    svm_cache_mb: float

    cv_n_splits: int
    cv_seed: int
    classification_jobs: int

    notch_freqs: tuple[float, ...]
    l_freq: float
    h_freq: float
    epoch_tmin: float
    epoch_tmax: float
    baseline: tuple[float, float] | None

    tfr_fmin: float
    tfr_fmax: float
    tfr_n_freqs: int
    tfr_decim: int
    tfr_batch_size: int
    tfr_n_jobs: int
    log_power_eps: float

    baseline_time_id: str
    baseline_freq_id: str
    reuse_feature_cache: bool
    overwrite_feature_cache: bool
    resume: bool
    save_svg: bool


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize(value: str) -> str:
    return (
        str(value)
        .replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
        .replace(":", "_")
    )


def save_figure(fig: plt.Figure, path_without_suffix: Path, save_svg: bool) -> None:
    ensure_dir(path_without_suffix.parent)
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=180, bbox_inches="tight")
    if save_svg:
        fig.savefig(path_without_suffix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def load_external_config(path: Path):
    path = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("stage3_external_config", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_manifest(root: Path) -> pd.DataFrame:
    root = root.expanduser()

    if root.is_dir():
        direct = root / "sample_manifest.csv"
        if direct.exists():
            return pd.read_csv(direct)

        matches = list(root.rglob("sample_manifest.csv"))
        if len(matches) == 1:
            return pd.read_csv(matches[0])
        raise FileNotFoundError(f"sample_manifest.csv not found under {root}")

    if root.suffix.lower() == ".zip":
        with zipfile.ZipFile(root, "r") as zf:
            matches = [
                name
                for name in zf.namelist()
                if name == "sample_manifest.csv" or name.endswith("/sample_manifest.csv")
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one sample_manifest.csv in {root}, found {matches}"
                )
            with zf.open(matches[0]) as stream:
                return pd.read_csv(stream)

    raise ValueError(f"Unsupported STUDY_DEF_PATH: {root}")


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )


def resolve_edf_path(row: pd.Series, data_root: Path) -> Path:
    original = Path(str(row["edf_path"]))
    if original.exists():
        return original

    session_dir = data_root / str(row["session"])
    candidate = session_dir / original.name
    if candidate.exists():
        return candidate

    matches = sorted(session_dir.glob("*.edf"))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"EDF not found for session={row['session']}. "
        f"Original={original}; candidate={candidate}; matches={matches}"
    )


def resolve_channel_names(
    requested: Iterable[str],
    available: Iterable[str],
) -> tuple[list[str], list[str]]:
    actual_by_lower = {name.lower(): name for name in available}
    resolved: list[str] = []
    missing: list[str] = []

    for name in requested:
        actual = actual_by_lower.get(str(name).lower())
        if actual is None:
            missing.append(str(name))
        elif actual not in resolved:
            resolved.append(actual)

    return resolved, missing


# -----------------------------------------------------------------------------
# EEG loading and preprocessing
# -----------------------------------------------------------------------------


def preprocess_raw(
    raw: mne.io.BaseRaw,
    candidate_channels: list[str],
    settings: Settings,
) -> mne.io.BaseRaw:
    """
    Fixed preprocessing for all ablation conditions.

    CAR is calculated once across all ch_to_keep channels. Selected/rejected
    slicing is performed only after epoching and TFR feature extraction.
    """
    out = raw.copy().pick(candidate_channels).load_data()
    nyquist = float(out.info["sfreq"]) / 2.0

    for freq in settings.notch_freqs:
        if freq < nyquist:
            out.notch_filter(
                freqs=[freq],
                method="iir",
                iir_params={"order": 2, "ftype": "butter"},
                verbose=False,
            )

    out.filter(
        l_freq=settings.l_freq,
        h_freq=min(settings.h_freq, nyquist - 1e-6),
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
        verbose=False,
    )

    data = out.get_data()
    data -= data.mean(axis=0, keepdims=True)

    car = mne.io.RawArray(data, out.info.copy(), verbose=False)
    car.set_annotations(out.annotations.copy())
    return car


def make_session_epochs(
    rows: pd.DataFrame,
    configured_channels: list[str],
    settings: Settings,
) -> tuple[mne.Epochs, dict[str, Any]]:
    rows = rows.sort_values(["sample", "annotation_index"]).reset_index(drop=True)

    edf_path = resolve_edf_path(rows.iloc[0], settings.data_root)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    actual_channels, missing = resolve_channel_names(configured_channels, raw.ch_names)
    if missing:
        warnings.warn(f"{rows['session'].iloc[0]}: missing channels {missing}")
    if not actual_channels:
        raise RuntimeError("No configured channels are present in EDF.")

    raw = preprocess_raw(raw, actual_channels, settings)

    events = np.zeros((len(rows), 3), dtype=int)
    events[:, 0] = rows["sample"].astype(int).to_numpy()
    events[:, 2] = rows["class_label"].astype(int).to_numpy() + 1

    metadata_columns = [
        "epoch_id",
        "session",
        "annotation_index",
        "event_code",
        "gesture_name",
        "class_name",
        "class_label",
        "sample",
        "marker_onset_s",
        "robot_onset_s",
    ]
    metadata = rows[metadata_columns].copy()

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"all_other_gestures": 1, "open_hand": 2},
        tmin=settings.epoch_tmin,
        tmax=settings.epoch_tmax,
        baseline=settings.baseline,
        preload=True,
        reject=None,
        metadata=metadata,
        event_repeated="drop",
        verbose=False,
    )

    audit = {
        "session": str(rows["session"].iloc[0]),
        "edf_path": str(edf_path),
        "input_rows": len(rows),
        "retained_epochs": len(epochs),
        "actual_channels": "|".join(actual_channels),
        "missing_channels": "|".join(missing),
    }
    return epochs, audit


def load_patient_epochs(
    patient_rows: pd.DataFrame,
    all_configured: list[str],
    settings: Settings,
    data_audit_dir: Path,
) -> tuple[mne.Epochs, list[str], list[dict[str, Any]]]:
    session_names = sorted(patient_rows["session"].unique())
    session_epochs: list[mne.Epochs] = []
    audit_rows: list[dict[str, Any]] = []

    for session_name in session_names:
        rows = patient_rows[patient_rows["session"] == session_name].copy()
        epochs, audit = make_session_epochs(rows, all_configured, settings)
        session_epochs.append(epochs)
        audit_rows.append(audit)
        print(f"[LOAD] {audit}", flush=True)

    channel_orders = [tuple(ep.ch_names) for ep in session_epochs]
    if len(set(channel_orders)) != 1:
        raise RuntimeError(f"Channel order differs between sessions: {channel_orders}")

    epochs = mne.concatenate_epochs(session_epochs, add_offset=True, verbose=False)

    pd.DataFrame(audit_rows).to_csv(
        data_audit_dir / "session_loading_audit.csv",
        index=False,
    )
    return epochs, session_names, audit_rows


# -----------------------------------------------------------------------------
# CV and model
# -----------------------------------------------------------------------------


def make_splits(
    labels: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    assignment = np.full(len(labels), -1, dtype=int)

    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(labels)), labels)):
        splits.append((train_idx, test_idx))
        assignment[test_idx] = fold

    if np.any(assignment < 0):
        raise RuntimeError("Incomplete fold assignment.")

    return splits, assignment


def make_model(settings: Settings) -> Pipeline:
    svc_kwargs: dict[str, Any] = {
        "C": float(settings.svm_c),
        "kernel": str(settings.svm_kernel),
        "class_weight": settings.svm_class_weight,
        "probability": False,
        "random_state": settings.cv_seed,
        "cache_size": float(settings.svm_cache_mb),
    }
    if settings.svm_kernel in {"rbf", "poly", "sigmoid"}:
        svc_kwargs["gamma"] = settings.svm_gamma

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(**svc_kwargs)),
        ]
    )


# -----------------------------------------------------------------------------
# TFR feature cache
# -----------------------------------------------------------------------------


def cache_signature(
    settings: Settings,
    channels: list[str],
    time_windows: dict[str, tuple[float, float]],
) -> str:
    payload = {
        "patient_id": settings.patient_id,
        "channels": channels,
        "epoch_tmin": settings.epoch_tmin,
        "epoch_tmax": settings.epoch_tmax,
        "baseline": settings.baseline,
        "notch_freqs": settings.notch_freqs,
        "l_freq": settings.l_freq,
        "h_freq": settings.h_freq,
        "tfr_fmin": settings.tfr_fmin,
        "tfr_fmax": settings.tfr_fmax,
        "tfr_n_freqs": settings.tfr_n_freqs,
        "tfr_decim": settings.tfr_decim,
        "time_windows": time_windows,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def extract_or_load_features(
    epochs: mne.Epochs,
    actual_channels: list[str],
    settings: Settings,
    time_windows: dict[str, tuple[float, float]],
    cache_dir: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], Path]:
    freqs = np.linspace(settings.tfr_fmin, settings.tfr_fmax, settings.tfr_n_freqs)
    n_cycles = freqs / 2.0
    tfr_times = epochs.times[:: settings.tfr_decim]

    time_masks: dict[str, np.ndarray] = {}
    for time_id, (start, stop) in time_windows.items():
        mask = (tfr_times >= start) & (tfr_times <= stop)
        if not np.any(mask):
            raise RuntimeError(
                f"No TFR points for {time_id}: {start}..{stop}; "
                f"available={tfr_times[0]}..{tfr_times[-1]}"
            )
        time_masks[time_id] = mask

    signature = cache_signature(settings, actual_channels, time_windows)
    cache_path = cache_dir / f"{settings.patient_id}__features__{signature}.npz"

    use_existing = (
        settings.reuse_feature_cache
        and not settings.overwrite_feature_cache
        and cache_path.exists()
    )

    if use_existing:
        print(f"[CACHE] Loading {cache_path}", flush=True)
        cache = np.load(cache_path, allow_pickle=False)
        cached_channels = list(cache["channels"].astype(str))
        if cached_channels != actual_channels:
            raise RuntimeError(
                f"Cache channels differ: {cached_channels} != {actual_channels}"
            )
        if not np.allclose(cache["freqs"], freqs):
            raise RuntimeError("Cache frequencies differ from current settings.")
        features = {
            time_id: cache[f"features__{time_id}"]
            for time_id in time_windows
        }
        return freqs, features, cache_path

    n_epochs = len(epochs)
    n_channels = len(actual_channels)
    features = {
        time_id: np.empty(
            (n_epochs, n_channels, len(freqs)),
            dtype=np.float32,
        )
        for time_id in time_windows
    }

    started = time.time()

    for channel_index, channel_name in enumerate(actual_channels):
        print(
            f"[TFR] channel {channel_index + 1}/{n_channels}: {channel_name}",
            flush=True,
        )
        channel_data = epochs.get_data(picks=[channel_name]).astype(
            np.float32,
            copy=False,
        )

        for batch_start in range(0, n_epochs, settings.tfr_batch_size):
            batch_stop = min(batch_start + settings.tfr_batch_size, n_epochs)
            batch = channel_data[batch_start:batch_stop]

            power = mne.time_frequency.tfr_array_morlet(
                batch,
                sfreq=float(epochs.info["sfreq"]),
                freqs=freqs,
                n_cycles=n_cycles,
                output="power",
                decim=settings.tfr_decim,
                n_jobs=settings.tfr_n_jobs,
                verbose=False,
            )[:, 0]

            for time_id, mask in time_masks.items():
                pooled = power[:, :, mask].mean(axis=-1)
                features[time_id][
                    batch_start:batch_stop,
                    channel_index,
                    :,
                ] = np.log10(
                    np.maximum(pooled, settings.log_power_eps)
                ).astype(np.float32)

            del power

    print(
        f"[TFR] Feature extraction completed in "
        f"{(time.time() - started) / 60.0:.1f} min",
        flush=True,
    )

    payload: dict[str, np.ndarray] = {
        "channels": np.asarray(actual_channels),
        "freqs": freqs,
    }
    payload.update(
        {
            f"features__{time_id}": values
            for time_id, values in features.items()
        }
    )

    # Uncompressed NPZ is intentionally used: it is substantially faster for
    # large numerical arrays and the cache is local working data.
    np.savez(cache_path, **payload)
    print(f"[CACHE] Saved {cache_path}", flush=True)
    return freqs, features, cache_path


# -----------------------------------------------------------------------------
# Condition evaluation and checkpointing
# -----------------------------------------------------------------------------


def condition_matrix(
    condition: Condition,
    features: dict[str, np.ndarray],
    freqs: np.ndarray,
    actual_channels: list[str],
) -> np.ndarray:
    channel_indices = [actual_channels.index(channel) for channel in condition.channels]
    frequency_mask = (
        (freqs >= condition.freq_start)
        & (freqs <= condition.freq_end)
    )
    if not np.any(frequency_mask):
        raise RuntimeError(f"No frequencies for {condition.freq_id}")

    tensor = features[condition.time_id][
        :,
        channel_indices,
        :,
    ][:, :, frequency_mask]
    return tensor.reshape(len(tensor), -1)


def evaluate_one_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    condition: Condition,
    settings: Settings,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model = make_model(settings)
    model.fit(X[train_idx], y[train_idx])

    y_pred = model.predict(X[test_idx])
    decision = model.decision_function(X[test_idx])

    try:
        auc = float(roc_auc_score(y[test_idx], decision))
    except ValueError:
        auc = float("nan")

    row = {
        **asdict(condition),
        "channels": "|".join(condition.channels),
        "n_channels": len(condition.channels),
        "n_features": X.shape[1],
        "fold": fold,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "accuracy": float(accuracy_score(y[test_idx], y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y[test_idx], y_pred)
        ),
        "macro_f1": float(f1_score(y[test_idx], y_pred, average="macro")),
        "roc_auc": auc,
    }

    predictions = metadata.iloc[test_idx].copy()
    predictions["epoch_index"] = test_idx
    predictions["fold"] = fold
    predictions["channel_group"] = condition.channel_group
    predictions["time_id"] = condition.time_id
    predictions["freq_id"] = condition.freq_id
    predictions["y_true"] = y[test_idx]
    predictions["y_pred"] = y_pred
    predictions["decision_function"] = decision
    return row, predictions


def evaluate_condition(
    condition: Condition,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = min(max(settings.classification_jobs, 1), len(splits))

    outputs = Parallel(n_jobs=jobs, prefer="threads")(
        delayed(evaluate_one_fold)(
            fold,
            train_idx,
            test_idx,
            X,
            y,
            metadata,
            condition,
            settings,
        )
        for fold, (train_idx, test_idx) in enumerate(splits)
    )

    rows = pd.DataFrame([item[0] for item in outputs])
    predictions = pd.concat([item[1] for item in outputs], ignore_index=True)
    return rows, predictions


def condition_checkpoint_paths(
    checkpoint_dir: Path,
    condition: Condition,
) -> tuple[Path, Path]:
    return (
        checkpoint_dir / f"fold_results__{condition.identifier}.csv",
        checkpoint_dir / f"predictions__{condition.identifier}.csv",
    )


def run_all_conditions(
    conditions: list[Condition],
    features: dict[str, np.ndarray],
    freqs: np.ndarray,
    actual_channels: list[str],
    y: np.ndarray,
    metadata: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    settings: Settings,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_fold_results: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    started = time.time()

    for number, condition in enumerate(conditions, start=1):
        result_path, prediction_path = condition_checkpoint_paths(
            checkpoint_dir,
            condition,
        )

        if settings.resume and result_path.exists() and prediction_path.exists():
            print(
                f"[SKIP {number:02d}/{len(conditions)}] {condition.identifier}",
                flush=True,
            )
            fold_rows = pd.read_csv(result_path)
            predictions = pd.read_csv(prediction_path)
        else:
            X = condition_matrix(condition, features, freqs, actual_channels)
            print(
                f"[FIT {number:02d}/{len(conditions)}] "
                f"{condition.identifier} | X={X.shape} | "
                f"parallel_folds={settings.classification_jobs}",
                flush=True,
            )
            condition_started = time.time()
            fold_rows, predictions = evaluate_condition(
                condition,
                X,
                y,
                metadata,
                splits,
                settings,
            )
            fold_rows.to_csv(result_path, index=False)
            predictions.to_csv(prediction_path, index=False)
            print(
                f"[DONE] {condition.identifier} | "
                f"{(time.time() - condition_started):.1f} s | "
                f"median F1={fold_rows['macro_f1'].median():.4f}",
                flush=True,
            )

        all_fold_results.append(fold_rows)
        all_predictions.append(predictions)

    print(
        f"[FIT] All conditions completed in "
        f"{(time.time() - started) / 60.0:.1f} min",
        flush=True,
    )
    return (
        pd.concat(all_fold_results, ignore_index=True),
        pd.concat(all_predictions, ignore_index=True),
    )


# -----------------------------------------------------------------------------
# Aggregation and plots
# -----------------------------------------------------------------------------


def aggregate_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "channel_group",
        "time_id",
        "freq_id",
        "channels",
        "n_channels",
        "n_features",
        "time_start",
        "time_end",
        "freq_start",
        "freq_end",
    ]

    summary = (
        fold_results.groupby(group_columns, as_index=False)
        .agg(
            macro_f1_median=("macro_f1", "median"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_median=("accuracy", "median"),
            balanced_accuracy_median=("balanced_accuracy", "median"),
            roc_auc_median=("roc_auc", "median"),
        )
    )

    q25 = (
        fold_results.groupby(group_columns)["macro_f1"]
        .quantile(0.25)
        .rename("macro_f1_q25")
        .reset_index()
    )
    q75 = (
        fold_results.groupby(group_columns)["macro_f1"]
        .quantile(0.75)
        .rename("macro_f1_q75")
        .reset_index()
    )

    summary = summary.merge(q25, on=group_columns).merge(q75, on=group_columns)
    return summary.sort_values(
        ["channel_group", "macro_f1_median"],
        ascending=[True, False],
    ).reset_index(drop=True)


def plot_macro_f1_grid(
    summary: pd.DataFrame,
    group_name: str,
    group_channels: list[str],
    time_windows: dict[str, tuple[float, float]],
    frequency_bands: dict[str, tuple[float, float]],
    output_dir: Path,
    patient_id: str,
    settings: Settings,
) -> None:
    group = summary[summary["channel_group"] == group_name]
    matrix = group.pivot(
        index="time_id",
        columns="freq_id",
        values="macro_f1_median",
    ).reindex(index=list(time_windows), columns=list(frequency_bands))

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Time window")
    ax.set_title(
        f"{patient_id} | {group_name} | median CV macro-F1\n"
        f"channels={group_channels}"
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            if pd.notna(value):
                ax.text(column, row, f"{value:.3f}", ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Median macro-F1")
    save_figure(fig, output_dir / f"01_{group_name}_macro_f1_grid", settings.save_svg)


def plot_selected_minus_rejected(
    summary: pd.DataFrame,
    time_windows: dict[str, tuple[float, float]],
    frequency_bands: dict[str, tuple[float, float]],
    output_dir: Path,
    patient_id: str,
    settings: Settings,
) -> pd.DataFrame:
    main = summary[summary["channel_group"].isin(["selected", "rejected"])]
    comparison = main.pivot_table(
        index=["time_id", "freq_id"],
        columns="channel_group",
        values="macro_f1_median",
    ).reset_index()
    comparison["selected_minus_rejected"] = (
        comparison["selected"] - comparison["rejected"]
    )
    comparison.to_csv(output_dir / "selected_vs_rejected.csv", index=False)

    difference = comparison.pivot(
        index="time_id",
        columns="freq_id",
        values="selected_minus_rejected",
    ).reindex(index=list(time_windows), columns=list(frequency_bands))

    limit = max(0.01, float(np.nanmax(np.abs(difference.to_numpy()))))
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    image = ax.imshow(
        difference.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_xticks(np.arange(len(difference.columns)))
    ax.set_xticklabels(difference.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(difference.index)))
    ax.set_yticklabels(difference.index)
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Time window")
    ax.set_title(f"{patient_id} | selected - rejected | median macro-F1")

    for row in range(difference.shape[0]):
        for column in range(difference.shape[1]):
            value = difference.iloc[row, column]
            if pd.notna(value):
                ax.text(column, row, f"{value:+.3f}", ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Delta median macro-F1")
    save_figure(
        fig,
        output_dir / "01_selected_minus_rejected_grid",
        settings.save_svg,
    )
    return comparison


def plot_metric_slice(
    summary: pd.DataFrame,
    fixed_column: str,
    fixed_value: str,
    varying_column: str,
    varying_order: list[str],
    title: str,
    output_path: Path,
    settings: Settings,
) -> None:
    main = summary[summary["channel_group"].isin(["selected", "rejected"])]
    subset = main[main[fixed_column] == fixed_value]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x = np.arange(len(varying_order))

    for group_name in ["selected", "rejected"]:
        group = (
            subset[subset["channel_group"] == group_name]
            .set_index(varying_column)
            .reindex(varying_order)
        )
        ax.plot(
            x,
            group["macro_f1_median"],
            marker="o",
            linewidth=2,
            label=group_name,
        )
        ax.fill_between(
            x,
            group["macro_f1_q25"].astype(float),
            group["macro_f1_q75"].astype(float),
            alpha=0.15,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(varying_order, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Macro-F1")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path, settings.save_svg)


def plot_confusion(
    predictions: pd.DataFrame,
    row: pd.Series,
    output_dir: Path,
    patient_id: str,
    settings: Settings,
) -> None:
    subset = predictions[
        (predictions["channel_group"] == row["channel_group"])
        & (predictions["time_id"] == row["time_id"])
        & (predictions["freq_id"] == row["freq_id"])
    ]
    matrix = confusion_matrix(subset["y_true"], subset["y_pred"], labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["all_other", "open_hand"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["all_other", "open_hand"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"{patient_id} | {row['channel_group']} | {row['role']}\n"
        f"{row['time_id']} | {row['freq_id']}"
    )

    for true_index in range(2):
        for pred_index in range(2):
            ax.text(
                pred_index,
                true_index,
                str(matrix[true_index, pred_index]),
                ha="center",
                va="center",
                fontsize=14,
            )

    fig.colorbar(image, ax=ax)
    save_figure(
        fig,
        output_dir
        / (
            f"confusion__{row['role']}__{row['channel_group']}__"
            f"{row['time_id']}__{row['freq_id']}"
        ),
        settings.save_svg,
    )


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def run(settings: Settings) -> None:
    time_windows = DEFAULT_TIME_WINDOWS.copy()
    frequency_bands = DEFAULT_FREQUENCY_BANDS.copy()

    run_root = ensure_dir(settings.output_storage / settings.patient_id)
    dirs = {
        "run_info": ensure_dir(run_root / "00_run_info"),
        "data_audit": ensure_dir(run_root / "01_data_audit"),
        "selected": ensure_dir(run_root / "02_selected_channels"),
        "rejected": ensure_dir(run_root / "03_rejected_channels"),
        "comparison": ensure_dir(run_root / "04_selected_vs_rejected"),
        "predictions": ensure_dir(run_root / "05_predictions"),
        "checkpoints": ensure_dir(run_root / "06_checkpoints"),
        "cache": ensure_dir(run_root / "_cache"),
    }

    cfg = load_external_config(settings.config_path)
    all_configured = list(cfg.ch_to_keep[settings.patient_id])
    selected_configured = list(cfg.best_ch_by_power[settings.patient_id])
    selected_lower = {name.lower() for name in selected_configured}
    rejected_configured = [
        name for name in all_configured if name.lower() not in selected_lower
    ]

    if not selected_configured:
        raise RuntimeError(f"best_ch_by_power[{settings.patient_id}] is empty.")
    if not rejected_configured:
        raise RuntimeError(f"No rejected channels for {settings.patient_id}.")

    print(f"[SETUP] patient={settings.patient_id}")
    print(f"[SETUP] selected={selected_configured}")
    print(f"[SETUP] rejected={rejected_configured}")
    print(f"[SETUP] output={run_root}")

    manifest = load_manifest(settings.study_def_path)
    manifest["accepted_after_rejection"] = parse_bool(
        manifest["accepted_after_rejection"]
    )
    manifest["included_in_main_task"] = parse_bool(
        manifest["included_in_main_task"]
    )

    patient_rows = manifest[
        (manifest["subject"] == settings.patient_id)
        & (manifest["accepted_after_rejection"] == True)
        & (manifest["included_in_main_task"] == True)
        & (manifest["class_label"].isin([0, 1]))
    ].copy()

    if settings.sessions is not None:
        patient_rows = patient_rows[
            patient_rows["session"].isin(settings.sessions)
        ].copy()

    if patient_rows.empty:
        raise RuntimeError(f"No accepted task epochs for {settings.patient_id}.")

    epochs, session_names, _ = load_patient_epochs(
        patient_rows,
        all_configured,
        settings,
        dirs["data_audit"],
    )

    actual_all = list(epochs.ch_names)
    actual_selected, missing_selected = resolve_channel_names(
        selected_configured,
        actual_all,
    )
    actual_rejected, missing_rejected = resolve_channel_names(
        rejected_configured,
        actual_all,
    )
    if missing_selected or missing_rejected:
        warnings.warn(
            f"Missing after resolution: selected={missing_selected}, "
            f"rejected={missing_rejected}"
        )
    if not actual_selected or not actual_rejected:
        raise RuntimeError(
            f"Invalid groups: selected={actual_selected}, rejected={actual_rejected}"
        )

    metadata = epochs.metadata.reset_index(drop=True).copy()
    y = metadata["class_label"].astype(int).to_numpy()
    metadata.to_csv(dirs["data_audit"] / "epoch_metadata.csv", index=False)

    splits, fold_assignment = make_splits(
        y,
        settings.cv_n_splits,
        settings.cv_seed,
    )
    fold_table = metadata.copy()
    fold_table["epoch_index"] = np.arange(len(metadata))
    fold_table["validation_fold"] = fold_assignment
    fold_table.to_csv(
        dirs["run_info"] / "fixed_cv_fold_assignment.csv",
        index=False,
    )

    freqs, features, feature_cache_path = extract_or_load_features(
        epochs,
        actual_all,
        settings,
        time_windows,
        dirs["cache"],
    )

    channel_groups = {
        "selected": actual_selected,
        "rejected": actual_rejected,
    }
    conditions = [
        Condition(
            channel_group=group_name,
            time_id=time_id,
            freq_id=freq_id,
            channels=tuple(group_channels),
            time_start=tmin,
            time_end=tmax,
            freq_start=fmin,
            freq_end=fmax,
        )
        for group_name, group_channels in channel_groups.items()
        for time_id, (tmin, tmax) in time_windows.items()
        for freq_id, (fmin, fmax) in frequency_bands.items()
    ]

    condition_table = pd.DataFrame([asdict(condition) for condition in conditions])
    condition_table["channels"] = condition_table["channels"].apply(
        lambda values: "|".join(values)
    )
    condition_table.to_csv(
        dirs["run_info"] / "frozen_ablation_conditions.csv",
        index=False,
    )

    fold_results, predictions = run_all_conditions(
        conditions,
        features,
        freqs,
        actual_all,
        y,
        metadata,
        splits,
        settings,
        dirs["checkpoints"],
    )
    fold_results.to_csv(run_root / "all_fold_results.csv", index=False)
    predictions.to_csv(
        dirs["predictions"] / "all_out_of_fold_predictions.csv",
        index=False,
    )

    summary = aggregate_results(fold_results)
    summary.to_csv(run_root / "condition_summary.csv", index=False)
    summary[summary["channel_group"] == "selected"].to_csv(
        dirs["selected"] / "selected_condition_summary.csv",
        index=False,
    )
    summary[summary["channel_group"] == "rejected"].to_csv(
        dirs["rejected"] / "rejected_condition_summary.csv",
        index=False,
    )

    plot_macro_f1_grid(
        summary,
        "selected",
        actual_selected,
        time_windows,
        frequency_bands,
        dirs["selected"],
        settings.patient_id,
        settings,
    )
    plot_macro_f1_grid(
        summary,
        "rejected",
        actual_rejected,
        time_windows,
        frequency_bands,
        dirs["rejected"],
        settings.patient_id,
        settings,
    )
    plot_selected_minus_rejected(
        summary,
        time_windows,
        frequency_bands,
        dirs["comparison"],
        settings.patient_id,
        settings,
    )
    plot_metric_slice(
        summary,
        fixed_column="freq_id",
        fixed_value=settings.baseline_freq_id,
        varying_column="time_id",
        varying_order=list(time_windows),
        title=(
            f"{settings.patient_id} | time-window ablation | "
            f"{settings.baseline_freq_id}"
        ),
        output_path=dirs["comparison"] / "02_time_window_slice",
        settings=settings,
    )
    plot_metric_slice(
        summary,
        fixed_column="time_id",
        fixed_value=settings.baseline_time_id,
        varying_column="freq_id",
        varying_order=list(frequency_bands),
        title=(
            f"{settings.patient_id} | frequency-band ablation | "
            f"{settings.baseline_time_id}"
        ),
        output_path=dirs["comparison"] / "03_frequency_band_slice",
        settings=settings,
    )

    baseline_rows = summary[
        (summary["time_id"] == settings.baseline_time_id)
        & (summary["freq_id"] == settings.baseline_freq_id)
    ].copy()
    baseline_rows["role"] = "baseline"
    best_rows = (
        summary.sort_values(
            ["channel_group", "macro_f1_median"],
            ascending=[True, False],
        )
        .groupby("channel_group", as_index=False)
        .head(1)
        .copy()
    )
    best_rows["role"] = "best_in_frozen_grid"
    report = pd.concat([baseline_rows, best_rows], ignore_index=True)
    report.to_csv(dirs["comparison"] / "baseline_and_best.csv", index=False)

    for _, row in report.iterrows():
        plot_confusion(
            predictions,
            row,
            dirs["comparison"],
            settings.patient_id,
            settings,
        )

    def get_one(group_name: str, time_id: str, freq_id: str) -> pd.Series:
        rows = summary[
            (summary["channel_group"] == group_name)
            & (summary["time_id"] == time_id)
            & (summary["freq_id"] == freq_id)
        ]
        if len(rows) != 1:
            raise RuntimeError(
                f"Expected one row for {group_name}/{time_id}/{freq_id}"
            )
        return rows.iloc[0]

    selected_baseline = get_one(
        "selected",
        settings.baseline_time_id,
        settings.baseline_freq_id,
    )
    rejected_baseline = get_one(
        "rejected",
        settings.baseline_time_id,
        settings.baseline_freq_id,
    )
    selected_best = best_rows[best_rows["channel_group"] == "selected"].iloc[0]
    rejected_best = best_rows[best_rows["channel_group"] == "rejected"].iloc[0]

    summary_text = f"""
Patient: {settings.patient_id}

Fixed experiment:
- accepted epochs: {len(epochs)}
- CV: {settings.cv_n_splits}-fold StratifiedKFold, seed={settings.cv_seed}
- model: StandardScaler(train only) + SVC(
    kernel={settings.svm_kernel}, C={settings.svm_c}, gamma={settings.svm_gamma}
  )
- CAR reference: all ch_to_keep channels before selected/rejected split
- feature cache: {feature_cache_path}

Channels:
- selected ({len(actual_selected)}): {actual_selected}
- rejected ({len(actual_rejected)}): {actual_rejected}

Baseline {settings.baseline_time_id} / {settings.baseline_freq_id}:
- selected macro-F1 median: {selected_baseline['macro_f1_median']:.4f}
- rejected macro-F1 median: {rejected_baseline['macro_f1_median']:.4f}
- selected - rejected: {selected_baseline['macro_f1_median'] - rejected_baseline['macro_f1_median']:+.4f}

Best selected in frozen grid:
- {selected_best['time_id']} / {selected_best['freq_id']}
- macro-F1 median={selected_best['macro_f1_median']:.4f}

Best rejected in frozen grid:
- {rejected_best['time_id']} / {rejected_best['freq_id']}
- macro-F1 median={rejected_best['macro_f1_median']:.4f}
""".strip()
    (dirs["run_info"] / "RESULT_SUMMARY.txt").write_text(
        summary_text,
        encoding="utf-8",
    )
    print("\n" + summary_text)

    run_config = {
        "settings": {
            **asdict(settings),
            "config_path": str(settings.config_path),
            "study_def_path": str(settings.study_def_path),
            "data_root": str(settings.data_root),
            "output_storage": str(settings.output_storage),
        },
        "sessions": session_names,
        "all_channels_configured": all_configured,
        "selected_channels_configured": selected_configured,
        "rejected_channels_configured": rejected_configured,
        "all_channels_actual": actual_all,
        "selected_channels_actual": actual_selected,
        "rejected_channels_actual": actual_rejected,
        "n_epochs": len(epochs),
        "class_counts": {
            str(key): int(value)
            for key, value in pd.Series(y).value_counts().items()
        },
        "time_windows": time_windows,
        "frequency_bands": frequency_bands,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "mne": mne.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (dirs["run_info"] / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_baseline(value: str) -> tuple[float, float] | None:
    if value.lower() in {"none", "off", "null"}:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Baseline must be 'start,end' or 'none'.")
    return float(parts[0]), float(parts[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed SVM time-frequency/channel ablation for one patient."
    )

    parser.add_argument("--patient", default="s11")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--study-def-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-storage", type=Path, required=True)
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Optional explicit session names. Omit to use all patient sessions.",
    )

    parser.add_argument("--svm-kernel", default="linear")
    parser.add_argument("--svm-c", type=float, default=752.6521969107142)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--svm-class-weight", default=None)
    parser.add_argument("--svm-cache-mb", type=float, default=4096.0)

    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument(
        "--classification-jobs",
        type=int,
        default=1,
        help="Parallel CV folds per condition. Start with 1-5 depending on RAM/CPU.",
    )

    parser.add_argument("--notch-freqs", type=float, nargs="*", default=[50, 100, 150])
    parser.add_argument("--l-freq", type=float, default=0.1)
    parser.add_argument("--h-freq", type=float, default=120.0)
    parser.add_argument("--epoch-tmin", type=float, default=-0.9)
    parser.add_argument("--epoch-tmax", type=float, default=2.1)
    parser.add_argument("--baseline", type=parse_baseline, default=(-0.1, 0.0))

    parser.add_argument("--tfr-fmin", type=float, default=0.1)
    parser.add_argument("--tfr-fmax", type=float, default=120.0)
    parser.add_argument("--tfr-n-freqs", type=int, default=100)
    parser.add_argument("--tfr-decim", type=int, default=20)
    parser.add_argument("--tfr-batch-size", type=int, default=24)
    parser.add_argument("--tfr-jobs", type=int, default=1)

    parser.add_argument("--baseline-time-id", default="T0_marker_0_1")
    parser.add_argument("--baseline-freq-id", default="F0_current_0p1_59p4")

    parser.add_argument(
        "--no-feature-cache",
        action="store_true",
        help="Do not reuse an existing feature cache.",
    )
    parser.add_argument(
        "--overwrite-feature-cache",
        action="store_true",
        help="Force feature recomputation even when a matching cache exists.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore per-condition checkpoints and recompute every condition.",
    )
    parser.add_argument("--save-svg", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    class_weight: str | None
    if args.svm_class_weight is None or str(args.svm_class_weight).lower() in {
        "none",
        "null",
        "off",
    }:
        class_weight = None
    else:
        class_weight = str(args.svm_class_weight)

    settings = Settings(
        patient_id=str(args.patient),
        config_path=args.config_path.expanduser().resolve(),
        study_def_path=args.study_def_path.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve(),
        output_storage=args.output_storage.expanduser().resolve(),
        sessions=None if args.sessions is None else tuple(args.sessions),
        svm_kernel=str(args.svm_kernel),
        svm_c=float(args.svm_c),
        svm_gamma=str(args.svm_gamma),
        svm_class_weight=class_weight,
        svm_cache_mb=float(args.svm_cache_mb),
        cv_n_splits=int(args.cv_splits),
        cv_seed=int(args.cv_seed),
        classification_jobs=max(int(args.classification_jobs), 1),
        notch_freqs=tuple(float(x) for x in args.notch_freqs),
        l_freq=float(args.l_freq),
        h_freq=float(args.h_freq),
        epoch_tmin=float(args.epoch_tmin),
        epoch_tmax=float(args.epoch_tmax),
        baseline=args.baseline,
        tfr_fmin=float(args.tfr_fmin),
        tfr_fmax=float(args.tfr_fmax),
        tfr_n_freqs=int(args.tfr_n_freqs),
        tfr_decim=int(args.tfr_decim),
        tfr_batch_size=int(args.tfr_batch_size),
        tfr_n_jobs=max(int(args.tfr_jobs), 1),
        log_power_eps=1e-20,
        baseline_time_id=str(args.baseline_time_id),
        baseline_freq_id=str(args.baseline_freq_id),
        reuse_feature_cache=not args.no_feature_cache,
        overwrite_feature_cache=bool(args.overwrite_feature_cache),
        resume=not args.no_resume,
        save_svg=bool(args.save_svg),
    )

    run(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
