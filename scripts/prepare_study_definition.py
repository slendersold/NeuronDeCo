#!/usr/bin/env python3
"""Build the technical artefacts required to freeze an sEEG decoding task.

The script is designed for the dataset layout used by NeuronDeCo:

    DATA_ROOT/
      <anything>_<subject>/
        session_*/
          *.edf
        rest/                 # ignored by default

It reads the retained-channel lists and patient-specific epoch rejection
thresholds from the same external ``config.py`` used by NeuronDeCo, applies
the preprocessing used in the current study, and writes:

* ``gesture_mapping.csv``     — annotation/event mapping and task classes;
* ``sample_manifest.csv``     — one row per event/epoch, including rejection;
* ``class_counts.csv``        — counts by patient, session, class and status;
* ``class_counts_wide.csv``   — compact patient-wise counts;
* ``task_definition.md``      — frozen human-readable task definition;
* ``trial_timing.png``        — marker/robot/epoch timing scheme;
* ``task_config.resolved.json`` and ``dataset_summary.json``;
* ``warnings.csv`` / ``errors.csv`` when applicable.

The default task reproduces the current NeuronDeCo binary setup:
``event 9 (open_hand)`` versus ``events 1-8 and 10``.  For a true pairwise
extension-versus-flexion experiment, pass a JSON task specification with
``--task-config``.  Run ``--write-task-template`` to create a template.

Examples
--------
Current open-hand-versus-all task::

    python scripts/prepare_study_definition.py \
        --data-root /path/to/PirogovDATA \
        --config /path/to/PreprocessedData/config.py \
        --output-dir study_definition

Pairwise task::

    python scripts/prepare_study_definition.py \
        --data-root /path/to/PirogovDATA \
        --config /path/to/PreprocessedData/config.py \
        --task-config task_extension_vs_flexion.json \
        --output-dir study_definition_pairwise

The script never invents gesture semantics.  Event names absent from the task
configuration are written as ``event_<code>_[TO_CONFIRM]``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


DEFAULT_GESTURE_CODES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


@dataclass(frozen=True)
class ClassSpec:
    name: str
    codes: tuple[int, ...]
    label: int


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    task_description: str
    positive: ClassSpec
    negative: ClassSpec
    event_names: dict[int, str]
    movement_types: dict[int, str]
    marker_to_robot_onset_s: float = 0.1
    visual_window_robot_s: tuple[float, float] = (-1.0, 2.0)
    decoder_window_marker_s: tuple[float, float] = (0.0, 1.0)

    @property
    def included_codes(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.positive.codes) | set(self.negative.codes)))

    def class_for_code(self, code: int) -> tuple[str, int] | None:
        if code in self.positive.codes:
            return self.positive.name, self.positive.label
        if code in self.negative.codes:
            return self.negative.name, self.negative.label
        return None


@dataclass(frozen=True)
class PreprocessingSpec:
    expected_sfreq: float = 2000.0
    notch_freqs: tuple[float, ...] = (50.0, 100.0, 150.0)
    l_freq: float = 0.1
    h_freq: float = 120.0
    tmin_marker_s: float = -0.2
    tmax_marker_s: float = 2.1
    baseline_marker_s: tuple[float, float] = (-0.1, 0.0)
    reference: str = "CAR across retained EEG channels"
    filter_method: str = "iir"


@dataclass
class RunMessages:
    warnings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def warn(self, *, subject: str = "", session: str = "", message: str) -> None:
        self.warnings.append(
            {"subject": subject, "session": session, "message": message}
        )

    def error(self, *, subject: str = "", session: str = "", message: str) -> None:
        self.errors.append(
            {"subject": subject, "session": session, "message": message}
        )


def _default_task_spec() -> TaskSpec:
    event_names = {code: f"gesture_{code}_[TO_CONFIRM]" for code in DEFAULT_GESTURE_CODES}
    event_names[9] = "open_hand"
    movement_types = {code: "[TO_CONFIRM]" for code in DEFAULT_GESTURE_CODES}
    movement_types[9] = "open-hand state / extension [confirm exact interpretation]"
    return TaskSpec(
        task_name="open_hand_vs_all_other_gestures",
        task_description=(
            "Offline single-epoch binary classification of event 9 (open hand) "
            "versus events 1-8 and 10 pooled as the negative class."
        ),
        positive=ClassSpec(name="open_hand", codes=(9,), label=1),
        negative=ClassSpec(
            name="all_other_gestures", codes=(1, 2, 3, 4, 5, 6, 7, 8, 10), label=0
        ),
        event_names=event_names,
        movement_types=movement_types,
    )


def _task_template() -> dict[str, Any]:
    return {
        "task_name": "hand_extension_vs_flexion",
        "task_description": (
            "Offline single-epoch binary classification of hand extension/opening "
            "versus a specific hand/finger flexion gesture."
        ),
        "positive": {
            "name": "hand_extension",
            "codes": [9],
            "label": 1,
        },
        "negative": {
            "name": "hand_flexion",
            "codes": ["REPLACE_WITH_REAL_EVENT_CODE"],
            "label": 0,
        },
        "event_names": {
            "1": "[TO_CONFIRM]",
            "2": "[TO_CONFIRM]",
            "3": "[TO_CONFIRM]",
            "4": "[TO_CONFIRM]",
            "5": "[TO_CONFIRM]",
            "6": "[TO_CONFIRM]",
            "7": "[TO_CONFIRM]",
            "8": "[TO_CONFIRM]",
            "9": "open_hand / hand_extension [confirm]",
            "10": "[TO_CONFIRM]",
        },
        "movement_types": {
            "1": "[TO_CONFIRM]",
            "2": "[TO_CONFIRM]",
            "3": "[TO_CONFIRM]",
            "4": "[TO_CONFIRM]",
            "5": "[TO_CONFIRM]",
            "6": "[TO_CONFIRM]",
            "7": "[TO_CONFIRM]",
            "8": "[TO_CONFIRM]",
            "9": "extension/open state [confirm]",
            "10": "[TO_CONFIRM]",
        },
        "marker_to_robot_onset_s": 0.1,
        "visual_window_robot_s": [-1.0, 2.0],
        "decoder_window_marker_s": [0.0, 1.0],
    }


def _parse_int_list(values: Any, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"'{field_name}' must be a non-empty JSON list of integers")
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"Boolean value is invalid in '{field_name}'")
        try:
            parsed.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid event code {value!r} in '{field_name}'. "
                "Replace all template placeholders with real integer codes."
            ) from exc
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Duplicate event codes in '{field_name}': {parsed}")
    return tuple(parsed)


def load_task_spec(path: Path | None) -> TaskSpec:
    if path is None:
        return _default_task_spec()

    payload = json.loads(path.read_text(encoding="utf-8"))
    positive = payload.get("positive", {})
    negative = payload.get("negative", {})

    positive_spec = ClassSpec(
        name=str(positive["name"]),
        codes=_parse_int_list(positive["codes"], field_name="positive.codes"),
        label=int(positive.get("label", 1)),
    )
    negative_spec = ClassSpec(
        name=str(negative["name"]),
        codes=_parse_int_list(negative["codes"], field_name="negative.codes"),
        label=int(negative.get("label", 0)),
    )

    overlap = set(positive_spec.codes) & set(negative_spec.codes)
    if overlap:
        raise ValueError(f"Positive and negative event codes overlap: {sorted(overlap)}")
    if positive_spec.label == negative_spec.label:
        raise ValueError("Positive and negative class labels must differ")

    event_names = {int(k): str(v) for k, v in payload.get("event_names", {}).items()}
    movement_types = {
        int(k): str(v) for k, v in payload.get("movement_types", {}).items()
    }
    for code in set(positive_spec.codes) | set(negative_spec.codes):
        event_names.setdefault(code, f"event_{code}_[TO_CONFIRM]")
        movement_types.setdefault(code, "[TO_CONFIRM]")

    visual_window = payload.get("visual_window_robot_s", [-1.0, 2.0])
    decoder_window = payload.get("decoder_window_marker_s", [0.0, 1.0])
    if len(visual_window) != 2 or len(decoder_window) != 2:
        raise ValueError("visual_window_robot_s and decoder_window_marker_s must have two values")

    return TaskSpec(
        task_name=str(payload["task_name"]),
        task_description=str(payload["task_description"]),
        positive=positive_spec,
        negative=negative_spec,
        event_names=event_names,
        movement_types=movement_types,
        marker_to_robot_onset_s=float(payload.get("marker_to_robot_onset_s", 0.1)),
        visual_window_robot_s=(float(visual_window[0]), float(visual_window[1])),
        decoder_window_marker_s=(float(decoder_window[0]), float(decoder_window[1])),
    )


def load_external_config(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    spec = importlib.util.spec_from_file_location("neurondeco_study_config", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_mapping(module: ModuleType, name: str) -> Mapping[str, Any]:
    value = getattr(module, name, None)
    if not isinstance(value, Mapping):
        raise KeyError(f"External config must contain mapping '{name}'")
    return value


def choose_channel_mapping(module: ModuleType, key: str) -> Mapping[str, Sequence[str]]:
    mapping = require_mapping(module, key)
    return {str(subject): tuple(map(str, channels)) for subject, channels in mapping.items()}


def choose_threshold(
    thresholds: Mapping[str, Any], subject: str, override: float | None
) -> float:
    if override is not None:
        return float(override)
    if subject in thresholds:
        return float(thresholds[subject])
    if "default" in thresholds:
        return float(thresholds["default"])
    raise KeyError(f"No epoch threshold for subject {subject!r} and no 'default' value")


def discover_subject_directories(data_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for path in sorted(data_root.iterdir()):
        if not path.is_dir():
            continue
        match = re.search(r"_(s\d+)$", path.name, flags=re.IGNORECASE)
        if match:
            subject = match.group(1).lower()
            result.setdefault(subject, []).append(path)
    return result


def discover_sessions(
    subject_dirs: Sequence[Path], strategy: str
) -> list[tuple[str, Path]]:
    sessions: list[tuple[str, Path]] = []
    for subject_dir in subject_dirs:
        for session_dir in sorted(subject_dir.glob("session_*")):
            if not session_dir.is_dir():
                continue
            candidates = sorted(
                f for f in session_dir.glob("*.edf") if "_raw" not in f.name
            )
            if not candidates:
                continue
            if strategy == "first":
                chosen = candidates[0]
            elif strategy == "last":
                chosen = candidates[-1]
            elif strategy == "biggest":
                chosen = max(candidates, key=lambda p: p.stat().st_size)
            elif strategy == "smallest":
                chosen = min(candidates, key=lambda p: p.stat().st_size)
            else:
                raise ValueError(f"Unknown EDF strategy: {strategy}")
            session_id = f"{subject_dir.name}/{session_dir.name}"
            sessions.append((session_id, chosen))
    return sessions


def parse_annotation_code(description: str) -> int | None:
    text = str(description).strip()
    integer_match = re.fullmatch(r"[+-]?\d+", text)
    if integer_match:
        return int(text)
    decimal_match = re.fullmatch(r"([+-]?\d+)\.0+", text)
    if decimal_match:
        return int(decimal_match.group(1))
    code_match = re.search(r"(?:event|code|marker|gesture)[_\s:=/-]*([+-]?\d+)$", text, re.I)
    if code_match:
        return int(code_match.group(1))
    return None


def annotation_records(
    raw: mne.io.BaseRaw,
    *,
    code_source: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _, mne_event_dict = mne.events_from_annotations(raw, verbose=False)
    records: list[dict[str, Any]] = []

    for annotation_index, (onset, duration, description) in enumerate(
        zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)
    ):
        description = str(description)
        annotation_code = parse_annotation_code(description)
        mne_code = int(mne_event_dict[description]) if description in mne_event_dict else None

        if code_source == "annotation":
            effective_code = annotation_code
        elif code_source == "mne":
            effective_code = mne_code
        elif code_source == "auto":
            effective_code = annotation_code if annotation_code is not None else mne_code
        else:
            raise ValueError(f"Unknown event-code source: {code_source}")

        sample_index = raw.time_as_index(
            float(onset),
            use_rounding=True,
            origin=raw.annotations.orig_time,
        )[0]
        if raw.annotations.orig_time is not None:
            sample_index += raw.first_samp
        sample = int(sample_index)
        records.append(
            {
                "annotation_index": annotation_index,
                "annotation_description": description,
                "annotation_onset_s": float(onset),
                "annotation_duration_s": float(duration),
                "sample": sample,
                "annotation_numeric_code": annotation_code,
                "mne_event_code": mne_code,
                "event_code": effective_code,
            }
        )
    return records, {str(k): int(v) for k, v in mne_event_dict.items()}


def apply_preprocessing(
    raw: mne.io.BaseRaw,
    retained_channels: Sequence[str],
    spec: PreprocessingSpec,
    *,
    n_jobs: int,
) -> mne.io.BaseRaw:
    missing = [channel for channel in retained_channels if channel not in raw.ch_names]
    if missing:
        raise KeyError(f"Retained channels absent from EDF: {missing}")
    if not retained_channels:
        raise ValueError("Retained-channel list is empty")

    clean = raw.copy().pick(list(retained_channels)).load_data()
    nyquist = float(clean.info["sfreq"]) / 2.0
    notch = [frequency for frequency in spec.notch_freqs if frequency < nyquist]
    if notch:
        clean.notch_filter(freqs=np.asarray(notch), n_jobs=n_jobs, verbose=False)
    clean.filter(
        l_freq=spec.l_freq,
        h_freq=min(spec.h_freq, nyquist - 1e-6),
        method=spec.filter_method,
        n_jobs=n_jobs,
        verbose=False,
    )

    eeg_picks = mne.pick_types(clean.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        # EDF readers sometimes assign a non-EEG type.  The original project
        # treated all retained signal channels as EEG for rejection and CAR.
        clean.set_channel_types({channel: "eeg" for channel in clean.ch_names}, verbose=False)
    clean.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    return clean


def build_epochs_and_status(
    clean: mne.io.BaseRaw,
    event_records: list[dict[str, Any]],
    task: TaskSpec,
    preprocessing: PreprocessingSpec,
    threshold: float,
) -> tuple[mne.Epochs | None, dict[int, tuple[bool, str]]]:
    indexed_events: list[tuple[int, dict[str, Any]]] = []
    for record_index, record in enumerate(event_records):
        code = record["event_code"]
        if code is None or int(code) not in task.included_codes:
            continue
        indexed_events.append((record_index, record))

    if not indexed_events:
        return None, {}

    events = np.asarray(
        [[record["sample"], 0, int(record["event_code"])] for _, record in indexed_events],
        dtype=int,
    )
    order = np.argsort(events[:, 0], kind="stable")
    events = events[order]
    indexed_events = [indexed_events[int(i)] for i in order]

    observed_codes = sorted(set(map(int, events[:, 2].tolist())))
    event_id: dict[str, int] = {f"event_{code}": code for code in observed_codes}

    epochs = mne.Epochs(
        clean,
        events=events,
        event_id=event_id,
        tmin=preprocessing.tmin_marker_s,
        tmax=preprocessing.tmax_marker_s,
        baseline=preprocessing.baseline_marker_s,
        reject={"eeg": float(threshold)},
        reject_by_annotation=True,
        preload=True,
        verbose=False,
    )

    selected_positions = set(map(int, epochs.selection.tolist()))
    status: dict[int, tuple[bool, str]] = {}
    for position, (record_index, _) in enumerate(indexed_events):
        reasons = epochs.drop_log[position] if position < len(epochs.drop_log) else tuple()
        accepted = position in selected_positions
        reason = "accepted" if accepted else ";".join(map(str, reasons)) or "rejected"
        status[record_index] = (accepted, reason)
    return epochs, status


def make_manifest_rows(
    *,
    subject: str,
    session: str,
    edf_path: Path,
    raw: mne.io.BaseRaw,
    retained_channels: Sequence[str],
    threshold: float,
    event_records: list[dict[str, Any]],
    event_status: Mapping[int, tuple[bool, str]],
    task: TaskSpec,
    preprocessing: PreprocessingSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sfreq = float(raw.info["sfreq"])
    for record_index, record in enumerate(event_records):
        code = record["event_code"]
        class_info = task.class_for_code(int(code)) if code is not None else None
        included = class_info is not None
        class_name, class_label = class_info if class_info is not None else ("excluded", np.nan)
        accepted, rejection_reason = event_status.get(
            record_index,
            (False, "not_in_task" if not included else "not_epoched"),
        )
        code_int = int(code) if code is not None else None
        marker_onset = float(record["annotation_onset_s"])
        robot_onset = marker_onset + task.marker_to_robot_onset_s
        epoch_start = marker_onset + preprocessing.tmin_marker_s
        epoch_end = marker_onset + preprocessing.tmax_marker_s
        rows.append(
            {
                "subject": subject,
                "session": session,
                "edf_path": str(edf_path),
                "epoch_id": f"{subject}::{session}::{record['annotation_index']}",
                "annotation_index": int(record["annotation_index"]),
                "annotation_description": record["annotation_description"],
                "annotation_numeric_code": record["annotation_numeric_code"],
                "mne_event_code": record["mne_event_code"],
                "event_code": code_int,
                "gesture_name": task.event_names.get(
                    code_int, f"event_{code_int}_[TO_CONFIRM]" if code_int is not None else "[UNPARSED]"
                ),
                "movement_type": task.movement_types.get(code_int, "[TO_CONFIRM]"),
                "class_name": class_name,
                "class_label": class_label,
                "included_in_main_task": included,
                "accepted_after_rejection": bool(accepted),
                "rejection_reason": rejection_reason,
                "sample": int(record["sample"]),
                "marker_onset_s": marker_onset,
                "robot_onset_s": robot_onset,
                "annotation_duration_s": float(record["annotation_duration_s"]),
                "epoch_start_s": epoch_start,
                "epoch_end_s": epoch_end,
                "sfreq_hz": sfreq,
                "recording_duration_s": float(raw.times[-1]) if len(raw.times) else 0.0,
                "n_retained_channels": len(retained_channels),
                "retained_channels": "|".join(retained_channels),
                "epoch_reject_threshold_v": float(threshold),
            }
        )
    return rows


def write_gesture_mapping(
    manifest: pd.DataFrame,
    task: TaskSpec,
    output_path: Path,
) -> pd.DataFrame:
    observed = manifest[
        [
            "annotation_description",
            "annotation_numeric_code",
            "mne_event_code",
            "event_code",
        ]
    ].drop_duplicates()

    grouped_rows: list[dict[str, Any]] = []
    for code in sorted(set(manifest["event_code"].dropna().astype(int).tolist()) | set(task.event_names)):
        subset = observed[observed["event_code"] == code]
        descriptions = sorted(set(subset["annotation_description"].astype(str)))
        annotation_codes = sorted(set(subset["annotation_numeric_code"].dropna().astype(int)))
        mne_codes = sorted(set(subset["mne_event_code"].dropna().astype(int)))
        class_info = task.class_for_code(code)
        grouped_rows.append(
            {
                "event_code": code,
                "annotation_descriptions": " | ".join(descriptions),
                "annotation_numeric_codes": " | ".join(map(str, annotation_codes)),
                "mne_event_codes": " | ".join(map(str, mne_codes)),
                "gesture_name": task.event_names.get(code, f"event_{code}_[TO_CONFIRM]"),
                "movement_type": task.movement_types.get(code, "[TO_CONFIRM]"),
                "main_task_class": class_info[0] if class_info else "excluded",
                "class_label": class_info[1] if class_info else "",
                "included_in_main_task": class_info is not None,
                "semantic_status": (
                    "TO_CONFIRM"
                    if "TO_CONFIRM" in task.event_names.get(code, "")
                    or "TO_CONFIRM" in task.movement_types.get(code, "")
                    or "confirm" in task.movement_types.get(code, "").lower()
                    else "confirmed_by_task_config"
                ),
            }
        )
    mapping_df = pd.DataFrame(grouped_rows)
    mapping_df.to_csv(output_path, index=False)
    return mapping_df


def write_class_counts(manifest: pd.DataFrame, output_dir: Path, task: TaskSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    included = manifest[manifest["included_in_main_task"]].copy()
    if included.empty:
        long_counts = pd.DataFrame(
            columns=["subject", "session", "class_name", "status", "n_epochs"]
        )
    else:
        included["status"] = np.where(
            included["accepted_after_rejection"], "accepted", "rejected"
        )
        long_counts = (
            included.groupby(["subject", "session", "class_name", "status"], dropna=False)
            .size()
            .rename("n_epochs")
            .reset_index()
            .sort_values(["subject", "session", "class_name", "status"])
        )
    long_counts.to_csv(output_dir / "class_counts.csv", index=False)

    per_subject = (
        included.groupby(["subject", "class_name", "accepted_after_rejection"])
        .size()
        .rename("count")
        .reset_index()
        if not included.empty
        else pd.DataFrame(columns=["subject", "class_name", "accepted_after_rejection", "count"])
    )
    rows: list[dict[str, Any]] = []
    for subject in sorted(included["subject"].unique()) if not included.empty else []:
        row: dict[str, Any] = {"subject": subject}
        subject_data = per_subject[per_subject["subject"] == subject]
        for class_spec in (task.positive, task.negative):
            class_data = subject_data[subject_data["class_name"] == class_spec.name]
            accepted = int(
                class_data.loc[class_data["accepted_after_rejection"] == True, "count"].sum()  # noqa: E712
            )
            rejected = int(
                class_data.loc[class_data["accepted_after_rejection"] == False, "count"].sum()  # noqa: E712
            )
            row[f"{class_spec.name}_accepted"] = accepted
            row[f"{class_spec.name}_rejected"] = rejected
            row[f"{class_spec.name}_total"] = accepted + rejected
        pos = row.get(f"{task.positive.name}_accepted", 0)
        neg = row.get(f"{task.negative.name}_accepted", 0)
        total = pos + neg
        row["accepted_total"] = total
        row["positive_fraction_accepted"] = pos / total if total else np.nan
        row["majority_class_accuracy"] = max(pos, neg) / total if total else np.nan
        rows.append(row)
    wide_counts = pd.DataFrame(rows)
    wide_counts.to_csv(output_dir / "class_counts_wide.csv", index=False)
    return long_counts, wide_counts


def write_trial_timing(
    task: TaskSpec,
    preprocessing: PreprocessingSpec,
    output_path: Path,
) -> None:
    offset = task.marker_to_robot_onset_s
    marker_robot_time = -offset
    epoch_start_robot = preprocessing.tmin_marker_s - offset
    epoch_end_robot = preprocessing.tmax_marker_s - offset
    decoder_start_robot = task.decoder_window_marker_s[0] - offset
    decoder_end_robot = task.decoder_window_marker_s[1] - offset

    xmin = min(task.visual_window_robot_s[0], epoch_start_robot) - 0.1
    xmax = max(task.visual_window_robot_s[1], epoch_end_robot) + 0.1

    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    ax.axhline(0, linewidth=1)
    ax.axvspan(epoch_start_robot, epoch_end_robot, alpha=0.16, label="epoch used by preprocessing")
    ax.axvspan(decoder_start_robot, decoder_end_robot, alpha=0.28, label="current decoder window")
    ax.axvline(marker_robot_time, linestyle="--", linewidth=1.6)
    ax.axvline(0.0, linestyle="-", linewidth=2.0)
    ax.scatter([marker_robot_time, 0.0], [0.0, 0.0], zorder=3)
    ax.text(marker_robot_time, 0.12, f"LSL marker\n{marker_robot_time:+.3f} s", ha="center")
    ax.text(0.0, -0.22, "robot movement onset\nt = 0", ha="center", va="top")
    ax.text(
        (epoch_start_robot + epoch_end_robot) / 2,
        0.28,
        f"epoch: {epoch_start_robot:+.2f} ... {epoch_end_robot:+.2f} s relative to robot onset",
        ha="center",
    )
    ax.text(
        (decoder_start_robot + decoder_end_robot) / 2,
        -0.36,
        f"decoder: {decoder_start_robot:+.2f} ... {decoder_end_robot:+.2f} s relative to robot onset",
        ha="center",
        va="top",
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.65, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("Time relative to robot movement onset, s")
    ax.set_title("Trial timing definition")
    ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _format_codes(codes: Sequence[int]) -> str:
    return ", ".join(map(str, codes))


def write_task_definition(
    *,
    output_path: Path,
    data_root: Path,
    config_path: Path,
    task: TaskSpec,
    preprocessing: PreprocessingSpec,
    subjects: Sequence[str],
    manifest: pd.DataFrame,
    wide_counts: pd.DataFrame,
    mapping: pd.DataFrame,
    code_source: str,
    messages: RunMessages,
) -> None:
    n_sessions = int(manifest[["subject", "session"]].drop_duplicates().shape[0]) if not manifest.empty else 0
    accepted = manifest[
        manifest["included_in_main_task"] & manifest["accepted_after_rejection"]
    ] if not manifest.empty else manifest
    positive_n = int((accepted["class_name"] == task.positive.name).sum()) if not accepted.empty else 0
    negative_n = int((accepted["class_name"] == task.negative.name).sum()) if not accepted.empty else 0
    total_n = positive_n + negative_n
    pos_fraction = positive_n / total_n if total_n else math.nan

    unresolved = mapping[mapping["semantic_status"] == "TO_CONFIRM"]["event_code"].tolist()
    marker_robot = -task.marker_to_robot_onset_s
    epoch_robot = (
        preprocessing.tmin_marker_s - task.marker_to_robot_onset_s,
        preprocessing.tmax_marker_s - task.marker_to_robot_onset_s,
    )
    decoder_robot = (
        task.decoder_window_marker_s[0] - task.marker_to_robot_onset_s,
        task.decoder_window_marker_s[1] - task.marker_to_robot_onset_s,
    )

    lines = [
        f"# Frozen study definition: `{task.task_name}`",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. Dataset selection",
        "",
        f"- Data root: `{data_root}`",
        f"- NeuronDeCo config: `{config_path}`",
        f"- Included subjects: {', '.join(subjects) if subjects else '[none]' }",
        f"- Analysed EDF sessions: {n_sessions}",
        f"- Event-code source: `{code_source}`",
        "",
        "## 2. Main decoding task",
        "",
        task.task_description,
        "",
        f"- Positive class (`{task.positive.label}`): **{task.positive.name}**, event code(s): {_format_codes(task.positive.codes)}.",
        f"- Negative class (`{task.negative.label}`): **{task.negative.name}**, event code(s): {_format_codes(task.negative.codes)}.",
        "- One sample is one event-centred sEEG epoch.",
        "- Events not listed in either class are excluded from the main task.",
        "",
        "## 3. Timing definition",
        "",
        f"- The LSL marker occurs {task.marker_to_robot_onset_s:.3f} s before the defined robot movement onset.",
        f"- Robot movement onset is `t = 0`; therefore the marker is at `{marker_robot:+.3f} s`.",
        f"- Preprocessing epoch relative to marker: `{preprocessing.tmin_marker_s:+.3f} ... {preprocessing.tmax_marker_s:+.3f} s`.",
        f"- The same epoch relative to robot onset: `{epoch_robot[0]:+.3f} ... {epoch_robot[1]:+.3f} s`.",
        f"- Current decoder window relative to marker: `{task.decoder_window_marker_s[0]:+.3f} ... {task.decoder_window_marker_s[1]:+.3f} s`.",
        f"- The decoder window relative to robot onset: `{decoder_robot[0]:+.3f} ... {decoder_robot[1]:+.3f} s`.",
        f"- Requested visualisation window relative to robot onset: `{task.visual_window_robot_s[0]:+.3f} ... {task.visual_window_robot_s[1]:+.3f} s`.",
        "",
        "## 4. Preprocessing reproduced for epoch counting",
        "",
        f"- Expected sampling rate: {preprocessing.expected_sfreq:g} Hz.",
        f"- Notch frequencies: {_format_codes([int(x) for x in preprocessing.notch_freqs])} Hz.",
        f"- Band-pass: {preprocessing.l_freq:g}-{preprocessing.h_freq:g} Hz.",
        f"- Reference: {preprocessing.reference}.",
        f"- Baseline relative to marker: {preprocessing.baseline_marker_s[0]:+.3f} ... {preprocessing.baseline_marker_s[1]:+.3f} s.",
        "- Epochs exceeding the patient-specific MNE EEG peak-to-peak threshold were rejected.",
        "",
        "## 5. Accepted class counts",
        "",
        f"- Positive accepted epochs: {positive_n}",
        f"- Negative accepted epochs: {negative_n}",
        f"- Accepted total: {total_n}",
        f"- Positive fraction: {pos_fraction:.4f}" if math.isfinite(pos_fraction) else "- Positive fraction: [not available]",
        f"- Per-patient counts: `class_counts_wide.csv` ({len(wide_counts)} subjects).",
        "",
        "## 6. Completion status",
        "",
        "The technical definition is complete when all lines below are true:",
        "",
        f"- [{'x' if task.positive.codes else ' '}] Positive event code(s) are explicitly defined.",
        f"- [{'x' if task.negative.codes else ' '}] Negative event code(s) are explicitly defined.",
        f"- [{'x' if not unresolved else ' '}] Every included event code has a confirmed semantic gesture name and movement type.",
        f"- [{'x' if total_n > 0 else ' '}] At least one accepted epoch exists for the main task.",
        f"- [{'x' if positive_n > 0 and negative_n > 0 else ' '}] Both classes contain accepted epochs.",
        f"- [{'x' if not messages.errors else ' '}] All requested sessions were processed without fatal errors.",
        "",
    ]

    if unresolved:
        lines.extend(
            [
                "### Unresolved event semantics",
                "",
                f"Confirm gesture names/movement types for event code(s): {_format_codes(unresolved)}.",
                "Edit the task JSON and rerun the script; the code cannot infer these semantics from NeuronDeCo.",
                "",
            ]
        )
    if messages.warnings:
        lines.extend(
            [
                "### Warnings",
                "",
                f"See `warnings.csv` ({len(messages.warnings)} warning(s)).",
                "",
            ]
        )
    if messages.errors:
        lines.extend(
            [
                "### Errors",
                "",
                f"See `errors.csv` ({len(messages.errors)} error(s)).",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def task_spec_to_json(task: TaskSpec) -> dict[str, Any]:
    return {
        "task_name": task.task_name,
        "task_description": task.task_description,
        "positive": asdict(task.positive),
        "negative": asdict(task.negative),
        "event_names": {str(k): v for k, v in task.event_names.items()},
        "movement_types": {str(k): v for k, v in task.movement_types.items()},
        "marker_to_robot_onset_s": task.marker_to_robot_onset_s,
        "visual_window_robot_s": list(task.visual_window_robot_s),
        "decoder_window_marker_s": list(task.decoder_window_marker_s),
    }


def parse_subjects(value: str | None) -> list[str] | None:
    if value is None:
        return None
    subjects = [item.strip().lower() for item in value.split(",") if item.strip()]
    return subjects or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the frozen technical definition and epoch counts for the NeuronDeCo sEEG task."
    )
    parser.add_argument("--data-root", type=Path, help="Root containing *_sXX/session_*/*.edf")
    parser.add_argument("--config", type=Path, help="External config.py with channel lists and thresholds")
    parser.add_argument("--output-dir", type=Path, default=Path("study_definition"))
    parser.add_argument("--task-config", type=Path, default=None, help="JSON task definition; defaults to event 9 vs 1-8,10")
    parser.add_argument("--write-task-template", type=Path, default=None, help="Write a pairwise task JSON template and exit")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects, e.g. s02,s03,s04")
    parser.add_argument(
        "--channel-list-key",
        choices=("best_ch_by_power", "ch_to_keep"),
        default="best_ch_by_power",
    )
    parser.add_argument(
        "--event-code-source",
        choices=("auto", "annotation", "mne"),
        default="auto",
        help=(
            "auto prefers numeric codes written in annotation descriptions; "
            "mne reproduces mne.events_from_annotations integer IDs"
        ),
    )
    parser.add_argument(
        "--edf-strategy",
        choices=("first", "last", "biggest", "smallest"),
        default="last",
    )
    parser.add_argument("--epoch-threshold", type=float, default=None, help="Override all patient-specific thresholds")
    parser.add_argument("--expected-sfreq", type=float, default=2000.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.write_task_template is not None:
        template_path = args.write_task_template.expanduser().resolve()
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(
            json.dumps(_task_template(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Task template written to: {template_path}")
        return 0

    if args.data_root is None or args.config is None:
        parser.error("--data-root and --config are required unless --write-task-template is used")

    data_root = args.data_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    task_path = args.task_config.expanduser().resolve() if args.task_config else None

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    task = load_task_spec(task_path)
    preprocessing = PreprocessingSpec(expected_sfreq=float(args.expected_sfreq))
    config = load_external_config(config_path)
    channel_mapping = choose_channel_mapping(config, args.channel_list_key)
    thresholds = require_mapping(config, "epoch_thresh_dict")

    discovered_dirs = discover_subject_directories(data_root)
    requested_subjects = parse_subjects(args.subjects)
    if requested_subjects is None:
        subjects = sorted(set(discovered_dirs) & set(map(str.lower, channel_mapping.keys())))
    else:
        subjects = requested_subjects

    if not subjects:
        raise RuntimeError(
            "No subjects selected. Check data directory names (*_sXX), config keys, or --subjects."
        )

    messages = RunMessages()
    manifest_rows: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []

    for subject in subjects:
        subject_dirs = discovered_dirs.get(subject, [])
        if not subject_dirs:
            messages.error(subject=subject, message="No matching *_subject directory found")
            if args.fail_fast:
                raise FileNotFoundError(f"No directory found for {subject}")
            continue

        retained_channels = channel_mapping.get(subject)
        if retained_channels is None:
            messages.error(subject=subject, message=f"No {args.channel_list_key} entry in config")
            if args.fail_fast:
                raise KeyError(subject)
            continue

        threshold = choose_threshold(thresholds, subject, args.epoch_threshold)
        sessions = discover_sessions(subject_dirs, args.edf_strategy)
        if not sessions:
            messages.error(subject=subject, message="No session_* EDF files found")
            if args.fail_fast:
                raise FileNotFoundError(f"No sessions for {subject}")
            continue

        for session, edf_path in sessions:
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
                sfreq = float(raw.info["sfreq"])
                if not math.isclose(sfreq, preprocessing.expected_sfreq, rel_tol=0, abs_tol=1e-6):
                    messages.warn(
                        subject=subject,
                        session=session,
                        message=f"Sampling rate is {sfreq:g} Hz, expected {preprocessing.expected_sfreq:g} Hz",
                    )

                records, mne_mapping = annotation_records(
                    raw, code_source=args.event_code_source
                )
                if not records:
                    raise RuntimeError("EDF contains no annotations")

                # Warn when numeric annotation codes and MNE-assigned codes diverge.
                mismatches = [
                    record
                    for record in records
                    if record["annotation_numeric_code"] is not None
                    and record["mne_event_code"] is not None
                    and int(record["annotation_numeric_code"]) != int(record["mne_event_code"])
                ]
                if mismatches:
                    messages.warn(
                        subject=subject,
                        session=session,
                        message=(
                            "Numeric annotation codes differ from MNE auto-assigned IDs. "
                            f"Current source='{args.event_code_source}'. See gesture_mapping.csv."
                        ),
                    )

                clean = apply_preprocessing(
                    raw,
                    retained_channels,
                    preprocessing,
                    n_jobs=args.n_jobs,
                )
                epochs, status = build_epochs_and_status(
                    clean,
                    records,
                    task,
                    preprocessing,
                    threshold,
                )
                rows = make_manifest_rows(
                    subject=subject,
                    session=session,
                    edf_path=edf_path,
                    raw=raw,
                    retained_channels=retained_channels,
                    threshold=threshold,
                    event_records=records,
                    event_status=status,
                    task=task,
                    preprocessing=preprocessing,
                )
                manifest_rows.extend(rows)
                session_summaries.append(
                    {
                        "subject": subject,
                        "session": session,
                        "edf_path": str(edf_path),
                        "sfreq_hz": sfreq,
                        "recording_duration_s": float(raw.times[-1]) if len(raw.times) else 0.0,
                        "n_raw_channels": len(raw.ch_names),
                        "n_retained_channels": len(retained_channels),
                        "n_annotations": len(records),
                        "n_task_epochs_accepted": len(epochs) if epochs is not None else 0,
                        "epoch_threshold_v": threshold,
                        "mne_event_mapping": json.dumps(mne_mapping, ensure_ascii=False, sort_keys=True),
                    }
                )
                print(
                    f"[{subject}] {session}: {len(records)} annotations, "
                    f"{len(epochs) if epochs is not None else 0} accepted task epochs"
                )
            except Exception as exc:
                messages.error(subject=subject, session=session, message=repr(exc))
                print(f"ERROR [{subject}] {session}: {exc}", file=sys.stderr)
                if args.fail_fast:
                    raise

    manifest = pd.DataFrame(manifest_rows)
    if manifest.empty:
        # Write diagnostics before failing.
        pd.DataFrame(messages.warnings).to_csv(output_dir / "warnings.csv", index=False)
        pd.DataFrame(messages.errors).to_csv(output_dir / "errors.csv", index=False)
        raise RuntimeError(f"No manifest rows produced. See {output_dir / 'errors.csv'}")

    manifest = manifest.sort_values(
        ["subject", "session", "marker_onset_s", "annotation_index"]
    ).reset_index(drop=True)
    manifest.to_csv(output_dir / "sample_manifest.csv", index=False)
    pd.DataFrame(session_summaries).to_csv(output_dir / "session_summary.csv", index=False)

    mapping = write_gesture_mapping(
        manifest, task, output_dir / "gesture_mapping.csv"
    )
    _, wide_counts = write_class_counts(manifest, output_dir, task)
    write_trial_timing(task, preprocessing, output_dir / "trial_timing.png")

    resolved = task_spec_to_json(task)
    (output_dir / "task_config.resolved.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dataset_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "config_path": str(config_path),
        "task": resolved,
        "preprocessing": asdict(preprocessing),
        "subjects_requested": subjects,
        "subjects_processed": sorted(manifest["subject"].unique().tolist()),
        "n_sessions_processed": int(
            manifest[["subject", "session"]].drop_duplicates().shape[0]
        ),
        "n_annotations": int(len(manifest)),
        "n_task_events": int(manifest["included_in_main_task"].sum()),
        "n_task_epochs_accepted": int(
            (
                manifest["included_in_main_task"]
                & manifest["accepted_after_rejection"]
            ).sum()
        ),
        "warning_count": len(messages.warnings),
        "error_count": len(messages.errors),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if messages.warnings:
        pd.DataFrame(messages.warnings).to_csv(output_dir / "warnings.csv", index=False)
    else:
        pd.DataFrame(columns=["subject", "session", "message"]).to_csv(
            output_dir / "warnings.csv", index=False
        )
    if messages.errors:
        pd.DataFrame(messages.errors).to_csv(output_dir / "errors.csv", index=False)
    else:
        pd.DataFrame(columns=["subject", "session", "message"]).to_csv(
            output_dir / "errors.csv", index=False
        )

    write_task_definition(
        output_path=output_dir / "task_definition.md",
        data_root=data_root,
        config_path=config_path,
        task=task,
        preprocessing=preprocessing,
        subjects=subjects,
        manifest=manifest,
        wide_counts=wide_counts,
        mapping=mapping,
        code_source=args.event_code_source,
        messages=messages,
    )

    print(f"\nDone. Artefacts written to: {output_dir}")
    print("Review task_definition.md and gesture_mapping.csv first.")
    return 0 if not messages.errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
