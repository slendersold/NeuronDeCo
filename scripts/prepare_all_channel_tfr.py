#!/usr/bin/env python3
"""Build full-channel TFR files for the s09/s12 three-model benchmark."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import mne
import numpy as np

from stage3_fixed_svm_ablation import (
    Settings,
    ensure_dir,
    load_external_config,
    load_manifest,
    load_patient_epochs,
    parse_bool,
)


POSITIVE_EVENT_CODE = 9
STORED_TMIN = 0.0
STORED_TMAX = 2.0
STORED_FMAX = 59.4


def restore_manifest_event_codes(epochs: mne.Epochs) -> dict[str, object]:
    """Restore original gesture codes carried by the frozen sample manifest.

    ``stage3_fixed_svm_ablation`` uses MNE event values 1/2 internally and keeps
    the authoritative binary label and original gesture code in epoch metadata.
    The established saved-TFR pipeline, however, keeps gesture codes 1..10 and
    downstream model loaders identify ``open_hand`` by code 9.  Restore that
    established representation before computing and saving the TFR.
    """
    if epochs.metadata is None:
        raise RuntimeError("Epoch metadata is required to restore event codes")

    required = {"event_code", "class_label"}
    missing = required.difference(epochs.metadata.columns)
    if missing:
        raise RuntimeError(f"Epoch metadata is missing columns: {sorted(missing)}")

    event_codes = epochs.metadata["event_code"].astype(int).to_numpy()
    class_labels = epochs.metadata["class_label"].astype(int).to_numpy()
    if len(event_codes) != len(epochs):
        raise RuntimeError(
            f"Epoch metadata length mismatch: {len(event_codes)} != {len(epochs)}"
        )

    labels, label_counts = np.unique(class_labels, return_counts=True)
    if set(labels.tolist()) != {0, 1}:
        raise RuntimeError(
            "Expected both manifest classes 0 and 1 before TFR computation; "
            f"got {dict(zip(labels.tolist(), label_counts.tolist()))}"
        )

    labels_from_codes = (event_codes == POSITIVE_EVENT_CODE).astype(int)
    if not np.array_equal(labels_from_codes, class_labels):
        mismatch = int(np.count_nonzero(labels_from_codes != class_labels))
        raise RuntimeError(
            f"Manifest event_code/class_label mismatch in {mismatch} epochs; "
            f"positive event code is expected to be {POSITIVE_EVENT_CODE}"
        )

    epochs.events[:, 2] = event_codes
    unique_codes = sorted(np.unique(event_codes).tolist())
    epochs.event_id = {f"gesture_{code}": int(code) for code in unique_codes}

    return {
        "event_codes": unique_codes,
        "positive_event_code": POSITIVE_EVENT_CODE,
        "class_counts": {
            str(label): int(count)
            for label, count in zip(labels.tolist(), label_counts.tolist())
        },
    }


def make_settings(args: argparse.Namespace, patient: str) -> Settings:
    """Create the preprocessing settings shared with the audited raw-data path."""
    return Settings(
        patient_id=patient,
        config_path=args.config_path.expanduser().resolve(),
        study_def_path=args.study_def_path.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve(),
        output_storage=args.work_root.expanduser().resolve(),
        sessions=None,
        svm_kernel="linear",
        svm_c=1.0,
        svm_gamma="scale",
        svm_class_weight=None,
        svm_cache_mb=1024.0,
        cv_n_splits=5,
        cv_seed=42,
        classification_jobs=1,
        notch_freqs=(50.0, 100.0, 150.0),
        l_freq=0.1,
        h_freq=120.0,
        epoch_tmin=-0.9,
        epoch_tmax=2.1,
        baseline=(-0.1, 0.0),
        tfr_fmin=0.1,
        tfr_fmax=120.0,
        tfr_n_freqs=100,
        tfr_decim=2,
        tfr_batch_size=max(int(args.tfr_batch_size), 1),
        tfr_n_jobs=max(int(args.tfr_jobs), 1),
        log_power_eps=1e-20,
        baseline_time_id="T2_0ms_1000ms",
        baseline_freq_id="F0_current_0p1_59p4",
        reuse_feature_cache=True,
        overwrite_feature_cache=False,
        resume=True,
        save_svg=False,
    )


def prepare_patient(args: argparse.Namespace, patient: str) -> Path:
    settings = make_settings(args, patient)
    output_root = ensure_dir(args.output_root.expanduser().resolve())
    output_path = output_root / f"tfr_{patient}.fif"
    patient_work = ensure_dir(settings.output_storage / patient)
    audit_path = patient_work / "tfr_preparation.json"
    if output_path.exists() and audit_path.is_file() and not args.overwrite:
        with open(audit_path, encoding="utf-8") as fh:
            previous_audit = json.load(fh)
        if previous_audit.get("output") == str(output_path):
            print(f"[SKIP] verified TFR already exists: {output_path}", flush=True)
            return output_path
    if output_path.exists() and not audit_path.is_file() and not args.overwrite:
        print(
            f"[REBUILD] {output_path} has no completion audit and may be partial",
            flush=True,
        )
    if args.overwrite and audit_path.exists():
        audit_path.unlink()

    cfg = load_external_config(settings.config_path)
    configured_channels = list(cfg.ch_to_keep[patient])
    if not configured_channels:
        raise RuntimeError(f"ch_to_keep[{patient!r}] is empty")

    manifest = load_manifest(settings.study_def_path)
    manifest["accepted_after_rejection"] = parse_bool(
        manifest["accepted_after_rejection"]
    )
    manifest["included_in_main_task"] = parse_bool(
        manifest["included_in_main_task"]
    )
    rows = manifest[
        (manifest["subject"] == patient)
        & (manifest["accepted_after_rejection"] == True)
        & (manifest["included_in_main_task"] == True)
        & (manifest["class_label"].isin([0, 1]))
    ].copy()
    if rows.empty:
        raise RuntimeError(f"No accepted task epochs for {patient}")

    epochs, sessions, _ = load_patient_epochs(
        rows,
        configured_channels,
        settings,
        ensure_dir(patient_work / "data_audit"),
    )
    if [name.lower() for name in epochs.ch_names] != [
        name.lower() for name in configured_channels
    ]:
        raise RuntimeError(
            f"Resolved EDF channels differ from ch_to_keep for {patient}: "
            f"{epochs.ch_names} != {configured_channels}"
        )

    label_audit = restore_manifest_event_codes(epochs)
    print(f"[LABELS] {label_audit}", flush=True)

    all_freqs = np.linspace(
        settings.tfr_fmin,
        settings.tfr_fmax,
        settings.tfr_n_freqs,
    )
    freqs = all_freqs[all_freqs <= STORED_FMAX]
    decimated_times = epochs.times[:: settings.tfr_decim]
    time_mask = (
        (decimated_times >= STORED_TMIN - 1e-12)
        & (decimated_times <= STORED_TMAX + 1e-12)
    )
    times = decimated_times[time_mask]
    if not len(freqs) or not len(times):
        raise RuntimeError("The stored TFR frequency/time selection is empty")

    final_shape = (len(epochs), len(epochs.ch_names), len(freqs), len(times))
    memmap_path = patient_work / f"tfr_{patient}_power.float32.npy"
    partial_output = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    power_store = np.lib.format.open_memmap(
        memmap_path,
        mode="w+",
        dtype=np.float32,
        shape=final_shape,
    )
    print(
        f"[TFR] batched float32 shape={final_shape} "
        f"batch_size={settings.tfr_batch_size} n_jobs={settings.tfr_n_jobs}",
        flush=True,
    )

    try:
        for start in range(0, len(epochs), settings.tfr_batch_size):
            stop = min(start + settings.tfr_batch_size, len(epochs))
            batch_data = epochs[start:stop].get_data()
            batch_power = mne.time_frequency.tfr_array_morlet(
                batch_data,
                sfreq=float(epochs.info["sfreq"]),
                freqs=freqs,
                n_cycles=freqs / 2.0,
                zero_mean=False,
                use_fft=True,
                decim=settings.tfr_decim,
                output="power",
                n_jobs=settings.tfr_n_jobs,
                verbose=False,
            )
            power_store[start:stop] = batch_power[..., time_mask].astype(
                np.float32,
                copy=False,
            )
            power_store.flush()
            del batch_data, batch_power
            gc.collect()
            print(f"[TFR] epochs {start}:{stop}/{len(epochs)}", flush=True)

        tfr = mne.time_frequency.EpochsTFR(
            epochs.info.copy(),
            power_store,
            times,
            freqs,
            comment="all-channel model benchmark",
            method="morlet",
            events=epochs.events.copy(),
            event_id=dict(epochs.event_id),
            metadata=epochs.metadata.reset_index(drop=True).copy(),
            verbose=False,
        )
        if partial_output.exists():
            partial_output.unlink()
        tfr.save(partial_output, overwrite=True)
        partial_output.replace(output_path)
        del tfr
    finally:
        del power_store
        gc.collect()
        if memmap_path.exists():
            memmap_path.unlink()

    audit = {
        "patient": patient,
        "channels": list(epochs.ch_names),
        "n_channels": len(epochs.ch_names),
        "n_epochs": len(epochs),
        "sessions": sessions,
        "shape": list(final_shape),
        "dtype": "float32",
        "stored_frequency_hz": [float(freqs[0]), float(freqs[-1])],
        "stored_time_s": [float(times[0]), float(times[-1])],
        "labels": label_audit,
        "output": str(output_path),
        "settings": asdict(settings),
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[SAVED] {output_path} shape={final_shape} dtype=float32", flush=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare full-ch_to_keep TFR files for confirmatory model evaluation."
    )
    parser.add_argument("--patients", nargs="+", default=["s09", "s12"])
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--study-def-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--tfr-jobs", type=int, default=2)
    parser.add_argument("--tfr-batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for patient in args.patients:
        prepare_patient(args, str(patient))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
