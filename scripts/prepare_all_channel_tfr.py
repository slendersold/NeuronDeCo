#!/usr/bin/env python3
"""Build full-channel TFR files for the s09/s12 three-model benchmark."""

from __future__ import annotations

import argparse
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
        tfr_batch_size=24,
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
    if output_path.exists() and not args.overwrite:
        print(f"[SKIP] {output_path} already exists", flush=True)
        return output_path

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

    patient_work = ensure_dir(settings.output_storage / patient)
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

    freqs = np.linspace(settings.tfr_fmin, settings.tfr_fmax, settings.tfr_n_freqs)
    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=freqs / 2.0,
        return_itc=False,
        decim=settings.tfr_decim,
        average=False,
        n_jobs=settings.tfr_n_jobs,
    )
    tfr.save(output_path, overwrite=True)

    audit = {
        "patient": patient,
        "channels": list(tfr.ch_names),
        "n_channels": len(tfr.ch_names),
        "n_epochs": len(tfr),
        "sessions": sessions,
        "shape": list(tfr.data.shape),
        "output": str(output_path),
        "settings": asdict(settings),
    }
    (patient_work / "tfr_preparation.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[SAVED] {output_path} shape={tfr.data.shape}", flush=True)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for patient in args.patients:
        prepare_patient(args, str(patient))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
