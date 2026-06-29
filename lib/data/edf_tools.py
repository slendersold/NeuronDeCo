"""
EDF discovery + auto-cleaning helpers.

Это перенос из legacy `utils.edf_tools`, но без зависимостей от `utils.*`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import mne
from scipy.signal import welch


def _band_power(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    f, Pxx = welch(data, sfreq, nperseg=int(sfreq * 1.0))
    band = (f >= fmin) & (f <= fmax)
    return np.trapz(Pxx[:, band], f[band], axis=1)


def robust_cut_hi(x: np.ndarray, k: float = 8.0) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-12
    return med + k * mad


def robust_cut_lo(x: np.ndarray, k: float = 3.0) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-12
    return med - k * mad


def flag_bad_eeg(
    raw: Any,
    win_s: float = 1.0,
    step_s: float = 0.5,
    k_amp: float = 8.0,
    k_corr: float = 3.0,
    hf_quant: float = 98,
    line_quant: float = 98,
    frac_bad_thresh: float = 0.25,
) -> list[str]:
    picks = mne.pick_types(raw.info, eeg=True)
    if len(picks) == 0:
        return []

    sf = float(raw.info["sfreq"])
    n = int(win_s * sf)
    hop = int(step_s * sf)
    X = raw.get_data(picks=picks)  # (C, T)

    idx = np.arange(0, X.shape[1] - n + 1, hop)
    C = int(len(picks))
    W = int(len(idx))

    RMS = np.empty((C, W))
    STD = np.empty((C, W))
    HF_over_SF = np.empty((C, W))
    LINE50 = np.empty((C, W))

    for wi, s in enumerate(idx):
        seg = X[:, s : s + n]
        RMS[:, wi] = np.sqrt(np.mean(seg**2, axis=1))
        STD[:, wi] = np.std(seg, axis=1)
        hf = _band_power(seg, sf, 40, min(100, sf / 2 - 1))
        sfp = _band_power(seg, sf, 1, 40)
        HF_over_SF[:, wi] = (hf + 1e-12) / (sfp + 1e-12)
        line = _band_power(seg, sf, 49, 51)
        neigh = _band_power(seg, sf, 46, 54) - line
        LINE50[:, wi] = (line + 1e-12) / (neigh + 1e-12)

    median_ref = np.median(X, axis=0, keepdims=True)
    MEAN_CORR = np.empty(C)
    for ci in range(C):
        corr_vals: list[float] = []
        for wi, s in enumerate(idx):
            r = np.corrcoef(X[ci, s : s + n], median_ref[0, s : s + n])[0, 1]
            if np.isfinite(r):
                corr_vals.append(float(r))
        MEAN_CORR[ci] = float(np.median(corr_vals)) if corr_vals else 0.0

    amp_thr = robust_cut_hi(RMS.ravel(), k=k_amp)
    hf_thr = float(np.percentile(HF_over_SF.ravel(), hf_quant))
    line_thr = float(np.percentile(LINE50.ravel(), line_quant))
    corr_thr = robust_cut_lo(MEAN_CORR, k=k_corr)

    bad: list[str] = []
    for ci, ch in enumerate(np.array(raw.ch_names)[picks]):
        f_amp = float(np.mean(RMS[ci] > amp_thr))
        f_hf = float(np.mean(HF_over_SF[ci] > hf_thr))
        f_line = float(np.mean(LINE50[ci] > line_thr))
        lowcorr = bool(MEAN_CORR[ci] < corr_thr)
        flat = float(np.mean(STD[ci] < np.percentile(STD, 2)))

        if (
            (f_amp >= frac_bad_thresh)
            or (f_hf >= frac_bad_thresh)
            or (f_line >= frac_bad_thresh)
            or (flat >= frac_bad_thresh)
            or lowcorr
        ):
            bad.append(str(ch))
    return bad


def flag_bad_emg(raw: Any, frac_bad_thresh: float = 0.25, k_amp: float = 10.0) -> list[str]:
    picks = mne.pick_types(raw.info, emg=True)
    if len(picks) == 0:
        return []

    sf = float(raw.info["sfreq"])
    X = raw.get_data(picks=picks)
    rms = np.sqrt(np.mean(X**2, axis=1))
    thr = robust_cut_hi(rms, k=k_amp)
    std = np.std(X, axis=1)
    low_thr = float(np.percentile(std, 2))

    bad: list[str] = []
    for i, ch in enumerate(np.array(raw.ch_names)[picks]):
        if (rms[i] > thr) or (std[i] < low_thr):
            bad.append(str(ch))
    return bad


def flag_bad_acc(raw: Any) -> list[str]:
    picks = [
        i
        for i, ch in enumerate(raw.info["chs"])
        if raw.get_channel_types()[i] == "misc"
        and ("acc" in raw.ch_names[i].lower() or "gyr" in raw.ch_names[i].lower())
    ]
    bad: list[str] = []
    if not picks:
        return bad

    X = raw.get_data(picks=picks)
    std = np.std(X, axis=1)
    if len(std) >= 3:
        lo = float(np.percentile(std, 2))
        hi = float(np.percentile(std, 98))
    else:
        lo = float(np.min(std) * 0.5)
        hi = float(np.max(std) * 2)

    for i, ch in enumerate(np.array(raw.ch_names)[picks]):
        if (std[i] < lo) or (std[i] > hi):
            bad.append(str(ch))
    return bad


def auto_clean_channels(
    raw: Any,
    *,
    z_thresh: float = 5.0,
    corr_thresh: float = 0.4,
    verbose: bool = True,
) -> Any:
    bad_eeg = flag_bad_eeg(raw)
    bad_emg = flag_bad_emg(raw)
    bad_acc = flag_bad_acc(raw)
    raw.info["bads"] = list(sorted(set(raw.info.get("bads", []) + bad_eeg + bad_emg + bad_acc)))
    return raw


def pick_edf_file_in_subfolder(folder: Path, strategy: str = "last") -> Path | None:
    files = [f for f in folder.glob("*.edf") if "_raw" not in f.name]
    if not files:
        return None

    strategies = {
        "first": lambda flist: sorted(flist)[0],
        "last": lambda flist: sorted(flist)[-1],
        "biggest": lambda flist: max(flist, key=lambda f: f.stat().st_size),
        "smallest": lambda flist: min(flist, key=lambda f: f.stat().st_size),
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy '{strategy}', available: {list(strategies.keys())}")
    return strategies[strategy](files)


def find_patient_edf(root: str, subject: str, strategy: str = "last") -> dict[str, Any]:
    """
    Найти EDF-файлы пациента внутри подпапок rest/ и session_*/.
    """
    root_path = Path(root)
    patient_dirs = list(root_path.glob(f"*_{subject}"))
    if not patient_dirs:
        raise FileNotFoundError(f"No directories found for subject {subject}")

    result: dict[str, Any] = {"rest": None, "sessions": {}}
    for patient_dir in patient_dirs:
        rest_dir = patient_dir / "rest"
        if rest_dir.exists():
            result["rest"] = pick_edf_file_in_subfolder(rest_dir, strategy)

        for session_dir in patient_dir.glob("session_*"):
            if session_dir.is_dir():
                chosen = pick_edf_file_in_subfolder(session_dir, strategy)
                result["sessions"][session_dir.name] = chosen
    return result

