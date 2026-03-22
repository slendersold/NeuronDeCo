from pathlib import Path

import numpy as np
import mne

import numpy as np
import mne
from scipy.signal import welch

def _band_power(data, sfreq, fmin, fmax):
    f, Pxx = welch(data, sfreq, nperseg=int(sfreq*1.0))
    band = (f >= fmin) & (f <= fmax)
    # мощность на окно: усредняем по частотам, потом по времени
    return np.trapz(Pxx[:, band], f[band], axis=1)

def robust_cut_hi(x, k=8.0):
    med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-12
    return med + k * mad

def robust_cut_lo(x, k=3.0):
    med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-12
    return med - k * mad

def flag_bad_eeg(raw, win_s=1.0, step_s=0.5,
                 k_amp=8.0, k_corr=3.0, hf_quant=98, line_quant=98,
                 frac_bad_thresh=0.25):
    picks = mne.pick_types(raw.info, eeg=True)
    if len(picks) == 0:
        return []

    sf = raw.info['sfreq']
    n = int(win_s * sf); hop = int(step_s * sf)
    X = raw.get_data(picks=picks)  # (C, T)

    # Окна
    idx = np.arange(0, X.shape[1] - n + 1, hop)
    C = len(picks); W = len(idx)

    # Метрики по окнам
    RMS = np.empty((C, W))
    STD = np.empty((C, W))
    HF_over_SF = np.empty((C, W))
    LINE50 = np.empty((C, W))
    for wi, s in enumerate(idx):
        seg = X[:, s:s+n]
        RMS[:, wi] = np.sqrt(np.mean(seg**2, axis=1))
        STD[:, wi] = np.std(seg, axis=1)
        # спектральные метрики на всём окне
        # (для скорости можно считать реже или кэшировать)
        # HF/SF
        hf = _band_power(seg, sf, 40, min(100, sf/2-1))
        sfp = _band_power(seg, sf, 1, 40)
        HF_over_SF[:, wi] = (hf + 1e-12) / (sfp + 1e-12)
        # line 50 Hz vs соседи
        # приближенно: отношение мощности в 49–51 к 46–54 без 49–51
        line = _band_power(seg, sf, 49, 51)
        neigh = _band_power(seg, sf, 46, 54) - line
        LINE50[:, wi] = (line + 1e-12) / (neigh + 1e-12)

    # Корреляции (с медианой по каналам)
    median_ref = np.median(X, axis=0, keepdims=True)
    # усредняем корреляцию по окнам (быстро и робастно)
    MEAN_CORR = np.empty(C)
    for ci in range(C):
        corr_vals = []
        for wi, s in enumerate(idx):
            r = np.corrcoef(X[ci, s:s+n], median_ref[0, s:s+n])[0,1]
            if np.isfinite(r):
                corr_vals.append(r)
        MEAN_CORR[ci] = np.median(corr_vals) if corr_vals else 0.0

    # Пороговые значения
    amp_thr = robust_cut_hi(RMS.ravel(), k=k_amp)
    hf_thr  = np.percentile(HF_over_SF.ravel(), hf_quant)
    line_thr = np.percentile(LINE50.ravel(), line_quant)
    corr_thr = robust_cut_lo(MEAN_CORR, k=k_corr)

    bad = []
    for ci, ch in enumerate(np.array(raw.ch_names)[picks]):
        f_amp  = np.mean(RMS[ci] > amp_thr)
        f_hf   = np.mean(HF_over_SF[ci] > hf_thr)
        f_line = np.mean(LINE50[ci] > line_thr)
        lowcorr = MEAN_CORR[ci] < corr_thr
        flat = np.mean(STD[ci] < np.percentile(STD, 2))  # локальный P2

        # «или»-логика + доля окон
        if (f_amp >= frac_bad_thresh) or (f_hf >= frac_bad_thresh) or (f_line >= frac_bad_thresh) \
           or (flat >= frac_bad_thresh) or lowcorr:
            bad.append(ch)
    return bad

def flag_bad_emg(raw, frac_bad_thresh=0.25, k_amp=10.0):
    picks = mne.pick_types(raw.info, emg=True)
    if len(picks) == 0:
        return []
    sf = raw.info['sfreq']
    # можно по-быстрому: глобальные std/RMS и MAD
    X = raw.get_data(picks=picks)
    rms = np.sqrt(np.mean(X**2, axis=1))
    thr = robust_cut_hi(rms, k=k_amp)
    std = np.std(X, axis=1)
    low_thr = np.percentile(std, 2)
    bad = []
    for i, ch in enumerate(np.array(raw.ch_names)[picks]):
        if (rms[i] > thr) or (std[i] < low_thr):
            bad.append(ch)
    return bad

def flag_bad_acc(raw):
    # простая проверка динамики
    picks = [i for i, ch in enumerate(raw.info['chs']) if raw.get_channel_types()[i] == 'misc' and
             ('acc' in raw.ch_names[i].lower() or 'gyr' in raw.ch_names[i].lower())]
    bad = []
    if not picks: return bad
    X = raw.get_data(picks=picks)
    std = np.std(X, axis=1)
    if len(std) >= 3:
        lo = np.percentile(std, 2); hi = np.percentile(std, 98)
    else:
        lo, hi = np.min(std)*0.5, np.max(std)*2
    for i, ch in enumerate(np.array(raw.ch_names)[picks]):
        if (std[i] < lo) or (std[i] > hi):
            bad.append(ch)
    return bad


def auto_clean_channels(raw, z_thresh=5.0, corr_thresh=0.4, verbose=True):
    
    bad_eeg = flag_bad_eeg(raw)          # только EEG-логика
    bad_emg = flag_bad_emg(raw)          # простая проверка EMG
    bad_acc = flag_bad_acc(raw)          # диапазон для акселя
    raw.info['bads'] = list(sorted(set(raw.info.get('bads', []) + bad_eeg + bad_emg + bad_acc)))

    return raw


def pick_edf_file_in_subfolder(folder: Path, strategy: str = "last"):
    """Выбрать EDF-файл в конкретной папке (rest или session_x)."""
    files = [f for f in folder.glob("*.edf") if "_raw" not in f.name]
    if not files:
        return None

    strategies = {
        "first":    lambda flist: sorted(flist)[0],
        "last":     lambda flist: sorted(flist)[-1],
        "biggest":  lambda flist: max(flist, key=lambda f: f.stat().st_size),
        "smallest": lambda flist: min(flist, key=lambda f: f.stat().st_size),
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy '{strategy}', "
                         f"available: {list(strategies.keys())}")

    return strategies[strategy](files)


def find_patient_edf(root: str, subject: str, strategy: str = "last") -> dict:
    """
    Найти EDF-файлы пациента внутри подпапок rest/ и session_*/.

    Returns словарь вида:
    {
        "rest": Path | None,
        "sessions": {
            "session_1": Path | None,
            "session_2": Path | None,
            ...
        }
    }
    """
    root_path = Path(root)
    patient_dirs = list(root_path.glob(f"*_{subject}"))
    if not patient_dirs:
        raise FileNotFoundError(f"No directories found for subject {subject}")

    result = {"rest": None, "sessions": {}}

    for patient_dir in patient_dirs:
        # rest
        rest_dir = patient_dir / "rest"
        if rest_dir.exists():
            result["rest"] = pick_edf_file_in_subfolder(rest_dir, strategy)

        # session_*
        for session_dir in patient_dir.glob("session_*"):
            if session_dir.is_dir():
                chosen = pick_edf_file_in_subfolder(session_dir, strategy)
                result["sessions"][session_dir.name] = chosen

    return result


# # === Пример использования ===
# root = r"C:\Users\user\Desktop\PirogovDATA"
# subject = "s09"

# patient_data = find_patient_edf(root, subject, strategy="last")
# print(patient_data)
