"""
Преобработка RAW перед эпохами/TFR.

Сюда вынесены шаги из legacy-ноутбука: notch + bandpass + CAR (global average reference).
"""

from __future__ import annotations

from typing import Sequence

import mne
import numpy as np


def apply_notch_bandpass_car(
    raw: mne.io.BaseRaw,
    *,
    notch_freqs: Sequence[float],
    l_freq: float,
    h_freq: float,
    method: str = "iir",
    n_jobs: int = -1,
) -> mne.io.BaseRaw:
    """
    Выполнить:
      1) notch фильтрацию
      2) bandpass фильтрацию
      3) CAR: x_i(t) = x_i(t) - mean_j x_j(t)
    """
    # важно: фильтры требуют данных в памяти
    r = raw.copy().load_data()

    # МНЕ допускает notch_filter по частотам-списку.
    # 1) notch (сетевой шум и гармоники)
    r.notch_filter(freqs=np.asarray(list(notch_freqs)), n_jobs=n_jobs)

    # 2) bandpass
    r.filter(l_freq=l_freq, h_freq=h_freq, method=method)

    X = r.get_data()  # (n_ch, n_times)

    # 3) global average reference (CAR): x_i(t) <- x_i(t) - mean_j x_j(t)
    X = X - X.mean(axis=0, keepdims=True)

    info = r.info.copy()
    r_car = mne.io.RawArray(X, info)

    # переносим аннотации (события)
    r_car.set_annotations(r.annotations)
    return r_car

