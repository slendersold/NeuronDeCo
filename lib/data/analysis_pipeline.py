"""
EDF -> RAW preprocess -> epochs helpers.

Перенос из legacy `utils.analysis_pipeline` с устранением зависимости от `utils.*`:
конфиги каналов/порогов нужно передавать в аргументах.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np

from lib.data.edf_tools import find_patient_edf


def load_and_preprocess(
    root: str,
    subject: str,
    *,
    strategy: str = "last",
    plot_psd: bool = True,
    autoclean: bool = True,
    ch_to_keep: Mapping[str, Sequence[str]] | None = None,
    best_ch_by_power: Mapping[str, Sequence[str]] | None = None,
) -> list[mne.io.BaseRaw]:
    """
    Загружает и предобрабатывает RAW для subject из session_1..session_{session_num}.
    """
    patient_data = find_patient_edf(root, subject, strategy=strategy)
    raws: list[mne.io.BaseRaw] = []

    for session_key, edf_path in patient_data["sessions"].items():
        if edf_path is None:
            raise FileNotFoundError(
                f"Не найден EDF для {subject} в {session_key} (strategy='{strategy}')."
            )

        raw = mne.io.read_raw_edf(str(edf_path), preload=True)

        if autoclean:
            if best_ch_by_power is None:
                raise ValueError("autoclean=True requires best_ch_by_power mapping")
            keep = best_ch_by_power.get(subject)
        else:
            if ch_to_keep is None:
                raise ValueError("autoclean=False requires ch_to_keep mapping")
            keep = ch_to_keep.get(subject)

        if keep is None:
            raise KeyError(f"Channel keep list not found for subject='{subject}'")

        ch_to_drop = [ch for ch in raw.ch_names if ch not in keep]
        if ch_to_drop:
            raw.drop_channels(ch_to_drop)

        if plot_psd:
            raw.plot_psd()
            plt.show()

        raws.append(raw)

    return raws


def create_epochs(
    files_list: Mapping[str, list[mne.io.BaseRaw]],
    subject: str,
    *,
    threshold: float | None = None,
    event_id: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 10),
    ctx: float = 0.2,
    epoch_thresh_dict: Mapping[str, float] | None = None,
) -> tuple[list[mne.Epochs], dict[str, Any]]:
    if threshold is None:
        if epoch_thresh_dict is None:
            raise ValueError("threshold is None but epoch_thresh_dict is not provided")
        threshold = float(epoch_thresh_dict[subject])

    epochs_list: list[mne.Epochs] = []
    event_dict: dict[str, Any] = {}

    for raw in files_list[subject]:
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(
            raw,
            events=events,
            tmin=0.0 - ctx,
            tmax=1.9 + ctx,
            baseline=(0, 0.1),
            reject={"eeg": float(threshold)},
            preload=True,
            event_id=list(event_id),
        )
        epochs_list.append(epochs)

    return epochs_list, event_dict


def save_epochs(
    subject: str,
    epochs_list: Sequence[mne.Epochs],
    *,
    out_dir: str = "../1_Epochs",
) -> mne.Epochs:
    epochs_united = mne.concatenate_epochs(list(epochs_list))
    events_united = epochs_united.events

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    epochs_united.save(out_path / f"{subject}_epochs-epo.fif", overwrite=True)
    np.savetxt(out_path / f"{subject}_events.csv", events_united, fmt="%d")
    return epochs_united


def load_epochs(subject: str, *, out_dir: str = "1_Epochs") -> tuple[mne.Epochs, np.ndarray]:
    """
    Загрузить сохранённые epochs и события для заданного пациента.
    """
    out_path = Path(out_dir)
    epochs_path = out_path / f"{subject}_epochs-epo.fif"
    events_path = out_path / f"{subject}_events.csv"

    if not epochs_path.exists():
        raise FileNotFoundError(f"File not found: {epochs_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"File not found: {events_path}")

    epochs = mne.read_epochs(epochs_path, preload=True)
    events = np.loadtxt(events_path, dtype=int)
    return epochs, events


def plot_epochs_images(epochs: mne.Epochs) -> None:
    for i in range(len(epochs.ch_names)):
        epochs.plot_image(i)

