import mne
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from utils.config import ch_to_keep, best_ch_by_power
from utils.edf_tools import find_patient_edf, auto_clean_channels
from utils.config import epoch_thresh_dict 
from mne.preprocessing import find_bad_channels_maxwell

def load_and_preprocess(root: str, subject: str,
                        strategy: str = "last", plot_psd: bool = True, autoclean: bool = True):
    """
    Загружает и предобрабатывает RAW для subject из session_1..session_{session_num},
    используя библиотеку поиска EDF (find_patient_edf) и конфиг каналов (ch_to_keep).

    strategy: 'first' | 'last' | 'biggest' | 'smallest' — как выбирать файл в каждой папке.
    """
    patient_data = find_patient_edf(root, subject, strategy=strategy)
    files_list = []

    for session_key, edf_path in patient_data["sessions"].items():
        if edf_path is None:
            raise FileNotFoundError(
                f"Не найден EDF для {subject} в {session_key} (strategy='{strategy}')."
            )

        raw = mne.io.read_raw_edf(edf_path, preload=True)
        if autoclean:
            # raw = auto_clean_channels(raw)
            # keep = mne.pick_types(raw.info, meg=True, eeg=True, emg=True, misc=True, exclude='bads')
            keep = best_ch_by_power.get(subject)
        else:
            keep = ch_to_keep.get(subject)
            
        if keep is None:
            raise KeyError(f"В config.ch_to_keep нет записи для subject='{subject}'")

        # дропаем все каналы, которых нет в списке keep
        ch_to_drop = [ch for ch in raw.ch_names if ch not in keep]
        if ch_to_drop:
            raw.drop_channels(ch_to_drop)

        if plot_psd:
            raw.plot_psd()
            plt.show()

        files_list.append(raw)

    return files_list

def create_epochs(files_list, subject, threshold = None, event_id = [1,2,3,4,5,6,7,8,10], ctx = 0.2):
    if threshold == None:
        threshold = epoch_thresh_dict[subject] 
    epochs_list = []

    for raw in files_list[subject]:
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(
            raw, events=events,
            tmin=0.0 - ctx, tmax=1.9 + ctx,
            baseline=(0, 0.1),
            reject={'eeg': threshold},
            preload=True,
            event_id=event_id
        )
        epochs_list.append(epochs)

    return epochs_list, event_dict

def save_epochs(subject, epochs_list, out_dir="../1_Epochs"):
    epochs_united = mne.concatenate_epochs(epochs_list)
    events_united = epochs_united.events

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs_united.save(out_dir / f"{subject}_epochs-epo.fif", overwrite=True)
    np.savetxt(out_dir / f"{subject}_events.csv", events_united, fmt="%d")

    return epochs_united

def load_epochs(subject: str, out_dir: str = "1_Epochs"):
    """
    Загрузить сохранённые epochs и события для заданного пациента.
    
    Parameters
    ----------
    subject : str
        ID пациента (например 's09').
    out_dir : str
        Папка, где лежат сохранённые файлы (по умолчанию '1_Epochs').
    
    Returns
    -------
    epochs : mne.Epochs
        Загруженные эпо́хи.
    events : np.ndarray
        Массив событий (из CSV).
    """
    out_dir = Path(out_dir)
    epochs_path = out_dir / f"{subject}_epochs-epo.fif"
    events_path = out_dir / f"{subject}_events.csv"

    if not epochs_path.exists():
        raise FileNotFoundError(f"Файл {epochs_path} не найден")
    if not events_path.exists():
        raise FileNotFoundError(f"Файл {events_path} не найден")

    epochs = mne.read_epochs(epochs_path, preload=True)
    events = np.loadtxt(events_path, dtype=int)

    return epochs, events

def plot_epochs_images(epochs):
    for i in range(len(epochs.ch_names)):
        epochs.plot_image(i)

