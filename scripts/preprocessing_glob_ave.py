import sys
from pathlib import Path

# === Настройка путей проекта ===
project_root = Path("/trinity/home/asma.benachour/notebooks/Pirogov/MNE_playground")
pirogov_root = Path("/trinity/home/asma.benachour/notebooks/Pirogov/PirogovDATA")

sys.path.append(str(project_root))
sys.path.append(str(pirogov_root))
print("Added to PYTHONPATH:", project_root, pirogov_root)

# === Импорты проекта ===
from lib.data.analysis_pipeline import (
    load_and_preprocess,
    create_epochs,
    save_epochs,
)
from lib.data.raw_preprocessing import apply_notch_bandpass_car
from lib.data.config import best_ch_by_power, epoch_thresh_dict

# === Библиотеки ===
import sys
from pathlib import Path
import numpy as np
import mne


import numpy as np
import mne

def run_car_epochs_tfr_all_subjects(
    root: str,
    subjects: list,
    *,
    strategy: str = "last",
    sfreq: int | None = None,
    events_avg=(1, 2, 3, 4, 5, 6, 7, 8, 10),
    event_single=(9,),
    l_freq: float = 1.0,
    h_freq: float = 150.0,
    notch_base: float = 50.0,
    notch_max: float | None = None,   # если None -> sfreq/2
    notch_n_jobs: int = -1,
    epochs_ctx: float = 0.2,
    epochs_threshold=None,            # если None -> epoch_thresh_dict[subject] внутри create_epochs
    tfr_fmin: float = 0.1,
    tfr_fmax: float = 120.0,
    tfr_n_freqs: int = 100,
    tfr_decim: int = 2,
    tfr_out_dir: str = "../",
    save_tfr: bool = True,
):
    """
    Проходчик по всем subject:
      load_and_preprocess -> notch + bandpass -> CAR -> create_epochs -> tfr_morlet -> save

    Требования:
      - load_and_preprocess(root, subject, strategy=...)
      - create_epochs(files_list_dict, subject, threshold=..., event_id=..., ctx=...) возвращает (epochs_list, event_dict)
        В твоей реализации ты берешь [0], т.е. epochs_list.
    """

    results = {
        "files_list": {},
        "clean": {},
        "epochs_list": {},
        "tfr_paths": {},
        "errors": {},
    }

    # частоты для TFR
    freqs = np.linspace(tfr_fmin, tfr_fmax, tfr_n_freqs)
    n_cycles = freqs / 2.0

    for subject in subjects:
        try:
            # 1) load
            raws = load_and_preprocess(
                root,
                subject,
                strategy=strategy,
                plot_psd=False,
                autoclean=True,
                best_ch_by_power=best_ch_by_power,
            )
            results["files_list"][subject] = raws

            # 2) determine sfreq (если не дали вручную)
            sf = sfreq if sfreq is not None else float(raws[0].info["sfreq"])

            # 3) notch freqs
            max_f = (sf / 2.0) if notch_max is None else float(notch_max)
            notch_freqs = np.arange(notch_base, max_f, notch_base)

            # 4) preprocess: notch + bandpass -> CAR
            clean_list = []
            for raw in raws:
                r_car = apply_notch_bandpass_car(
                    raw,
                    notch_freqs=notch_freqs,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    method="iir",
                    n_jobs=notch_n_jobs,
                )
                clean_list.append(r_car)

            results["clean"][subject] = clean_list

            # 5) epochs (create_epochs у тебя ожидает dict[subject] -> list[raw])
            clean_dict = {subject: clean_list}
            epochs_list, event_dict = create_epochs(
                clean_dict,
                subject,
                threshold=epochs_threshold,                 # если None — внутри create_epochs возьмется epoch_thresh_dict[subject]
                event_id=[*events_avg, *event_single],
                ctx=epochs_ctx,
                epoch_thresh_dict=epoch_thresh_dict,
            )
            results["epochs_list"][subject] = epochs_list

            # 6) TFR на первой сессии (как у тебя epochs[0])
            tfr = mne.time_frequency.tfr_morlet(
                epochs_list[0],
                freqs=freqs,
                n_cycles=n_cycles,
                return_itc=False,
                decim=tfr_decim,
                average=False,
            )

            if save_tfr:
                out_path = f"{tfr_out_dir.rstrip('/')}/tfr_{subject}.fif"
                tfr.save(out_path, overwrite=True)
                results["tfr_paths"][subject] = out_path

        except Exception as e:
            results["errors"][subject] = repr(e)

    return results


# 1) 
subjects = sorted(best_ch_by_power.keys())

res = run_car_epochs_tfr_all_subjects(
    root=pirogov_root,
    subjects=subjects,
    strategy="last",
    # sfreq=2000,          # если уверен, что везде так; иначе убери и он возьмёт из raw.info['sfreq']
    tfr_out_dir=f"{project_root}/car",
    save_tfr=True,
)

print("Errors:", res["errors"])
print("Saved TFR:", res["tfr_paths"])
