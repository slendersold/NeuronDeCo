"""Данные и preprocessing для обучения на TFR (только ``lib``, без ``utils``)."""

from lib.data.tfr_dataset import TFRDataset
from lib.data.normalisation import normalize_tfr_robust
from lib.data.raw_preprocessing import apply_notch_bandpass_car
from lib.data.analysis_pipeline import (
    create_epochs,
    load_and_preprocess,
    load_epochs,
    plot_epochs_images,
    save_epochs,
)
from lib.data.config import best_ch_by_power, ch_to_keep, epoch_thresh_dict

__all__ = [
    "TFRDataset",
    "normalize_tfr_robust",
    "apply_notch_bandpass_car",
    "load_and_preprocess",
    "create_epochs",
    "save_epochs",
    "load_epochs",
    "plot_epochs_images",
    "ch_to_keep",
    "best_ch_by_power",
    "epoch_thresh_dict",
]
