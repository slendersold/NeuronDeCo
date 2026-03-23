"""Циклы обучения / валидации для классификаторов TFR (только ``lib``)."""

from lib.training.epochs import eval_one_epoch_f1_macro, train_one_epoch

__all__ = ["eval_one_epoch_f1_macro", "train_one_epoch"]
