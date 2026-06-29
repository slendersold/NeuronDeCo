from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import optuna
from optuna.study import Study
from optuna.trial import FrozenTrial


FoldCurve = Mapping[str, Any]


def _find_trial_by_number(study: Study, trial_number: int) -> FrozenTrial:
    for t in study.get_trials(deepcopy=False):
        if t.number == trial_number:
            return t
    raise KeyError(f"Trial with number={trial_number} not found in study.")


def plot_trial_fold_curves(
    study: Study,
    trial_number: int,
    *,
    fold_curves_attr: str = "fold_curves",
    split_name_key: str = "split",
    show: bool = True,
) -> list[plt.Figure]:
    """
    Строит loss и f1 curves по фолдам для одного trial.

    Ожидаемая номенклатура (как в ноутбуках):
    - ``t.user_attrs[fold_curves_attr]``: список dict, где на каждом элементе:
      ``train_losses``, ``val_losses``, ``val_f1s`` и (обычно) ``split``.
    - fallback (если fold_curves отсутствует):
      ``t.user_attrs['train_losses']``, ``t.user_attrs['val_losses']``,
      ``t.user_attrs['val_f1s']``.
    """

    t = _find_trial_by_number(study, trial_number)
    fold_curves = t.user_attrs.get(fold_curves_attr, None)  # can be missing / None

    figs: list[plt.Figure] = []

    if fold_curves is not None and len(fold_curves) > 0:
        # Loss curves по фолдам.
        fig_loss = plt.figure(constrained_layout=True)
        for fc in fold_curves:
            name = fc.get(split_name_key, "fold")
            tr_losses = fc.get("train_losses", []) or []
            va_losses = fc.get("val_losses", []) or []
            if tr_losses:
                plt.plot(tr_losses, label=f"{name} train")
            if va_losses:
                plt.plot(va_losses, label=f"{name} val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Loss curves by fold (trial {t.number})")
        plt.legend()
        figs.append(fig_loss)
        if show:
            plt.show()

        # F1 curves по фолдам.
        fig_f1 = plt.figure(constrained_layout=True)
        for fc in fold_curves:
            name = fc.get(split_name_key, "fold")
            va_f1s = fc.get("val_f1s", []) or []
            if va_f1s:
                plt.plot(va_f1s, label=f"{name} val f1_macro")
        plt.xlabel("epoch")
        plt.ylabel("f1_macro")
        plt.title(f"Validation F1_macro by fold (trial {t.number})")
        plt.legend()
        figs.append(fig_f1)
        if show:
            plt.show()
        return figs

    # Fallback: старая номенклатура без fold_curves.
    train_losses = t.user_attrs.get("train_losses", []) or []
    val_losses = t.user_attrs.get("val_losses", []) or []
    val_f1s = t.user_attrs.get("val_f1s", []) or []

    fig_loss = plt.figure(constrained_layout=True)
    if train_losses:
        plt.plot(train_losses, label="train loss")
    if val_losses:
        plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss curves (trial {t.number})")
    plt.legend()
    figs.append(fig_loss)
    if show:
        plt.show()

    fig_f1 = plt.figure(constrained_layout=True)
    if val_f1s:
        plt.plot(val_f1s, label="val f1_macro")
    plt.xlabel("epoch")
    plt.ylabel("f1_macro")
    plt.title(f"Validation F1_macro (trial {t.number})")
    plt.legend()
    figs.append(fig_f1)
    if show:
        plt.show()

    return figs


def plot_trials_fold_curves(
    study: Study,
    trial_numbers: Sequence[int],
    *,
    fold_curves_attr: str = "fold_curves",
    show: bool = True,
) -> list[plt.Figure]:
    """
    Тонкая обертка над :func:`plot_trial_fold_curves` для списка trial-ов.
    """

    figs: list[plt.Figure] = []
    for n in trial_numbers:
        figs.extend(
            plot_trial_fold_curves(
                study,
                n,
                fold_curves_attr=fold_curves_attr,
                show=show,
            )
        )
    return figs

