from __future__ import annotations

from pathlib import Path

import optuna


def load_study_sqlite(*, db_path: Path, study_name: str) -> optuna.Study:
    """
    Подгружает Optuna study из SQLite-хранилища.

    Параметры
    ----------
    db_path:
        Путь к SQLite-базе (f.e. ``tfr_s11.db``).
    study_name:
        Имя study внутри этой базы.
    """

    storage = f"sqlite:///{db_path}"
    return optuna.load_study(study_name=study_name, storage=storage)

