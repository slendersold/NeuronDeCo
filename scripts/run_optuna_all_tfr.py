#!/usr/bin/env python3

# Скрипт прогона пациентов в бинарной классификации
# пример использования
# python run_optuna_all_tfr.py \
#   --tfr-root /trinity/home/asma.benachour/notebooks/Pirogov/MNE_playground/car \
#   --out-dir  /trinity/home/asma.benachour/notebooks/Pirogov/MNE_playground/5-30-no_time_crop_on_train \
#   --pattern "tfr_*.fif" \
#   --min-f 5 --max-f 30 \
#   --min-t 100 --max-t -400 \
#   --crop-tmin 0.0 --crop-tmax 1.0 \
#   --event-pos-code 9 \
#   --n-trials 100 \
#   --max-epochs 100 \
#   --cv \
#   --min-free-gb 20

# прогнать пациентов на разных окнах частот [:, :, (0-5-10):(10-20-30), 100:-400]
import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
import shutil

import numpy as np
import mne
import optuna

# === Настройка путей проекта ===
project_root = Path("/trinity/home/asma.benachour/notebooks/Pirogov/MNE_playground")
sys.path.append(str(project_root))
print("Added to PYTHONPATH:", project_root)

# --- твои импорты проекта ---
# предполагается, что PYTHONPATH уже настроен как в твоих скриптах
from utils.normalisation import normalize_tfr_robust
from utils.AlexNet import AlexNetTFR
from utils.optuna_objective_makers import make_multi_objective
from utils.optuna_constraints import slope_constraint


# ---------------------------
# Utils: logging / errors
# ---------------------------
def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def die(msg: str, code: int = 2):
    log("FATAL: " + msg)
    sys.exit(code)


def fmt_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


# ---------------------------
# Disk / path checks
# ---------------------------
def ensure_dir_writable(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    testfile = path / ".write_test.tmp"
    try:
        with open(testfile, "wb") as f:
            f.write(b"ok")
        testfile.unlink(missing_ok=True)
    except Exception as e:
        raise PermissionError(f"Output dir not writable: {path} ({fmt_exc(e)})")


def bytes_to_gb(n: int) -> float:
    return n / (1024**3)


def check_free_space(path: Path, min_free_gb: float):
    usage = shutil.disk_usage(str(path))
    free_gb = bytes_to_gb(usage.free)
    if free_gb < min_free_gb:
        raise OSError(
            f"Low disk space at {path}: free={free_gb:.2f}GB < required={min_free_gb:.2f}GB"
        )


def preflight_storage_sqlite(out_dir: Path):
    """
    Проверяем заранее: можем ли создать SQLite-файл в out_dir и записать туда study.
    """
    test_db = out_dir / "_optuna_storage_test.db"
    storage_url = f"sqlite:///{test_db}"
    try:
        # создадим/удалим маленькую study
        study = optuna.create_study(
            directions=["maximize", "minimize"],
            storage=storage_url,
            study_name="__test__",
            load_if_exists=False,
        )
        def obj(trial):
            x = trial.suggest_float("x", 0, 1)
            return float(x), float(-x)
        study.optimize(obj, n_trials=2)
    finally:
        # чистим тестовый db
        try:
            if test_db.exists():
                test_db.unlink()
        except Exception:
            pass


# ---------------------------
# TFR loading / dataset
# ---------------------------
def load_xy_from_tfr(
    tfr_path: Path,
    *,
    event_pos_code: int,
    crop_tmin: float,
    crop_tmax: float,
    min_f: int,
    max_f: int,
    min_t: int,
    max_t: int,
):
    """
    Читает TFR, делает y (binary), достает X по общим границам.
    ВАЖНО: min/max_f и min/max_t — индексы по freq/time bins после crop.
    """
    tfr_list = mne.time_frequency.read_tfrs(str(tfr_path))
    if not tfr_list:
        raise ValueError("read_tfrs returned empty list")
    tfr = tfr_list[0]

    # y: 1 если код == event_pos_code иначе 0
    if getattr(tfr, "events", None) is None:
        raise ValueError("TFR has no events attribute")
    y = np.where(tfr.events[:, 2] == event_pos_code, 1, 0)

    # crop по времени (в секундах)
    tfr = tfr.crop(tmin=crop_tmin, tmax=crop_tmax)

    # X: (epochs, channels, freqs, times)
    X_full = tfr.data
    if X_full.ndim != 4:
        raise ValueError(f"Unexpected tfr.data shape: {X_full.shape}")

    # защита от выхода за границы
    n_f = X_full.shape[2]
    n_t = X_full.shape[3]
    if not (0 <= min_f < n_f and 0 < max_f <= n_f and min_f < max_f):
        raise IndexError(f"Bad freq slice [{min_f}:{max_f}] for n_f={n_f}")
    # max_t может быть отрицательным как “-400” — но ты хочешь ОДИНАКОВОЕ для всех,
    # поэтому приводим к python-срезу корректно:
    # если max_t < 0 -> python slicing ok, но проверим, что |max_t| <= n_t
    if max_t < 0:
        if abs(max_t) > n_t:
            raise IndexError(f"Bad time slice [{min_t}:{max_t}] for n_t={n_t} (abs(max_t) too large)")
    else:
        if not (0 < max_t <= n_t):
            raise IndexError(f"Bad time slice [{min_t}:{max_t}] for n_t={n_t}")
    if not (0 <= min_t < n_t):
        raise IndexError(f"Bad time slice start min_t={min_t} for n_t={n_t}")

    X = X_full[:, :, min_f:max_f, min_t:max_t]
    X = normalize_tfr_robust(X)

    # освободим tfr
    del tfr, tfr_list, X_full
    gc.collect()

    return X, y


# ---------------------------
# Study save helpers
# ---------------------------
def save_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def study_summary(study: optuna.Study) -> dict:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return {
        "study_name": study.study_name,
        "n_trials_total": len(study.trials),
        "n_trials_complete": len(complete),
        "directions": [d.name for d in study.directions],
        "best_trials_count": len(study.best_trials),
        "best_trials_preview": [
            {"number": t.number, "values": t.values, "params": t.params}
            for t in study.best_trials[:10]
        ],
    }


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfr-root", required=True, help="Folder with tfr_*.fif files")
    ap.add_argument("--out-dir", required=True, help="Where to save optuna studies and summaries")
    ap.add_argument("--pattern", default="tfr_*.fif", help="Glob pattern inside tfr-root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--cv", action="store_true", help="Use StratifiedKFold inside objective")
    ap.add_argument("--cv-aggregate", default="median", choices=["mean", "median"])
    ap.add_argument("--event-pos-code", type=int, default=9, help="events[:,2] == this => y=1")
    ap.add_argument("--crop-tmin", type=float, default=0.0)
    ap.add_argument("--crop-tmax", type=float, default=1.0)

    # ОДНА настройка для всех файлов (как ты просишь)
    ap.add_argument("--min-f", type=int, required=True)
    ap.add_argument("--max-f", type=int, required=True)
    ap.add_argument("--min-t", type=int, required=True)
    ap.add_argument("--max-t", type=int, required=True)

    ap.add_argument("--min-free-gb", type=float, default=10.0, help="Fail early if free space below this")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = ap.parse_args()

    tfr_root = Path(args.tfr_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"tfr_root={tfr_root}")
    log(f"out_dir={out_dir}")

    # ---- PRE-FLIGHT (раньше всего) ----
    if not tfr_root.exists():
        die(f"--tfr-root does not exist: {tfr_root}")
    if not tfr_root.is_dir():
        die(f"--tfr-root is not a directory: {tfr_root}")

    try:
        ensure_dir_writable(out_dir)
    except Exception as e:
        die(str(e))

    try:
        check_free_space(out_dir, args.min_free_gb)
    except Exception as e:
        die(f"Disk space check failed: {e}")

    # Проверим, что Optuna сможет писать SQLite прямо сейчас
    try:
        preflight_storage_sqlite(out_dir)
        log("Preflight: Optuna SQLite storage write OK")
    except Exception as e:
        die(f"Preflight: cannot create Optuna SQLite storage in {out_dir}: {fmt_exc(e)}")

    # Найдём файлы
    tfr_files = sorted(tfr_root.glob(args.pattern))
    if not tfr_files:
        die(f"No files matched pattern {args.pattern} in {tfr_root}")

    log(f"Found {len(tfr_files)} files")

    # Устройство
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cuda"
    else:
        device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") and os.environ.get("CUDA_VISIBLE_DEVICES") != "") else "cpu"
    log(f"Using device={device}")

    # Global error report
    errors = {}
    saved = {}

    # Чтобы заранее поймать “плохой срез” на первом файле (раньше многочасового прогона)
    # читаем один файл и проверяем slicing
    try:
        log("Preflight: reading first TFR to validate slicing...")
        _ = load_xy_from_tfr(
            tfr_files[0],
            event_pos_code=args.event_pos_code,
            crop_tmin=args.crop_tmin,
            crop_tmax=args.crop_tmax,
            min_f=args.min_f, max_f=args.max_f,
            min_t=args.min_t, max_t=args.max_t,
        )
        log("Preflight: slicing OK on first file")
    except Exception as e:
        die(f"Preflight failed on first file {tfr_files[0].name}: {fmt_exc(e)}")

    # ---- MAIN LOOP ----
    for tfr_path in tfr_files:
        subj = tfr_path.stem.replace("tfr_", "")  # tfr_s11 -> s11 (если другое имя — останется stem)
        log(f"=== SUBJECT {subj} | FILE {tfr_path.name} ===")

        # чек места перед каждым пациентом (чтобы ловить “вдруг закончилось” хотя бы заранее)
        try:
            check_free_space(out_dir, args.min_free_gb)
        except Exception as e:
            errors[subj] = f"DiskSpaceError: {e}"
            log(f"ERROR [{subj}] disk space: {e}")
            continue

        # пути сохранения
        study_db = out_dir / f"{tfr_path.stem}.db"          # например tfr_s11.db
        summary_json = out_dir / f"{tfr_path.stem}.json"     # summary
        err_txt = out_dir / f"{tfr_path.stem}.error.txt"     # stacktrace if needed

        # создадим storage URL
        storage_url = f"sqlite:///{study_db}"

        try:
            # 1) READ + BUILD DATASET
            X, y = load_xy_from_tfr(
                tfr_path,
                event_pos_code=args.event_pos_code,
                crop_tmin=args.crop_tmin,
                crop_tmax=args.crop_tmax,
                min_f=args.min_f, max_f=args.max_f,
                min_t=args.min_t, max_t=args.max_t,
            )
            channels = int(X.shape[1])

            # 2) CREATE STUDY
            sampler = optuna.samplers.NSGAIISampler(
                seed=args.seed,
                constraints_func=slope_constraint,  # твои “родные” ограничения
            )
            study = optuna.create_study(
                directions=["maximize", "minimize"],
                sampler=sampler,
                storage=storage_url,
                study_name=tfr_path.stem,
                load_if_exists=True,  # чтобы можно было перезапускать
            )

            # 3) OBJECTIVE
            objective = make_multi_objective(
                X,
                y,
                test_size=args.test_size,
                seed=args.seed,
                device=device,
                ModelCls=AlexNetTFR,
                in_channels=channels,
                num_classes=2,
                max_epochs=args.max_epochs,
                patience=args.patience,
                cv=args.cv,
                cv_aggregate=args.cv_aggregate,
            )

            # 4) OPTIMIZE
            study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

            # 5) SAVE SUMMARY (атомарно)
            summ = study_summary(study)
            summ.update({
                "file": str(tfr_path),
                "out_db": str(study_db),
                "min_f": args.min_f, "max_f": args.max_f,
                "min_t": args.min_t, "max_t": args.max_t,
                "crop_tmin": args.crop_tmin, "crop_tmax": args.crop_tmax,
                "event_pos_code": args.event_pos_code,
            })
            save_json(summary_json, summ)

            saved[subj] = {
                "study_db": str(study_db),
                "summary": str(summary_json),
                "n_trials": len(study.trials),
            }

            log(f"OK [{subj}] saved: {study_db.name} + {summary_json.name}")

            # освобождение памяти
            del X, y, study, objective
            gc.collect()

        except (OSError, IOError) as e:
            # критичные ошибки чтения/записи/места — логируем отдельно
            errors[subj] = f"I/O Error: {fmt_exc(e)}"
            log(f"ERROR [{subj}] I/O: {fmt_exc(e)}")

            # stacktrace в файл
            try:
                with open(err_txt, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
            except Exception:
                pass

        except Exception as e:
            errors[subj] = f"Runtime Error: {fmt_exc(e)}"
            log(f"ERROR [{subj}] runtime: {fmt_exc(e)}")

            try:
                with open(err_txt, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
            except Exception:
                pass

    # ---- FINAL REPORT ----
    log("=== DONE ===")
    log(f"Saved studies: {len(saved)}")
    log(f"Errors: {len(errors)}")

    # общий отчёт
    report = {
        "tfr_root": str(tfr_root),
        "out_dir": str(out_dir),
        "pattern": args.pattern,
        "min_f": args.min_f, "max_f": args.max_f,
        "min_t": args.min_t, "max_t": args.max_t,
        "crop_tmin": args.crop_tmin, "crop_tmax": args.crop_tmax,
        "event_pos_code": args.event_pos_code,
        "n_trials": args.n_trials,
        "saved": saved,
        "errors": errors,
    }
    save_json(out_dir / "_RUN_REPORT.json", report)

    # печать коротко в консоль
    print("Saved:", saved)
    print("Errors:", errors)


if __name__ == "__main__":
    main()
