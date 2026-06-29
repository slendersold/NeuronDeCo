#!/usr/bin/env python3
"""
Static Optuna parallel-coordinate plots grouped by:

    output_root / study_type / patient / model /

Main visual rules
-----------------
1. The selected objective (default: objective 0 = F1) defines quality.
2. Every non-best trial follows one and the same critically damped second-order
   aperiodic decay. There is no separate visual class for top-N trials.
3. --top-n is only a calibration anchor: the decay rate is selected so the
   N-th best trial reaches TOP_N_BOUNDARY_WEIGHT.
4. Better objective -> more opaque saturated blue. Worse objective -> blue
   fades continuously toward the background.
5. The best trial is not drawn as a blue line first; it is drawn once as a red
   dashed line, so the highlight cannot merge with an underlying blue trace.
6. Optimization direction is read from Optuna study.directions by default.

Compatible with Python 3.10+ and Optuna 4.x.

python optuna_parallel_coordinates_aperiodic.py \
    --storage /path/to/optuna.db \
    --output-root /path/to/figures \
    --top-n 10 \
    --objective-index 0 \
    --objective-label F1 \
    --direction auto
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from matplotlib.lines import Line2D
from optuna.trial import FrozenTrial, TrialState


# =============================================================================
# USER SETTINGS
# =============================================================================

STUDY_TYPE_ATTR = "study_type"
PATIENT_ATTR = "patient"
MODEL_ATTR = "model"

# Fallback study-name patterns, for example:
#   time_frequency__s11__alexnet
#   time-frequency-s11-transformer
#   ablation_s11_svm
#   experiment/s11/alexnet
STUDY_NAME_RE = re.compile(
    r"^(?P<study_type>.+?)(?:__|[-_/])"
    r"(?P<patient>s\d{1,3})(?:__|[-_/])"
    r"(?P<model>.+)$",
    flags=re.IGNORECASE,
)

# Optional model-specific axis order. Parameters absent here are appended.
PARAM_ORDER_BY_MODEL: dict[str, list[str]] = {
    # "svm": ["C", "gamma", "kernel"],
    # "alexnet": ["learning_rate", "batch_size", "dropout"],
    # "transformer": ["learning_rate", "n_heads", "n_layers", "dropout"],
}

EXCLUDE_PARAMS: set[str] = set()

DEFAULT_OBJECTIVE_INDEX = 0
DEFAULT_OBJECTIVE_LABEL = "F1"
DEFAULT_TOP_N = 10
DEFAULT_DIRECTION: Literal["auto", "maximize", "minimize"] = "auto"

# Pure critically damped second-order decay:
#     w(d) = (1 + a*d) * exp(-a*d)
# where d=0 for the best objective and d=1 for the worst objective.
#
# --top-n calibrates a. The N-th best trial is assigned this response weight.
# No discontinuous top-N styling is applied.
TOP_N_BOUNDARY_WEIGHT = 0.35

TRIAL_BLUE = "#08306B"
BEST_COLOR = "#D62728"
BEST_LINEWIDTH = 2.9
BEST_DASH_PATTERN = (0, (5, 2.4))

# All ordinary trials share one continuous mapping from response weight to
# opacity and line width.
MIN_VISIBLE_ALPHA = 0.018
MAX_ALPHA = 0.98
MIN_LINEWIDTH = 0.30
MAX_LINEWIDTH = 2.15

FIGURE_FACE_COLOR = "#EEF2F6"
AXIS_COLOR = "#9AA6B3"
TEXT_COLOR = "#203040"
OUTPUT_FORMATS = ("png", "svg")
DPI = 190
FIGURE_HEIGHT = 8.8
WIDTH_PER_AXIS = 2.05
MIN_FIGURE_WIDTH = 12.5
MAX_CATEGORICAL_TICKS = 14
NUMERIC_TICK_COUNT = 5
MISSING_LABEL = "<N/A>"


# =============================================================================
# DATA TYPES
# =============================================================================

Direction = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class StudyIdentity:
    study_type: str
    patient: str
    model: str


@dataclass
class TrialRecord:
    study_name: str
    identity: StudyIdentity
    direction: Direction
    trial_number: int
    objective: float
    params: dict[str, Any]
    distributions: dict[str, Any]
    datetime_start: datetime | None
    datetime_complete: datetime | None
    duration_seconds: float | None
    sequence_id: int


@dataclass
class AxisSpec:
    name: str
    label: str
    values: np.ndarray
    tick_positions: np.ndarray
    tick_labels: list[str]


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def sanitize_component(value: str) -> str:
    value = re.sub(r"[^\w.\-]+", "_", str(value).strip(), flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "unknown"


def normalized_storage_url(storage: str) -> str:
    if "://" in storage:
        return storage
    db_path = Path(storage).expanduser().resolve()
    return f"sqlite:///{db_path.as_posix()}"


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def objective_from_trial(trial: FrozenTrial, objective_index: int) -> float | None:
    if trial.values is not None:
        if objective_index >= len(trial.values):
            return None
        return safe_float(trial.values[objective_index])

    if objective_index != 0:
        return None

    return safe_float(trial.value)


def parse_study_identity(study: optuna.study.Study) -> StudyIdentity | None:
    attrs = study.user_attrs
    study_type = attrs.get(STUDY_TYPE_ATTR)
    patient = attrs.get(PATIENT_ATTR)
    model = attrs.get(MODEL_ATTR)

    if all(
        value is not None and str(value).strip()
        for value in (study_type, patient, model)
    ):
        return StudyIdentity(
            study_type=str(study_type).strip(),
            patient=str(patient).strip(),
            model=str(model).strip(),
        )

    match = STUDY_NAME_RE.match(study.study_name.strip())
    if match is None:
        return None

    return StudyIdentity(
        study_type=match.group("study_type").strip(" _-/"),
        patient=match.group("patient").strip(),
        model=match.group("model").strip(" _-/"),
    )


def read_direction(
    study: optuna.study.Study,
    objective_index: int,
    direction_override: Literal["auto", "maximize", "minimize"],
) -> Direction:
    if direction_override != "auto":
        return direction_override

    if objective_index >= len(study.directions):
        raise IndexError(
            f"Study {study.study_name!r} has {len(study.directions)} objectives, "
            f"but objective index {objective_index} was requested."
        )

    name = study.directions[objective_index].name.lower()
    if name not in {"maximize", "minimize"}:
        raise ValueError(
            f"Unsupported Optuna direction {name!r} in {study.study_name!r}."
        )
    return name  # type: ignore[return-value]


def chronological_key(record: TrialRecord) -> tuple[datetime, int]:
    dt = record.datetime_complete or record.datetime_start
    if dt is None:
        dt = datetime.min.replace(tzinfo=timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt, record.sequence_id


def is_numeric_scalar(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def format_number(value: float) -> str:
    absolute = abs(value)
    if value == 0:
        return "0"
    if absolute >= 1e4 or absolute < 1e-3:
        return f"{value:.2e}"
    if absolute >= 100:
        return f"{value:.0f}"
    if absolute >= 10:
        return f"{value:.2f}"
    return f"{value:.4g}"


# =============================================================================
# OPTUNA DATABASE READING
# =============================================================================


def load_all_trial_groups(
    storage_url: str,
    objective_index: int,
    direction_override: Literal["auto", "maximize", "minimize"],
) -> dict[StudyIdentity, list[TrialRecord]]:
    groups: dict[StudyIdentity, list[TrialRecord]] = defaultdict(list)

    summaries = optuna.study.get_all_study_summaries(
        storage=storage_url,
        include_best_trial=False,
    )

    sequence_id = 0
    for summary in summaries:
        study = optuna.load_study(
            study_name=summary.study_name,
            storage=storage_url,
        )
        identity = parse_study_identity(study)

        if identity is None:
            print(
                f"[SKIP] Cannot parse identity from study {study.study_name!r}. "
                "Set study_type/patient/model user_attrs or edit STUDY_NAME_RE.",
                file=sys.stderr,
            )
            continue

        direction = read_direction(study, objective_index, direction_override)
        trials = study.get_trials(
            deepcopy=False,
            states=(TrialState.COMPLETE,),
        )

        for trial in trials:
            objective = objective_from_trial(trial, objective_index)
            if objective is None:
                continue

            groups[identity].append(
                TrialRecord(
                    study_name=study.study_name,
                    identity=identity,
                    direction=direction,
                    trial_number=trial.number,
                    objective=objective,
                    params=dict(trial.params),
                    distributions=dict(trial.distributions),
                    datetime_start=trial.datetime_start,
                    datetime_complete=trial.datetime_complete,
                    duration_seconds=(
                        trial.duration.total_seconds()
                        if trial.duration is not None
                        else None
                    ),
                    sequence_id=sequence_id,
                )
            )
            sequence_id += 1

    # Validate that studies merged into one patient/model group optimize in the
    # same direction.
    for identity, records in groups.items():
        directions = {record.direction for record in records}
        if len(directions) > 1:
            raise ValueError(
                f"Direction mismatch inside group {identity}: {sorted(directions)}"
            )
        records.sort(key=chronological_key)

    return dict(groups)


# =============================================================================
# RANKING AND EMPHASIS
# =============================================================================


def quality_order(objectives: np.ndarray, direction: Direction) -> np.ndarray:
    """
    Return indices from worst to best.

    Drawing in this order guarantees that better trials are drawn later.
    """
    if direction == "maximize":
        return np.argsort(objectives, kind="stable")
    return np.argsort(-objectives, kind="stable")


def best_first_order(objectives: np.ndarray, direction: Direction) -> np.ndarray:
    if direction == "maximize":
        return np.argsort(-objectives, kind="stable")
    return np.argsort(objectives, kind="stable")


def metric_ranks(
    objectives: np.ndarray,
    direction: Direction,
) -> np.ndarray:
    """One-based ranks: rank 1 is the best objective value."""
    order = best_first_order(objectives, direction)
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def objective_quality(
    objectives: np.ndarray,
    direction: Direction,
) -> np.ndarray:
    """
    Normalize objective values to [0, 1].

    0 = worst observed objective
    1 = best observed objective
    """
    minimum = float(np.min(objectives))
    maximum = float(np.max(objectives))

    if math.isclose(minimum, maximum):
        return np.ones_like(objectives, dtype=float)

    if direction == "maximize":
        return (objectives - minimum) / (maximum - minimum)
    return (maximum - objectives) / (maximum - minimum)


def second_order_decay(
    distance_from_best: float | np.ndarray,
    rate: float,
) -> float | np.ndarray:
    """
    Critically damped second-order aperiodic decay.

        w(d) = (1 + a*d) * exp(-a*d)

    d = 0 at the best objective, so w(0)=1.
    The response decreases monotonically toward zero as objective quality
    moves away from the best trial.
    """
    d = np.clip(np.asarray(distance_from_best, dtype=float), 0.0, 1.0)
    z = rate * d
    weight = (1.0 + z) * np.exp(-z)
    return float(weight) if np.ndim(distance_from_best) == 0 else weight


def solve_dimensionless_decay_target(target_weight: float) -> float:
    """Solve (1+z)*exp(-z)=target_weight for z>=0 by bisection."""
    if not 0.0 < target_weight < 1.0:
        raise ValueError("target_weight must be strictly between 0 and 1")

    low = 0.0
    high = 1.0
    while (1.0 + high) * math.exp(-high) > target_weight:
        high *= 2.0

    for _ in range(100):
        mid = 0.5 * (low + high)
        value = (1.0 + mid) * math.exp(-mid)
        if value > target_weight:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def calibrate_decay_rate(
    quality: np.ndarray,
    best_to_worst: np.ndarray,
    top_n: int,
    boundary_weight: float = TOP_N_BOUNDARY_WEIGHT,
) -> tuple[float, int, float, float]:
    """
    Select the second-order rate from --top-n.

    The N-th best objective is used as the calibration anchor. The rate a is
    chosen so its response equals boundary_weight. All trials still use the
    same continuous response; top-N is not a separate visual category.

    Returns:
        rate,
        effective_top_n,
        anchor_quality,
        anchor_distance_from_best
    """
    n_trials = len(quality)
    if n_trials == 0:
        raise ValueError("Cannot calibrate an empty trial set")

    effective_top_n = min(max(int(top_n), 1), n_trials)
    anchor_index = int(best_to_worst[effective_top_n - 1])
    anchor_quality = float(quality[anchor_index])
    anchor_distance = max(1.0 - anchor_quality, 0.0)

    # Equal or near-equal objective values can make the metric distance zero.
    # Fall back to the corresponding normalized rank distance only for the
    # calibration denominator; the actual line weights remain metric-based.
    if anchor_distance < 1e-12:
        if n_trials == 1:
            anchor_distance = 1.0
        else:
            anchor_distance = (effective_top_n - 1) / (n_trials - 1)
            if anchor_distance < 1e-12:
                anchor_distance = 1.0 / (n_trials - 1)

    z_target = solve_dimensionless_decay_target(boundary_weight)
    rate = z_target / anchor_distance
    return rate, effective_top_n, anchor_quality, anchor_distance


def continuous_line_style(weight: float) -> tuple[float, float]:
    """Map one pure response weight continuously to alpha and line width."""
    w = float(np.clip(weight, 0.0, 1.0))
    alpha = MIN_VISIBLE_ALPHA + (MAX_ALPHA - MIN_VISIBLE_ALPHA) * w
    linewidth = MIN_LINEWIDTH + (MAX_LINEWIDTH - MIN_LINEWIDTH) * w
    return alpha, linewidth


# =============================================================================
# AXIS ENCODING
# =============================================================================


def find_log_distribution(
    records: Sequence[TrialRecord],
    param_name: str,
) -> bool:
    flags: list[bool] = []
    for record in records:
        distribution = record.distributions.get(param_name)
        if distribution is not None and hasattr(distribution, "log"):
            flags.append(bool(getattr(distribution, "log")))
    return bool(flags) and all(flags)


def numeric_axis(
    name: str,
    label: str,
    raw_values: Sequence[Any],
    *,
    is_log: bool,
    allow_missing: bool,
) -> AxisSpec:
    numeric_values = np.array(
        [
            float(value) if is_numeric_scalar(value) else np.nan
            for value in raw_values
        ],
        dtype=float,
    )
    finite_mask = np.isfinite(numeric_values)
    if not finite_mask.any():
        raise ValueError(f"No finite values for numeric axis {name!r}")

    actual = numeric_values[finite_mask]
    if is_log and np.any(actual <= 0):
        is_log = False

    transformed_actual = np.log10(actual) if is_log else actual
    minimum = float(np.min(transformed_actual))
    maximum = float(np.max(transformed_actual))

    lower = 0.14 if allow_missing and not finite_mask.all() else 0.0
    upper = 1.0
    encoded = np.full(len(raw_values), 0.0 if lower > 0 else np.nan)

    if math.isclose(minimum, maximum):
        encoded[finite_mask] = (lower + upper) / 2.0
        ticks = np.array([minimum])
        tick_positions = np.array([(lower + upper) / 2.0])
    else:
        transformed = (
            np.log10(numeric_values[finite_mask])
            if is_log
            else numeric_values[finite_mask]
        )
        encoded[finite_mask] = lower + (
            (transformed - minimum) / (maximum - minimum)
        ) * (upper - lower)
        ticks = np.linspace(minimum, maximum, NUMERIC_TICK_COUNT)
        tick_positions = lower + (
            (ticks - minimum) / (maximum - minimum)
        ) * (upper - lower)

    display_ticks = np.power(10.0, ticks) if is_log else ticks
    tick_labels = [format_number(float(value)) for value in display_ticks]

    if lower > 0:
        tick_positions = np.concatenate(([0.0], tick_positions))
        tick_labels = [MISSING_LABEL] + tick_labels

    return AxisSpec(
        name=name,
        label=label + (" [log]" if is_log else ""),
        values=encoded,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
    )


def categorical_axis(
    name: str,
    label: str,
    raw_values: Sequence[Any],
) -> AxisSpec:
    display_values = [
        MISSING_LABEL if value is None else str(value)
        for value in raw_values
    ]
    categories = sorted(
        set(display_values),
        key=lambda value: (value != MISSING_LABEL, value.lower()),
    )

    if len(categories) == 1:
        mapping = {categories[0]: 0.5}
    else:
        mapping = {
            category: index / (len(categories) - 1)
            for index, category in enumerate(categories)
        }

    encoded = np.array(
        [mapping[value] for value in display_values],
        dtype=float,
    )

    if len(categories) <= MAX_CATEGORICAL_TICKS:
        shown = categories
    else:
        indices = np.unique(
            np.linspace(
                0,
                len(categories) - 1,
                MAX_CATEGORICAL_TICKS,
            ).round().astype(int)
        )
        shown = [categories[index] for index in indices]

    return AxisSpec(
        name=name,
        label=label,
        values=encoded,
        tick_positions=np.array(
            [mapping[value] for value in shown],
            dtype=float,
        ),
        tick_labels=shown,
    )


def parameter_order(
    records: Sequence[TrialRecord],
    model_name: str,
) -> list[str]:
    prevalence: dict[str, int] = defaultdict(int)
    for record in records:
        for param_name in record.params:
            if param_name not in EXCLUDE_PARAMS:
                prevalence[param_name] += 1

    explicit = [
        name
        for name in PARAM_ORDER_BY_MODEL.get(model_name.lower(), [])
        if name in prevalence
    ]
    remaining = sorted(
        (name for name in prevalence if name not in explicit),
        key=lambda name: (-prevalence[name], name.lower()),
    )
    return explicit + remaining


def build_axes(
    records: Sequence[TrialRecord],
    objective_label: str,
) -> list[AxisSpec]:
    axes = [
        numeric_axis(
            "__objective__",
            objective_label,
            [record.objective for record in records],
            is_log=False,
            allow_missing=False,
        )
    ]

    for param_name in parameter_order(records, records[0].identity.model):
        values = [record.params.get(param_name) for record in records]
        nonmissing = [value for value in values if value is not None]
        numeric = bool(nonmissing) and all(
            is_numeric_scalar(value) for value in nonmissing
        )

        if numeric:
            axes.append(
                numeric_axis(
                    param_name,
                    param_name,
                    values,
                    is_log=find_log_distribution(records, param_name),
                    allow_missing=True,
                )
            )
        else:
            axes.append(
                categorical_axis(param_name, param_name, values)
            )

    return axes


# =============================================================================
# CSV / JSON EXPORT
# =============================================================================


def records_to_dataframe(
    records: Sequence[TrialRecord],
    top_n: int,
) -> pd.DataFrame:
    objectives = np.array([record.objective for record in records], dtype=float)
    direction = records[0].direction
    ranks = metric_ranks(objectives, direction)
    quality = objective_quality(objectives, direction)
    best_to_worst = best_first_order(objectives, direction)
    rate, effective_top_n, anchor_quality, anchor_distance = calibrate_decay_rate(
        quality,
        best_to_worst,
        top_n,
    )
    weights = np.asarray(second_order_decay(1.0 - quality, rate), dtype=float)

    param_names = sorted(
        {name for record in records for name in record.params}
    )
    rows: list[dict[str, Any]] = []

    for chronological_order, (record, rank, q, weight) in enumerate(
        zip(records, ranks, quality, weights)
    ):
        alpha, linewidth = continuous_line_style(float(weight))
        row: dict[str, Any] = {
            "chronological_order": chronological_order,
            "study_name": record.study_name,
            "study_type": record.identity.study_type,
            "patient": record.identity.patient,
            "model": record.identity.model,
            "direction": direction,
            "trial_number": record.trial_number,
            "objective": record.objective,
            "metric_rank": int(rank),
            "objective_quality_0_1": float(q),
            "aperiodic_weight": float(weight),
            "line_alpha": alpha,
            "line_width": linewidth,
            "calibration_top_n": effective_top_n,
            "calibration_anchor_quality": anchor_quality,
            "calibration_anchor_distance": anchor_distance,
            "calibrated_second_order_rate": rate,
            "datetime_start": record.datetime_start,
            "datetime_complete": record.datetime_complete,
            "duration_seconds": record.duration_seconds,
        }
        for param_name in param_names:
            row[f"param__{param_name}"] = record.params.get(param_name)
        rows.append(row)

    return pd.DataFrame(rows)


def write_summary_json(
    records: Sequence[TrialRecord],
    output_path: Path,
    *,
    top_n: int,
    objective_index: int,
    objective_label: str,
) -> None:
    objectives = np.array([record.objective for record in records], dtype=float)
    direction = records[0].direction
    quality = objective_quality(objectives, direction)
    order = best_first_order(objectives, direction)
    best = records[int(order[0])]
    rate, effective_top_n, anchor_quality, anchor_distance = calibrate_decay_rate(
        quality,
        order,
        top_n,
    )

    payload = {
        "study_type": best.identity.study_type,
        "patient": best.identity.patient,
        "model": best.identity.model,
        "n_complete_trials": len(records),
        "objective_index": objective_index,
        "objective_label": objective_label,
        "direction": direction,
        "best_objective": best.objective,
        "best_study_name": best.study_name,
        "best_trial_number": best.trial_number,
        "best_params": best.params,
        "visual_law": "critically_damped_second_order_decay",
        "visual_law_formula": "w(d)=(1+a*d)*exp(-a*d)",
        "calibration_top_n": effective_top_n,
        "boundary_weight_at_nth_best": TOP_N_BOUNDARY_WEIGHT,
        "calibration_anchor_quality": anchor_quality,
        "calibration_anchor_distance_from_best": anchor_distance,
        "calibrated_second_order_rate": rate,
        "best_line_style": "red_dashed_only_not_underlaid_by_blue",
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# =============================================================================
# PLOTTING
# =============================================================================


def draw_axis(ax: plt.Axes, x: float, spec: AxisSpec) -> None:
    ax.plot(
        [x, x],
        [0, 1],
        color=AXIS_COLOR,
        linewidth=0.9,
        zorder=0,
    )

    for position, label in zip(spec.tick_positions, spec.tick_labels):
        ax.plot(
            [x - 0.025, x + 0.025],
            [position, position],
            color=AXIS_COLOR,
            linewidth=0.7,
            zorder=0,
        )
        ax.text(
            x - 0.04,
            position,
            label,
            ha="right",
            va="center",
            fontsize=7.5,
            color=TEXT_COLOR,
        )

    ax.text(
        x,
        1.075,
        spec.label,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="semibold",
        color=TEXT_COLOR,
    )


def metric_colormap(
    direction: Direction,
    rate: float,
) -> mpl.colors.Colormap:
    """
    Build a colorbar that matches the actual nonlinear blue-opacity response
    on the figure background.
    """
    quality = np.linspace(0.0, 1.0, 256)
    weights = np.asarray(second_order_decay(1.0 - quality, rate), dtype=float)

    background_rgb = np.array(mpl.colors.to_rgb(FIGURE_FACE_COLOR))
    blue_rgb = np.array(mpl.colors.to_rgb(TRIAL_BLUE))
    effective_alpha = MIN_VISIBLE_ALPHA + (MAX_ALPHA - MIN_VISIBLE_ALPHA) * weights
    rgb = (
        effective_alpha[:, None] * blue_rgb[None, :]
        + (1.0 - effective_alpha[:, None]) * background_rgb[None, :]
    )
    rgba = np.column_stack([rgb, np.ones(len(rgb))])
    cmap = mpl.colors.ListedColormap(rgba, name="aperiodic_blue_quality")

    if direction == "maximize":
        return cmap
    return cmap.reversed(name="aperiodic_blue_quality_minimize")


def plot_parallel_coordinates(
    records: Sequence[TrialRecord],
    output_dir: Path,
    *,
    top_n: int,
    objective_label: str,
) -> Path:
    if not records:
        raise ValueError("No complete trials to plot.")

    records = sorted(records, key=chronological_key)
    identity = records[0].identity
    direction = records[0].direction
    axis_specs = build_axes(records, objective_label)

    if len(axis_specs) < 2:
        raise ValueError("No hyperparameters found; only the objective is available.")

    coordinates = np.column_stack([spec.values for spec in axis_specs])
    n_trials, n_axes = coordinates.shape
    x_positions = np.arange(n_axes, dtype=float)
    objectives = np.array([record.objective for record in records], dtype=float)
    quality = objective_quality(objectives, direction)

    worst_to_best = quality_order(objectives, direction)
    best_to_worst = worst_to_best[::-1]
    best_index = int(best_to_worst[0])
    best = records[best_index]

    rate, effective_top_n, anchor_quality, anchor_distance = calibrate_decay_rate(
        quality,
        best_to_worst,
        top_n,
    )
    weights = np.asarray(second_order_decay(1.0 - quality, rate), dtype=float)

    objective_min = float(np.min(objectives))
    objective_max = float(np.max(objectives))
    if math.isclose(objective_min, objective_max):
        color_norm = mpl.colors.Normalize(
            objective_min - 0.5,
            objective_max + 0.5,
        )
    else:
        color_norm = mpl.colors.Normalize(objective_min, objective_max)

    cmap = metric_colormap(direction, rate)
    fig_width = max(MIN_FIGURE_WIDTH, WIDTH_PER_AXIS * n_axes)
    fig, ax = plt.subplots(
        figsize=(fig_width, FIGURE_HEIGHT),
        constrained_layout=True,
        facecolor=FIGURE_FACE_COLOR,
    )
    ax.set_facecolor(FIGURE_FACE_COLOR)

    for axis_index, spec in enumerate(axis_specs):
        draw_axis(ax, float(axis_index), spec)

    # One continuous law for every ordinary trial. The best trial is skipped
    # here and drawn once below as a red dashed line, preventing line overlap.
    for trial_index in worst_to_best:
        trial_index = int(trial_index)
        if trial_index == best_index:
            continue

        alpha, linewidth = continuous_line_style(float(weights[trial_index]))
        ax.plot(
            x_positions,
            coordinates[trial_index],
            color=TRIAL_BLUE,
            alpha=alpha,
            linewidth=linewidth,
            zorder=1 + 3 * float(weights[trial_index]),
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    # Best configuration: only one red dashed trace, with no blue trace below.
    best_line, = ax.plot(
        x_positions,
        coordinates[best_index],
        color=BEST_COLOR,
        alpha=1.0,
        linewidth=BEST_LINEWIDTH,
        linestyle=BEST_DASH_PATTERN,
        zorder=9,
        dash_capstyle="round",
        solid_joinstyle="round",
    )
    best_line.set_path_effects(
        [
            path_effects.Stroke(
                linewidth=BEST_LINEWIDTH + 1.2,
                foreground=(1.0, 1.0, 1.0, 0.92),
            ),
            path_effects.Normal(),
        ]
    )
    ax.scatter(
        [x_positions[0]],
        [coordinates[best_index, 0]],
        s=125,
        marker="*",
        facecolor=BEST_COLOR,
        edgecolor="white",
        linewidth=1.4,
        zorder=10,
    )

    ax.set_xlim(-0.72, n_axes - 0.35)
    ax.set_ylim(-0.08, 1.14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    quality_word = "higher" if direction == "maximize" else "lower"
    ax.set_title(
        f"{identity.study_type} | {identity.patient} | {identity.model}\n"
        f"{n_trials} COMPLETE trials; objective {objective_label}; "
        f"{quality_word} is better; best = {best.objective:.5f}\n"
        f"one second-order decay for all trials; --top-n={effective_top_n} "
        f"calibrates a={rate:.3g} at weight={TOP_N_BOUNDARY_WEIGHT:.2f}",
        fontsize=12.5,
        pad=23,
        color=TEXT_COLOR,
    )

    scalar_mappable = mpl.cm.ScalarMappable(norm=color_norm, cmap=cmap)
    scalar_mappable.set_array(objectives)
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        fraction=0.025,
        pad=0.018,
    )
    colorbar.outline.set_edgecolor(AXIS_COLOR)
    colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    colorbar.set_label(
        f"{objective_label}: faded blue = worse, saturated blue = better "
        f"({direction}; nonlinear second-order opacity)",
        color=TEXT_COLOR,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TRIAL_BLUE,
            alpha=0.18,
            linewidth=0.7,
            label="All ordinary trials: one continuous aperiodic law",
        ),
        Line2D(
            [0],
            [0],
            color=BEST_COLOR,
            alpha=1.0,
            linewidth=BEST_LINEWIDTH,
            linestyle=BEST_DASH_PATTERN,
            marker="*",
            markerfacecolor=BEST_COLOR,
            markeredgecolor="white",
            label=f"Best {objective_label}",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=2,
        frameon=False,
        labelcolor=TEXT_COLOR,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = (
        "parallel_coordinates__"
        f"{sanitize_component(identity.study_type)}__"
        f"{sanitize_component(identity.patient)}__"
        f"{sanitize_component(identity.model)}"
    )

    for extension in OUTPUT_FORMATS:
        fig.savefig(
            output_dir / f"{base_name}.{extension}",
            dpi=DPI,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    plt.close(fig)
    return output_dir / f"{base_name}.{OUTPUT_FORMATS[0]}"


def output_directory(
    output_root: Path,
    identity: StudyIdentity,
) -> Path:
    return (
        output_root
        / sanitize_component(identity.study_type)
        / sanitize_component(identity.patient)
        / sanitize_component(identity.model)
    )


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one Optuna parallel-coordinate plot per "
            "study_type/patient/model using one continuous second-order "
            "aperiodic opacity law calibrated by --top-n."
        )
    )
    parser.add_argument(
        "--storage",
        required=True,
        help=(
            "Optuna storage URL or SQLite path, e.g. "
            "sqlite:////data/optuna.db or /data/optuna.db"
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=("Calibration anchor: the N-th best trial is assigned "              "TOP_N_BOUNDARY_WEIGHT. No hard top-N highlighting."),
    )
    parser.add_argument(
        "--objective-index",
        type=int,
        default=DEFAULT_OBJECTIVE_INDEX,
        help="Objective index. 0 means the first objective/F1.",
    )
    parser.add_argument(
        "--objective-label",
        default=DEFAULT_OBJECTIVE_LABEL,
        help="Human-readable objective label used in figures.",
    )
    parser.add_argument(
        "--direction",
        choices=("auto", "maximize", "minimize"),
        default=DEFAULT_DIRECTION,
        help=(
            "Optimization direction. 'auto' reads study.directions from Optuna."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    top_n = max(args.top_n, 1)
    objective_index = max(args.objective_index, 0)
    storage_url = normalized_storage_url(args.storage)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Optuna {optuna.__version__}")
    print(f"Storage: {storage_url}")
    print(f"Output: {output_root}")
    print(f"Objective: index={objective_index}, label={args.objective_label}")
    print(f"Direction: {args.direction}")
    print(f"Second-order calibration top N: {top_n}")

    groups = load_all_trial_groups(
        storage_url=storage_url,
        objective_index=objective_index,
        direction_override=args.direction,
    )

    if not groups:
        print(
            "No plottable groups found. Check study names/user_attrs and "
            "COMPLETE trials.",
            file=sys.stderr,
        )
        return 1

    generated = 0
    failed = 0

    for identity, records in sorted(
        groups.items(),
        key=lambda item: (
            item[0].study_type.lower(),
            item[0].patient.lower(),
            item[0].model.lower(),
        ),
    ):
        group_dir = output_directory(output_root, identity)
        group_dir.mkdir(parents=True, exist_ok=True)

        try:
            records_to_dataframe(records, top_n=top_n).to_csv(
                group_dir / "trials_used_for_parallel_coordinates.csv",
                index=False,
            )
            write_summary_json(
                records,
                group_dir / "parallel_coordinates_summary.json",
                top_n=top_n,
                objective_index=objective_index,
                objective_label=args.objective_label,
            )
            image_path = plot_parallel_coordinates(
                records,
                group_dir,
                top_n=top_n,
                objective_label=args.objective_label,
            )

        except Exception as exc:
            failed += 1
            print(
                f"[ERROR] {identity.study_type}/{identity.patient}/"
                f"{identity.model}: {exc}",
                file=sys.stderr,
            )
            continue

        generated += 1
        print(f"[OK] {image_path}")

    print(f"Generated: {generated}; failed: {failed}")
    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
