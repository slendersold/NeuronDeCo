#!/usr/bin/env python3
"""Generate publication figures from confirmatory analysis outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot confirmatory analysis results.")
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument(
        "--old-results-json",
        type=Path,
        default=None,
        help="Optional JSON map model->old macro-F1 median for comparison table.",
    )
    return p.parse_args()


def save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(summary: pd.DataFrame, out: Path, dpi: int) -> None:
    pivot = summary.pivot(index="patient", columns="model", values="macro_f1_median")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title("Macro-F1 median by patient and model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Patient")
    save_fig(fig, out / "patient_model_heatmap", dpi)


def plot_boxplot(summary: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=summary, x="model", y="macro_f1_median", ax=ax)
    sns.stripplot(
        data=summary,
        x="model",
        y="macro_f1_median",
        color="black",
        alpha=0.5,
        ax=ax,
    )
    ax.set_title("Patient-level macro-F1 median across models")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro-F1 median")
    save_fig(fig, out / "macro_f1_boxplot", dpi)


def plot_paired_lines(summary: pd.DataFrame, out: Path, dpi: int) -> None:
    pivot = summary.pivot(index="patient", columns="model", values="macro_f1_median")
    fig, ax = plt.subplots(figsize=(8, 4))
    for patient in pivot.index:
        ax.plot(pivot.columns, pivot.loc[patient], marker="o", label=patient, alpha=0.8)
    ax.set_title("Paired macro-F1 median lines by patient")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro-F1 median")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save_fig(fig, out / "paired_lines_macro_f1", dpi)


def plot_confusion_matrices(fold_metrics: pd.DataFrame, out: Path, dpi: int) -> None:
    for model in sorted(fold_metrics["model"].unique()):
        sub = fold_metrics[fold_metrics["model"] == model]
        cm = np.zeros((2, 2), dtype=int)
        for mat in sub["confusion_matrix"]:
            arr = np.asarray(mat, dtype=int)
            cm += arr
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Aggregated confusion matrix: {model}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        save_fig(fig, out / f"confusion_matrix__{model}", dpi)


def plot_error_overlap(oof: pd.DataFrame, out: Path, dpi: int) -> None:
    if oof.empty:
        return
    wide = oof.pivot_table(
        index=["patient", "epoch_id", "fold"],
        columns="model",
        values="predicted_label",
        aggfunc="first",
    )
    true = oof.drop_duplicates(["patient", "epoch_id", "fold"])[
        ["patient", "epoch_id", "fold", "true_label"]
    ].set_index(["patient", "epoch_id", "fold"])
    err = wide.join(true)
    models = [c for c in wide.columns if c in {"svm", "alexnet", "transformer"}]
    if len(models) < 2:
        return
    overlap_counts: dict[str, int] = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            both_wrong = (
                (err[a] != err["true_label"]) & (err[b] != err["true_label"])
            ).sum()
            overlap_counts[f"{a}&{b}"] = int(both_wrong)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(overlap_counts.keys(), overlap_counts.values())
    ax.set_title("Shared misclassifications between model pairs")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20)
    save_fig(fig, out / "error_overlap_between_models", dpi)


def main() -> None:
    args = parse_args()
    root = args.input_root.expanduser().resolve()
    out = (args.output_dir or root / "figures").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary_path = root / "patient_model_summary.csv"
    fold_path = root / "all_fold_metrics.csv"
    oof_path = root / "all_oof_predictions.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary = pd.read_csv(summary_path)
    fold_metrics = pd.read_csv(fold_path) if fold_path.is_file() else pd.DataFrame()
    oof = pd.read_csv(oof_path) if oof_path.is_file() else pd.DataFrame()

    plot_heatmap(summary, out, args.dpi)
    plot_boxplot(summary, out, args.dpi)
    plot_paired_lines(summary, out, args.dpi)
    if not fold_metrics.empty and "confusion_matrix" in fold_metrics.columns:
        import ast

        fold_metrics["confusion_matrix"] = fold_metrics["confusion_matrix"].apply(ast.literal_eval)
        plot_confusion_matrices(fold_metrics, out, args.dpi)
    if not oof.empty:
        plot_error_overlap(oof, out, args.dpi)

    stats_path = root / "statistical_summary.json"
    if stats_path.is_file() and args.old_results_json and args.old_results_json.is_file():
        with open(args.old_results_json, encoding="utf-8") as fh:
            old_map = json.load(fh)
        with open(stats_path, encoding="utf-8") as fh:
            stats = json.load(fh)
        rows = []
        for model, old_val in old_map.items():
            new_val = stats.get("global", {}).get(model, {}).get("median")
            rows.append(
                {
                    "model": model,
                    "old_macro_f1_median": old_val,
                    "confirmatory_macro_f1_median": new_val,
                    "difference": (new_val - old_val) if new_val is not None else None,
                }
            )
        pd.DataFrame(rows).to_csv(out / "old_vs_confirmatory_table.csv", index=False)

    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
