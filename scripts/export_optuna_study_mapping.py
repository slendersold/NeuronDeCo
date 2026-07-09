#!/usr/bin/env python3
"""Scan an Optuna run directory and print/export study mapping YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from confirmatory_support import audit_study_cv_mode, scan_run_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Optuna study mapping from a run directory.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="Write YAML to this path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    refs = scan_run_dir(run_dir)
    studies = [
        {
            "patient": ref.patient,
            "model": ref.model,
            "storage": str(ref.storage),
            "study_name": ref.study_name,
        }
        for ref in refs
    ]
    payload = {
        "optuna": {"run_dir": str(run_dir)},
        "studies": studies,
    }
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)

    print("\n# CV audit", file=sys.stderr)
    for ref in refs:
        row = audit_study_cv_mode(ref)
        print(
            f"{row.patient}\t{row.model}\t{row.cv_mode}\t{row.number_of_folds}\t{row.evidence_source}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
