"""
aggregate_results.py
=====================
Concatenate all per-run metrics CSVs in a folder into one combined
comparison table (used for set2 / set3 hyperparameter sweeps, where
each run writes its own small CSV).

Usage:
    python aggregate_results.py --results_dir results/set2 --out results/set2_combined.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def run(args):
    files = sorted(Path(args.results_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {args.results_dir}")

    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True).sort_values("r2", ascending=False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"Combined {len(files)} files -> {args.out}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args)
