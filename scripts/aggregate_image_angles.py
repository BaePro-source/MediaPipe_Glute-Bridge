from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate image angle CSV files across samples.")
    parser.add_argument("--output-dir", required=True, help="Root directory containing sample output folders.")
    parser.add_argument("--pattern", default="*_angles.csv", help="Glob pattern for per-sample angle CSV files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    csv_paths = sorted(
        path for path in output_dir.glob(f"*/{args.pattern}")
        if path.parent.name != "aggregate"
    )

    if not csv_paths:
        print(f"No angle CSV files found in: {output_dir}")
        return 1

    rows: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        sample_name = csv_path.parent.name
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        if "sample_name" not in df.columns:
            df.insert(0, "sample_name", sample_name)
        rows.append(df)

    if not rows:
        print("Angle CSV files were found, but all were empty.")
        return 1

    combined_df = pd.concat(rows, ignore_index=True)
    metric_columns = [column for column in combined_df.columns if column != "sample_name"]
    mean_row = combined_df[metric_columns].mean(numeric_only=True)
    min_row = combined_df[metric_columns].min(numeric_only=True)
    max_row = combined_df[metric_columns].max(numeric_only=True)
    std_row = combined_df[metric_columns].std(numeric_only=True)

    aggregate_dir = output_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    combined_csv_path = aggregate_dir / "all_samples_angles.csv"
    combined_df.to_csv(combined_csv_path, index=False)

    mean_csv_path = aggregate_dir / "mean_angles.csv"
    pd.DataFrame([mean_row.to_dict()]).to_csv(mean_csv_path, index=False)

    stats_csv_path = aggregate_dir / "angle_stats.csv"
    stats_df = pd.DataFrame(
        [
            {"stat": "mean", **mean_row.to_dict()},
            {"stat": "min", **min_row.to_dict()},
            {"stat": "max", **max_row.to_dict()},
            {"stat": "std", **std_row.to_dict()},
        ]
    )
    stats_df.to_csv(stats_csv_path, index=False)

    range_json_path = aggregate_dir / "angle_ranges.json"
    range_payload = {
        "sample_count": int(combined_df.shape[0]),
        "worst_range": {
            "alpha": {
                "mean": float(mean_row["worst_alpha"]),
                "min": float(min_row["worst_alpha"]),
                "max": float(max_row["worst_alpha"]),
                "std": float(std_row["worst_alpha"]),
            },
            "beta": {
                "mean": float(mean_row["worst_beta"]),
                "min": float(min_row["worst_beta"]),
                "max": float(max_row["worst_beta"]),
                "std": float(std_row["worst_beta"]),
            },
        },
        "best_range": {
            "alpha": {
                "mean": float(mean_row["best_alpha"]),
                "min": float(min_row["best_alpha"]),
                "max": float(max_row["best_alpha"]),
                "std": float(std_row["best_alpha"]),
            },
            "beta": {
                "mean": float(mean_row["best_beta"]),
                "min": float(min_row["best_beta"]),
                "max": float(max_row["best_beta"]),
                "std": float(std_row["best_beta"]),
            },
        },
        "all_samples_csv": str(combined_csv_path),
        "mean_csv": str(mean_csv_path),
        "stats_csv": str(stats_csv_path),
    }
    with range_json_path.open("w", encoding="utf-8") as f:
        json.dump(range_payload, f, ensure_ascii=False, indent=2)

    summary_json_path = aggregate_dir / "mean_angles.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_count": int(combined_df.shape[0]),
                "metrics": {
                    key: {
                        "mean": float(mean_row[key]),
                        "min": float(min_row[key]),
                        "max": float(max_row[key]),
                        "std": float(std_row[key]),
                    }
                    for key in metric_columns
                },
                "all_samples_csv": str(combined_csv_path),
                "mean_csv": str(mean_csv_path),
                "stats_csv": str(stats_csv_path),
                "ranges_json": str(range_json_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Aggregated {combined_df.shape[0]} sample(s).")
    print(f"Combined CSV: {combined_csv_path}")
    print(f"Mean CSV: {mean_csv_path}")
    print(f"Stats CSV: {stats_csv_path}")
    print(f"Ranges JSON: {range_json_path}")
    print(f"Mean JSON: {summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
