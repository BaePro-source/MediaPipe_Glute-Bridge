from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp_glute_bridge.angle_utils import load_angle_config
from mp_glute_bridge.image_analyzer import PoseImageAnalyzer
from mp_glute_bridge.io_utils import ensure_directory, find_phase_image, find_sample_directories, has_existing_output, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch analyze best/worst example images with MediaPipe Pose.")
    parser.add_argument("--input-dir", required=True, help="Directory containing sample subdirectories.")
    parser.add_argument("--output-dir", required=True, help="Directory to write CSV and JSON results.")
    parser.add_argument("--angle-config", required=True, help="JSON file describing angle definitions.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--only-new", action="store_true", help="Process only samples that do not already have an output folder and summary file.")
    parser.add_argument("--flip-samples", default="", help="Comma-separated sample names to flip horizontally before pose estimation, e.g. gb2,gb5")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_directory(args.output_dir)
    angle_defs = load_angle_config(args.angle_config)
    sample_dirs = find_sample_directories(input_dir)
    flip_samples = {name.strip() for name in args.flip_samples.split(",") if name.strip()}

    if not sample_dirs:
        print(f"No sample directories found in: {input_dir}")
        return 1

    analyzer = PoseImageAnalyzer(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
    )

    results = []
    skipped_samples = []
    invalid_samples = []

    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        if args.only_new and has_existing_output(output_dir, sample_name):
            skipped_samples.append(
                {
                    "sample_name": sample_name,
                    "reason": "existing_output",
                    "output_dir": str(output_dir / sample_name),
                }
            )
            print(f"Skipping existing sample: {sample_name}")
            continue

        phase_image_paths = {
            "worst": find_phase_image(sample_dir, "worst"),
            "best": find_phase_image(sample_dir, "best"),
        }
        missing_phases = [phase for phase, image_path in phase_image_paths.items() if image_path is None]
        if missing_phases:
            invalid_samples.append(
                {
                    "sample_name": sample_name,
                    "reason": "missing_phase_images",
                    "missing_phases": missing_phases,
                }
            )
            print(f"Skipping invalid sample: {sample_name} (missing {', '.join(missing_phases)})")
            continue

        result = analyzer.analyze_sample(
            sample_name=sample_name,
            phase_image_paths={phase: path for phase, path in phase_image_paths.items() if path is not None},
            output_dir=output_dir,
            angle_defs=angle_defs,
            save_annotated_images=True,
            flip_horizontal=sample_name in flip_samples,
        )
        results.append(
            {
                "sample_name": result.sample_name,
                "flip_horizontal": sample_name in flip_samples,
                "processed_phases": result.processed_phases,
                "landmarks_csv": result.landmarks_csv,
                "angles_csv": result.angles_csv,
                "summary_json": result.summary_json,
                "annotated_images": result.annotated_images,
            }
        )

    batch_summary_path = output_dir / "batch_summary.json"
    write_json(
        batch_summary_path,
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "analysis_mode": "image_pair",
            "processed_sample_count": len(results),
            "skipped_sample_count": len(skipped_samples),
            "invalid_sample_count": len(invalid_samples),
            "samples": results,
            "skipped_samples": skipped_samples,
            "invalid_samples": invalid_samples,
        },
    )
    print(f"Processed {len(results)} sample(s).")
    if skipped_samples:
        print(f"Skipped {len(skipped_samples)} existing sample(s).")
    if invalid_samples:
        print(f"Skipped {len(invalid_samples)} invalid sample(s).")
    print(f"Batch summary saved to: {batch_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
