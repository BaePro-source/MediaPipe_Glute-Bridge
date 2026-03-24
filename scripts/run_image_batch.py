from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp_glute_bridge.angle_utils import load_angle_config
from mp_glute_bridge.image_analyzer import PoseImageAnalyzer
from mp_glute_bridge.judgment_utils import classify_angles, load_judgment_ranges
from mp_glute_bridge.io_utils import ensure_directory, find_image_files, find_phase_image, find_sample_directories, has_existing_output, write_json


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
    parser.add_argument("--judgment-ranges", default=None, help="Optional JSON file with posture judgment ranges.")
    parser.add_argument("--judgment-method", default="min_max", choices=["min_max", "mean_std"], help="Range rule for automatic posture classification.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_directory(args.output_dir)
    angle_defs = load_angle_config(args.angle_config)
    sample_dirs = find_sample_directories(input_dir)
    flip_samples = {name.strip() for name in args.flip_samples.split(",") if name.strip()}
    judgment_ranges = load_judgment_ranges(args.judgment_ranges) if args.judgment_ranges else None

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
        available_phase_paths = {phase: path for phase, path in phase_image_paths.items() if path is not None}
        image_files = find_image_files(sample_dir)

        if not available_phase_paths and len(image_files) == 1:
            available_phase_paths = {"single": image_files[0]}

        missing_phases = [phase for phase, image_path in phase_image_paths.items() if image_path is None]
        if not available_phase_paths or (len(image_files) > 1 and len(available_phase_paths) == 1):
            invalid_samples.append(
                {
                    "sample_name": sample_name,
                    "reason": "missing_phase_images",
                    "missing_phases": missing_phases,
                    "found_images": [str(path) for path in image_files],
                }
            )
            print(f"Skipping invalid sample: {sample_name} (missing {', '.join(missing_phases)})")
            continue

        result = analyzer.analyze_sample(
            sample_name=sample_name,
            phase_image_paths=available_phase_paths,
            output_dir=output_dir,
            angle_defs=angle_defs,
            save_annotated_images=True,
            flip_horizontal=sample_name in flip_samples,
        )

        classification = None
        classification_json_path = None
        if judgment_ranges is not None:
            classification = classify_angles(result.angles, judgment_ranges, method=args.judgment_method)
            classification_json_path = output_dir / sample_name / f"{sample_name}_classification.json"
            write_json(classification_json_path, classification)

            with Path(result.summary_json).open("r", encoding="utf-8") as f:
                summary_payload = json.load(f)
            summary_payload["classification"] = classification
            summary_payload["classification_json"] = str(classification_json_path)
            write_json(result.summary_json, summary_payload)

        results.append(
            {
                "sample_name": result.sample_name,
                "flip_horizontal": sample_name in flip_samples,
                "processed_phases": result.processed_phases,
                "landmarks_csv": result.landmarks_csv,
                "angles_csv": result.angles_csv,
                "summary_json": result.summary_json,
                "annotated_images": result.annotated_images,
                "classification": classification,
                "classification_json": str(classification_json_path) if classification_json_path else None,
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
