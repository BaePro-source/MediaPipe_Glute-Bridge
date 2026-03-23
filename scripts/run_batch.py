from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mp_glute_bridge.analyzer import PoseVideoAnalyzer
from mp_glute_bridge.angle_utils import load_angle_config, load_video_window_config, load_window_config, resolve_window_defs
from mp_glute_bridge.io_utils import ensure_directory, find_video_files, has_existing_output, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch analyze exercise videos with MediaPipe Pose.")
    parser.add_argument("--input-dir", required=True, help="Directory containing local video files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write CSV and JSON results.")
    parser.add_argument("--angle-config", default=None, help="Optional JSON file describing angle definitions.")
    parser.add_argument("--window-config", default=None, help="Optional JSON file describing named time windows.")
    parser.add_argument("--video-window-config", default=None, help="Optional JSON file describing per-video time windows.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--save-skeleton-video", action="store_true", help="Write an annotated skeleton video for each input.")
    parser.add_argument("--only-new", action="store_true", help="Process only videos that do not already have an output folder and summary file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_directory(args.output_dir)
    angle_defs = load_angle_config(args.angle_config)
    window_defs = load_window_config(args.window_config)
    video_window_map = load_video_window_config(args.video_window_config)
    video_files = find_video_files(input_dir)

    if not video_files:
        print(f"No video files found in: {input_dir}")
        return 1

    analyzer = PoseVideoAnalyzer(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
    )

    results = []
    skipped_videos = []
    for video_path in video_files:
        if args.only_new and has_existing_output(output_dir, video_path.stem):
            skipped_videos.append(
                {
                    "video_name": video_path.name,
                    "reason": "existing_output",
                    "output_dir": str(output_dir / video_path.stem),
                }
            )
            print(f"Skipping existing video: {video_path.name}")
            continue

        resolved_window_defs = resolve_window_defs(video_path.stem, window_defs, video_window_map)

        result = analyzer.analyze_video(
            video_path=video_path,
            output_dir=output_dir,
            angle_defs=angle_defs,
            window_defs=resolved_window_defs,
            save_skeleton_video=args.save_skeleton_video,
        )
        results.append(
            {
                "video_name": result.video_name,
                "window_source": "video_window_config" if video_path.stem in video_window_map else "default_window_config",
                "total_frames": result.total_frames,
                "processed_frames": result.processed_frames,
                "detected_frames": result.detected_frames,
                "fps": result.fps,
                "width": result.width,
                "height": result.height,
                "landmarks_csv": result.landmarks_csv,
                "angles_csv": result.angles_csv,
                "angle_summary_json": result.angle_summary_json,
                "skeleton_video": result.skeleton_video,
                "summary_json": result.summary_json,
            }
        )

    batch_summary_path = output_dir / "batch_summary.json"
    write_json(
        batch_summary_path,
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "processed_video_count": len(results),
            "skipped_video_count": len(skipped_videos),
            "videos": results,
            "skipped_videos": skipped_videos,
        },
    )
    print(f"Processed {len(results)} video(s).")
    if skipped_videos:
        print(f"Skipped {len(skipped_videos)} existing video(s).")
    print(f"Batch summary saved to: {batch_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
