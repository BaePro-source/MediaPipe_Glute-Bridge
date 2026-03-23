from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

from mp_glute_bridge.angle_utils import build_angle_dataframe, summarize_angle_windows
from mp_glute_bridge.io_utils import ensure_directory, write_json


POSE_LANDMARK = mp.solutions.pose.PoseLandmark
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
DRAWING_UTILS = mp.solutions.drawing_utils
DRAWING_STYLES = mp.solutions.drawing_styles

LABEL_LANDMARKS = {
    POSE_LANDMARK.NOSE,
    POSE_LANDMARK.LEFT_SHOULDER,
    POSE_LANDMARK.RIGHT_SHOULDER,
    POSE_LANDMARK.LEFT_ELBOW,
    POSE_LANDMARK.RIGHT_ELBOW,
    POSE_LANDMARK.LEFT_WRIST,
    POSE_LANDMARK.RIGHT_WRIST,
    POSE_LANDMARK.LEFT_HIP,
    POSE_LANDMARK.RIGHT_HIP,
    POSE_LANDMARK.LEFT_KNEE,
    POSE_LANDMARK.RIGHT_KNEE,
    POSE_LANDMARK.LEFT_ANKLE,
    POSE_LANDMARK.RIGHT_ANKLE,
    POSE_LANDMARK.LEFT_HEEL,
    POSE_LANDMARK.RIGHT_HEEL,
    POSE_LANDMARK.LEFT_FOOT_INDEX,
    POSE_LANDMARK.RIGHT_FOOT_INDEX,
}


@dataclass
class VideoAnalysisResult:
    video_name: str
    total_frames: int
    processed_frames: int
    detected_frames: int
    fps: float
    width: int
    height: int
    landmarks_csv: str
    angles_csv: str | None
    angle_summary_json: str | None
    skeleton_video: str | None
    summary_json: str


class PoseVideoAnalyzer:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

    def analyze_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        angle_defs: list[dict] | None = None,
        window_defs: list[dict] | None = None,
        save_skeleton_video: bool = False,
    ) -> VideoAnalysisResult:
        video_path = Path(video_path)
        output_dir = ensure_directory(output_dir)
        video_output_dir = ensure_directory(output_dir / video_path.stem)
        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        landmark_rows: list[dict] = []
        processed_frames = 0
        detected_frames = 0
        skeleton_video_path: Path | None = None
        video_writer: cv2.VideoWriter | None = None

        if save_skeleton_video:
            skeleton_video_path = video_output_dir / f"{video_path.stem}_skeleton.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(skeleton_video_path),
                fourcc,
                fps if fps > 0 else 30.0,
                (width, height),
            )

        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as pose:
            for frame_index in tqdm(range(total_frames), desc=video_path.name):
                success, frame = capture.read()
                if not success:
                    break

                processed_frames += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if video_writer is not None:
                    annotated_frame = frame.copy()
                    if results.pose_landmarks:
                        DRAWING_UTILS.draw_landmarks(
                            annotated_frame,
                            results.pose_landmarks,
                            POSE_CONNECTIONS,
                            landmark_drawing_spec=DRAWING_STYLES.get_default_pose_landmarks_style(),
                        )
                        self._draw_landmark_labels(annotated_frame, results.pose_landmarks.landmark, width, height)

                    cv2.putText(
                        annotated_frame,
                        f"frame={frame_index}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    video_writer.write(annotated_frame)

                if not results.pose_landmarks:
                    continue

                detected_frames += 1
                for landmark_enum, landmark in zip(POSE_LANDMARK, results.pose_landmarks.landmark):
                    landmark_rows.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_sec": frame_index / fps if fps else None,
                            "landmark_name": landmark_enum.name,
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "visibility": landmark.visibility,
                            "presence": getattr(landmark, "presence", None),
                        }
                    )

        capture.release()
        if video_writer is not None:
            video_writer.release()

        landmarks_df = pd.DataFrame(landmark_rows)
        base_name = video_path.stem
        landmarks_csv_path = video_output_dir / f"{base_name}_landmarks.csv"
        landmarks_df.to_csv(landmarks_csv_path, index=False)

        angles_csv_path: Path | None = None
        angle_summary_json_path: Path | None = None
        angle_defs = angle_defs or []
        window_defs = window_defs or []
        if angle_defs and not landmarks_df.empty:
            angles_df = build_angle_dataframe(landmarks_df, angle_defs)
            if not angles_df.empty:
                angles_csv_path = video_output_dir / f"{base_name}_angles.csv"
                angles_df.to_csv(angles_csv_path, index=False)

                if window_defs:
                    angle_summary_json_path = video_output_dir / f"{base_name}_angle_summary.json"
                    angle_summary = summarize_angle_windows(angles_df, window_defs)
                    write_json(angle_summary_json_path, angle_summary)

        summary = {
            "video_name": video_path.name,
            "video_path": str(video_path),
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "detected_frames": detected_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "landmarks_csv": str(landmarks_csv_path),
            "angles_csv": str(angles_csv_path) if angles_csv_path else None,
            "angle_summary_json": str(angle_summary_json_path) if angle_summary_json_path else None,
            "skeleton_video": str(skeleton_video_path) if skeleton_video_path else None,
            "detection_rate": (detected_frames / processed_frames) if processed_frames else 0.0,
        }
        summary_json_path = video_output_dir / f"{base_name}_summary.json"
        write_json(summary_json_path, summary)

        return VideoAnalysisResult(
            video_name=video_path.name,
            total_frames=total_frames,
            processed_frames=processed_frames,
            detected_frames=detected_frames,
            fps=fps,
            width=width,
            height=height,
            landmarks_csv=str(landmarks_csv_path),
            angles_csv=str(angles_csv_path) if angles_csv_path else None,
            angle_summary_json=str(angle_summary_json_path) if angle_summary_json_path else None,
            skeleton_video=str(skeleton_video_path) if skeleton_video_path else None,
            summary_json=str(summary_json_path),
        )

    def _draw_landmark_labels(self, frame, landmarks, width: int, height: int) -> None:
        for landmark_enum in LABEL_LANDMARKS:
            landmark = landmarks[landmark_enum.value]
            if landmark.visibility < 0.5:
                continue

            x = min(max(int(landmark.x * width), 0), width - 1)
            y = min(max(int(landmark.y * height), 0), height - 1)
            label = landmark_enum.name.replace("LEFT_", "L_").replace("RIGHT_", "R_")
            cv2.putText(
                frame,
                label,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
