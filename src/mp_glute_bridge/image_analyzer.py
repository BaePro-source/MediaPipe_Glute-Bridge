from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

from mp_glute_bridge.angle_utils import build_angle_dataframe
from mp_glute_bridge.io_utils import ensure_directory, write_json


POSE_LANDMARK = mp.solutions.pose.PoseLandmark
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
DRAWING_UTILS = mp.solutions.drawing_utils
DRAWING_STYLES = mp.solutions.drawing_styles

LABEL_LANDMARKS = {
    POSE_LANDMARK.NOSE,
    POSE_LANDMARK.LEFT_SHOULDER,
    POSE_LANDMARK.RIGHT_SHOULDER,
    POSE_LANDMARK.LEFT_HIP,
    POSE_LANDMARK.RIGHT_HIP,
    POSE_LANDMARK.LEFT_KNEE,
    POSE_LANDMARK.RIGHT_KNEE,
    POSE_LANDMARK.LEFT_ANKLE,
    POSE_LANDMARK.RIGHT_ANKLE,
    POSE_LANDMARK.LEFT_FOOT_INDEX,
    POSE_LANDMARK.RIGHT_FOOT_INDEX,
}

PHASE_ORDER = {"worst": 0, "best": 1}


@dataclass
class ImageSampleAnalysisResult:
    sample_name: str
    processed_phases: list[str]
    landmarks_csv: str
    angles_csv: str
    summary_json: str
    annotated_images: dict[str, str]


class PoseImageAnalyzer:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

    def analyze_sample(
        self,
        sample_name: str,
        phase_image_paths: dict[str, Path],
        output_dir: str | Path,
        angle_defs: list[dict],
        save_annotated_images: bool = True,
        flip_horizontal: bool = False,
    ) -> ImageSampleAnalysisResult:
        sample_output_dir = ensure_directory(Path(output_dir) / sample_name)
        landmark_rows: list[dict] = []
        annotated_images: dict[str, str] = {}
        phase_status: dict[str, dict] = {}

        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as pose:
            for phase_name, image_path in phase_image_paths.items():
                frame_index = PHASE_ORDER[phase_name]
                image = cv2.imread(str(image_path))
                if image is None:
                    phase_status[phase_name] = {"status": "read_failed", "image_path": str(image_path)}
                    continue

                if flip_horizontal:
                    image = cv2.flip(image, 1)

                height, width = image.shape[:2]
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_image)

                if save_annotated_images:
                    annotated = image.copy()
                    if results.pose_landmarks:
                        DRAWING_UTILS.draw_landmarks(
                            annotated,
                            results.pose_landmarks,
                            POSE_CONNECTIONS,
                            landmark_drawing_spec=DRAWING_STYLES.get_default_pose_landmarks_style(),
                        )
                        self._draw_landmark_labels(annotated, results.pose_landmarks.landmark, width, height)

                    annotated_path = sample_output_dir / f"{sample_name}_{phase_name}_skeleton.jpg"
                    cv2.imwrite(str(annotated_path), annotated)
                    annotated_images[phase_name] = str(annotated_path)

                if not results.pose_landmarks:
                    phase_status[phase_name] = {"status": "pose_not_detected", "image_path": str(image_path)}
                    continue

                phase_status[phase_name] = {
                    "status": "ok",
                    "image_path": str(image_path),
                    "flip_horizontal": flip_horizontal,
                    "width": width,
                    "height": height,
                }
                for landmark_enum, landmark in zip(POSE_LANDMARK, results.pose_landmarks.landmark):
                    landmark_rows.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_sec": None,
                            "phase": phase_name,
                            "landmark_name": landmark_enum.name,
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "visibility": landmark.visibility,
                            "presence": getattr(landmark, "presence", None),
                        }
                    )

        landmarks_df = pd.DataFrame(landmark_rows)
        if not landmarks_df.empty:
            landmarks_df = landmarks_df.sort_values(["frame_index", "landmark_name"]).reset_index(drop=True)

        landmarks_csv_path = sample_output_dir / f"{sample_name}_landmarks.csv"
        landmarks_df.to_csv(landmarks_csv_path, index=False)

        angles_long_df = build_angle_dataframe(landmarks_df, angle_defs) if not landmarks_df.empty else pd.DataFrame()
        angles_csv_path = sample_output_dir / f"{sample_name}_angles.csv"
        angle_summary = self._build_angle_summary(sample_name, angles_long_df)
        pd.DataFrame([angle_summary["angles"]]).to_csv(angles_csv_path, index=False)

        summary_payload = {
            "sample_name": sample_name,
            "analysis_mode": "image_pair",
            "flip_horizontal": flip_horizontal,
            "phase_status": phase_status,
            "landmarks_csv": str(landmarks_csv_path),
            "angles_csv": str(angles_csv_path),
            "annotated_images": annotated_images,
            "angles": angle_summary["angles"],
        }
        summary_json_path = sample_output_dir / f"{sample_name}_summary.json"
        write_json(summary_json_path, summary_payload)

        return ImageSampleAnalysisResult(
            sample_name=sample_name,
            processed_phases=sorted(phase_status.keys()),
            landmarks_csv=str(landmarks_csv_path),
            angles_csv=str(angles_csv_path),
            summary_json=str(summary_json_path),
            annotated_images=annotated_images,
        )

    def _build_angle_summary(self, sample_name: str, angles_long_df: pd.DataFrame) -> dict:
        result = {
            "sample_name": sample_name,
            "angles": {
                "worst_alpha": None,
                "worst_beta": None,
                "best_alpha": None,
                "best_beta": None,
            },
        }
        if angles_long_df.empty:
            return result

        phase_df_map = {
            "worst": angles_long_df[angles_long_df["frame_index"] == PHASE_ORDER["worst"]],
            "best": angles_long_df[angles_long_df["frame_index"] == PHASE_ORDER["best"]],
        }
        for phase_name, phase_df in phase_df_map.items():
            if phase_df.empty:
                continue
            row = phase_df.iloc[0].to_dict()
            if phase_name == "worst":
                result["angles"]["worst_alpha"] = row.get("worst_alpha")
                result["angles"]["worst_beta"] = row.get("worst_beta")
            else:
                result["angles"]["best_alpha"] = row.get("best_alpha")
                result["angles"]["best_beta"] = row.get("best_beta")
        return result

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
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
