from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


LANDMARK_ALIASES = {
    "L_FOOT": "LEFT_FOOT_INDEX",
    "R_FOOT": "RIGHT_FOOT_INDEX",
    "L_HEEL": "LEFT_HEEL",
    "R_HEEL": "RIGHT_HEEL",
    "L_ANKLE": "LEFT_ANKLE",
    "R_ANKLE": "RIGHT_ANKLE",
    "L_KNEE": "LEFT_KNEE",
    "R_KNEE": "RIGHT_KNEE",
    "L_HIP": "LEFT_HIP",
    "R_HIP": "RIGHT_HIP",
    "L_SHOULDER": "LEFT_SHOULDER",
    "R_SHOULDER": "RIGHT_SHOULDER",
}


def load_angle_config(config_path: str | None) -> list[dict]:
    if not config_path:
        return []

    with Path(config_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("angles", [])


def load_window_config(config_path: str | None) -> list[dict]:
    if not config_path:
        return []

    with Path(config_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("windows", [])


def load_video_window_config(config_path: str | None) -> dict:
    if not config_path:
        return {}

    with Path(config_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("videos", {})


def resolve_window_defs(video_stem: str, default_window_defs: list[dict], video_window_map: dict) -> list[dict]:
    video_payload = video_window_map.get(video_stem)
    if not video_payload:
        return default_window_defs

    return video_payload.get("windows", default_window_defs)


def resolve_landmark_name(name: str) -> str:
    return LANDMARK_ALIASES.get(name, name)


def calculate_angle(point_a: tuple[float, float], point_b: tuple[float, float], point_c: tuple[float, float]) -> float:
    ba_x = point_a[0] - point_b[0]
    ba_y = point_a[1] - point_b[1]
    bc_x = point_c[0] - point_b[0]
    bc_y = point_c[1] - point_b[1]

    dot = (ba_x * bc_x) + (ba_y * bc_y)
    mag_ba = math.hypot(ba_x, ba_y)
    mag_bc = math.hypot(bc_x, bc_y)

    if mag_ba == 0 or mag_bc == 0:
        return float("nan")

    cos_theta = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_theta))


def build_angle_dataframe(landmarks_df: pd.DataFrame, angle_defs: list[dict]) -> pd.DataFrame:
    if landmarks_df.empty or not angle_defs:
        return pd.DataFrame()

    rows: list[dict] = []

    for frame_index, frame_df in landmarks_df.groupby("frame_index"):
        landmark_map = {
            row["landmark_name"]: (row["x"], row["y"])
            for _, row in frame_df.iterrows()
        }
        timestamp_sec = frame_df["timestamp_sec"].iloc[0] if "timestamp_sec" in frame_df.columns else None
        result = {"frame_index": frame_index, "timestamp_sec": timestamp_sec}

        for angle_def in angle_defs:
            p1, p2, p3 = [resolve_landmark_name(point) for point in angle_def["points"]]
            if p1 in landmark_map and p2 in landmark_map and p3 in landmark_map:
                result[angle_def["name"]] = calculate_angle(
                    landmark_map[p1],
                    landmark_map[p2],
                    landmark_map[p3],
                )
            else:
                result[angle_def["name"]] = float("nan")

        rows.append(result)

    return pd.DataFrame(rows)


def summarize_angle_windows(angles_df: pd.DataFrame, window_defs: list[dict]) -> dict:
    if angles_df.empty or not window_defs:
        return {"windows": []}

    summaries: list[dict] = []
    all_angle_columns = [col for col in angles_df.columns if col not in {"frame_index", "timestamp_sec"}]

    for window_def in window_defs:
        start_sec = window_def["start_sec"]
        end_sec = window_def["end_sec"]
        angle_columns = window_def.get("angle_names", all_angle_columns)
        window_df = angles_df[
            (angles_df["timestamp_sec"] >= start_sec) &
            (angles_df["timestamp_sec"] <= end_sec)
        ]

        angle_summary: dict[str, dict] = {}
        for angle_column in angle_columns:
            series = window_df[angle_column].dropna()
            angle_summary[angle_column] = {
                "frame_count": int(series.shape[0]),
                "mean": float(series.mean()) if not series.empty else None,
                "min": float(series.min()) if not series.empty else None,
                "max": float(series.max()) if not series.empty else None,
            }

        summaries.append(
            {
                "name": window_def["name"],
                "start_sec": start_sec,
                "end_sec": end_sec,
                "angles": angle_summary,
            }
        )

    return {"windows": summaries}
