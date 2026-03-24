from __future__ import annotations

import json
from pathlib import Path


def load_judgment_ranges(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def classify_angles(angles: dict[str, float | None], judgment_ranges: dict, method: str = "min_max") -> dict:
    if "alpha" in angles and "beta" in angles:
        return classify_single_image_angles(angles, judgment_ranges, method=method)

    ranges = judgment_ranges.get("ranges", [])
    range_map = {item["metric"]: item for item in ranges}

    posture_checks: dict[str, dict] = {}
    for posture_name, alpha_metric, beta_metric in [
        ("worst", "worst_alpha", "worst_beta"),
        ("best", "best_alpha", "best_beta"),
    ]:
        alpha_value = angles.get(alpha_metric)
        beta_value = angles.get(beta_metric)
        alpha_range = _resolve_range(range_map.get(alpha_metric, {}), method)
        beta_range = _resolve_range(range_map.get(beta_metric, {}), method)

        alpha_in_range = _is_in_range(alpha_value, alpha_range)
        beta_in_range = _is_in_range(beta_value, beta_range)
        posture_checks[posture_name] = {
            "alpha_metric": alpha_metric,
            "beta_metric": beta_metric,
            "alpha_value": alpha_value,
            "beta_value": beta_value,
            "alpha_range": alpha_range,
            "beta_range": beta_range,
            "alpha_in_range": alpha_in_range,
            "beta_in_range": beta_in_range,
            "both_in_range": alpha_in_range and beta_in_range,
        }

    worst_match = posture_checks["worst"]["both_in_range"]
    best_match = posture_checks["best"]["both_in_range"]

    if worst_match and not best_match:
        final_label = "worst"
    elif best_match and not worst_match:
        final_label = "best"
    elif best_match and worst_match:
        final_label = "ambiguous"
    else:
        final_label = "uncertain"

    return {
        "method": method,
        "final_label": final_label,
        "classification_mode": "paired_images",
        "posture_checks": posture_checks,
    }


def classify_single_image_angles(angles: dict[str, float | None], judgment_ranges: dict, method: str = "min_max") -> dict:
    ranges = judgment_ranges.get("ranges", [])
    range_map = {item["metric"]: item for item in ranges}
    alpha_value = angles.get("alpha")
    beta_value = angles.get("beta")

    posture_checks: dict[str, dict] = {}
    for posture_name, alpha_metric, beta_metric in [
        ("worst", "worst_alpha", "worst_beta"),
        ("best", "best_alpha", "best_beta"),
    ]:
        alpha_range = _resolve_range(range_map.get(alpha_metric, {}), method)
        beta_range = _resolve_range(range_map.get(beta_metric, {}), method)
        alpha_in_range = _is_in_range(alpha_value, alpha_range)
        beta_in_range = _is_in_range(beta_value, beta_range)
        posture_checks[posture_name] = {
            "alpha_metric": alpha_metric,
            "beta_metric": beta_metric,
            "alpha_value": alpha_value,
            "beta_value": beta_value,
            "alpha_range": alpha_range,
            "beta_range": beta_range,
            "alpha_in_range": alpha_in_range,
            "beta_in_range": beta_in_range,
            "both_in_range": alpha_in_range and beta_in_range,
        }

    worst_match = posture_checks["worst"]["both_in_range"]
    best_match = posture_checks["best"]["both_in_range"]

    if worst_match and not best_match:
        final_label = "worst"
    elif best_match and not worst_match:
        final_label = "best"
    elif best_match and worst_match:
        final_label = "ambiguous"
    else:
        final_label = "uncertain"

    return {
        "method": method,
        "final_label": final_label,
        "classification_mode": "single_image",
        "posture_checks": posture_checks,
    }


def _resolve_range(range_item: dict, method: str) -> dict:
    if method == "mean_std":
        return {
            "lower": range_item.get("mean_minus_std"),
            "upper": range_item.get("mean_plus_std"),
        }

    return {
        "lower": range_item.get("min_value"),
        "upper": range_item.get("max_value"),
    }


def _is_in_range(value: float | None, range_def: dict) -> bool:
    if value is None:
        return False

    lower = range_def.get("lower")
    upper = range_def.get("upper")
    if lower is None or upper is None:
        return False
    return lower <= value <= upper
