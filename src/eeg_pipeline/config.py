from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "data_root": "../EEG_data",
    "output_root": "outputs",
    "expected_subjects": [
        "25009",
        "25010",
        "25011",
        "25012",
        "25013",
        "25014",
        "25015",
        "25016",
        "25017",
        "25018",
        "25019",
        "25020",
        "25021",
        "25022",
        "25023",
    ],
    "driving_conditions": {
        "normal_driving": "normal",
        "fast_driving": "fast",
    },
    "resting_conditions": [
        "resting_before_normal_driving",
        "resting_after_normal_driving",
        "resting_before_fast_driving",
        "resting_after_fast_driving",
    ],
    "crop": {
        "events_per_lap": 9,
        "laps_per_session": 6,
        "accept_truncated_laps": [
            {
                "subject_id": "25019",
                "condition": "fast_driving",
                "lap_index": 4,
                "reason": "Recording stopped before the final marker; user confirmed the available lap data are usable.",
            }
        ],
    },
    "preprocess": {
        "target_sfreq": 250.0,
        "l_freq": 1.0,
        "h_freq": 40.0,
        "reference_channels": ["A1", "A2"],
        "acquisition_ground": "AFz",
        "acquisition_reference": "FCz",
        "offline_reference_mode": "average_of_A1_A2",
        "montage": "standard_1020",
        "bad_channel_z": 6.0,
        "flat_std_uV": 0.1,
        "baseline_mode": "none",
        "ica": {
            "enabled": True,
            "method": "infomax",
            "n_components": 0.99,
            "min_components": 2,
            "random_state": 97,
            "max_iter": 2000,
            "eye_channels": ["Fp1", "Fp2"],
            "eye_corr_threshold": 0.35,
            "muscle_band": [30.0, 40.0],
            "total_band": [1.0, 40.0],
            "muscle_ratio_threshold": 0.45,
            "kurtosis_threshold": 8.0,
            "max_auto_exclude": 2,
        },
    },
    "features": {
        "windows_per_lap": 6,
        "exclude_channels": ["A1", "A2"],
        "bands": {
            "delta": [1.0, 4.0],
            "theta": [4.0, 8.0],
            "alpha": [8.0, 13.0],
            "beta": [13.0, 30.0],
            "low_gamma": [30.0, 40.0],
        },
    },
    "model": {
        "group_level": "lap",
        "balance_method": "class_weight",
        "select_k": 100,
        "labels": ["misc_3class_id", "misc_binary_2plus", "misc_binary_6plus", "susceptibility_id"],
        "random_state": 97,
    },
}


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        return deepcopy(DEFAULT_CONFIG)
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    return _merge_dict(DEFAULT_CONFIG, user_config)


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
