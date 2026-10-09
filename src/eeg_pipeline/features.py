from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from .io import read_raw_signal


def misc_three_class(score: int) -> tuple[int, str]:
    if score <= 1:
        return 0, "none_to_low"
    if score <= 5:
        return 1, "mild_to_moderate"
    return 2, "high"


def split_indices(n_times: int, n_segments: int) -> list[tuple[int, int]]:
    bounds = np.linspace(0, n_times, n_segments + 1, dtype=int)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_segments) if bounds[i + 1] > bounds[i]]


def _bandpower(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    idx = (freqs >= low) & (freqs < high)
    if not np.any(idx):
        return 0.0
    integrator = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if integrator is None:
        raise AttributeError("NumPy provides neither trapezoid nor trapz")
    return float(integrator(psd[idx], freqs[idx]))


def _hjorth(x: np.ndarray) -> tuple[float, float, float]:
    d1 = np.diff(x)
    d2 = np.diff(d1)
    var0 = float(np.var(x))
    var1 = float(np.var(d1)) if len(d1) else 0.0
    var2 = float(np.var(d2)) if len(d2) else 0.0
    mobility = float(np.sqrt(var1 / var0)) if var0 > 0 else 0.0
    complexity = float(np.sqrt(var2 / var1) / mobility) if var1 > 0 and mobility > 0 else 0.0
    return var0, mobility, complexity


def _extract_window_features(
    data_v: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    bands: dict[str, list[float]],
) -> dict[str, float]:
    data_uV = data_v * 1e6
    features: dict[str, float] = {}

    for idx, ch in enumerate(ch_names):
        x = data_uV[idx]
        var, mobility, complexity = _hjorth(x)
        rms = float(np.sqrt(np.mean(x**2)))
        ptp = float(np.ptp(x))
        line_length = float(np.mean(np.abs(np.diff(x)))) if len(x) > 1 else 0.0
        features[f"{ch}__time__rms_uV"] = rms
        features[f"{ch}__time__ptp_uV"] = ptp
        features[f"{ch}__time__line_length_uV_per_sample"] = line_length
        features[f"{ch}__hjorth__activity_uV2"] = var
        features[f"{ch}__hjorth__mobility"] = mobility
        features[f"{ch}__hjorth__complexity"] = complexity
        freqs, psd = welch(x, fs=sfreq, window="hann", nperseg=min(int(sfreq * 2), len(x)))
        total_power = _bandpower(freqs, psd, min(v[0] for v in bands.values()), max(v[1] for v in bands.values()))
        for band, (low, high) in bands.items():
            power = _bandpower(freqs, psd, float(low), float(high))
            rel = power / total_power if total_power > 0 else 0.0
            de = float(0.5 * np.log(2 * np.pi * np.e * power)) if power > 0 else 0.0
            features[f"{ch}__psd__{band}_abs_uV2"] = power
            features[f"{ch}__psd__{band}_rel"] = rel
            features[f"{ch}__de__{band}"] = de

    features["qc__median_rms_uV"] = float(np.median([features[f"{ch}__time__rms_uV"] for ch in ch_names]))
    features["qc__max_abs_uV"] = float(np.max(np.abs(data_uV)))
    return features


_BAND_RANGES = {
    "delta": "1-4 Hz",
    "theta": "4-8 Hz",
    "alpha": "8-13 Hz",
    "beta": "13-30 Hz",
    "low_gamma": "30-40 Hz",
}

_PREPROCESSING = (
    "Raw data are cropped by event markers and filtered at 1-40 Hz with an IIR "
    "band-pass (there is no additional DC-offset subtraction). Robust bad-channel "
    "detection (|robust z(SD)|>6 or SD<0.1 uV) and montage-based interpolation are "
    "then applied, followed by average A1/A2 rereferencing, resampling to 250 Hz, "
    "and per-lap Infomax ICA. ICA targets 99% explained variance with a minimum "
    "of two components when a one-component solution would otherwise result, and "
    "uses fixed eye/muscle rules with at most two exclusions. No "
    "resting-state baseline subtraction or power normalization is applied. A1/A2 "
    "are not model channels; the surviving signal is converted from V to uV before "
    "feature calculation."
)

_WINDOW_DEFINITION = (
    "Six non-overlapping sample-index windows per lap; bounds are "
    "numpy.linspace(0, n_times, 7, dtype=int), so adjacent windows cover the lap "
    "exactly and differ by at most one sample."
)

_WELCH_DEFINITION = (
    "scipy.signal.welch on uV data, Hann window, nperseg=min(500, N), default 50% "
    "overlap, density scaling; band power is trapezoidal integration over [low, high)."
)


def describe_feature(
    feature_name: str,
    modeling_channels: list[str] | None = None,
) -> dict[str, str]:
    modeling_channels = modeling_channels or []

    if feature_name.startswith("qc__"):
        metric = feature_name.split("__", 1)[1]
        if metric == "median_rms_uV":
            formula = "median_c[rms_c] across the 30 model channels"
            unit = "uV"
        elif metric == "max_abs_uV":
            formula = "max_{c,t} |x_c(t)| across the window and 30 model channels"
            unit = "uV"
        else:
            formula = "quality-control summary of the preprocessed window"
            unit = "see variable name"
        return {
            "feature": feature_name,
            "scope": "quality_control",
            "feature_family": "quality_control",
            "formula": formula,
            "unit": unit,
            "band_hz": "",
            "spatial_aggregation": "window summary across all 30 model channels",
            "channels_used": ", ".join(modeling_channels),
            "preprocessing": _PREPROCESSING,
            "window_definition": _WINDOW_DEFINITION,
            "spectral_estimator": "not applicable",
        }

    parts = feature_name.split("__")
    scope = parts[0] if parts else ""
    family = parts[1] if len(parts) > 1 else ""
    metric = parts[2] if len(parts) > 2 else ""
    base_family = family
    members_text = scope

    band_name = next((name for name in _BAND_RANGES if metric.startswith(name)), "")
    band_hz = _BAND_RANGES.get(band_name, "")
    if base_family == "time":
        if "rms" in metric:
            base_formula, unit = "sqrt(mean_t[x(t)^2])", "uV"
        elif "ptp" in metric:
            base_formula, unit = "max_t[x(t)] - min_t[x(t)]", "uV"
        else:
            base_formula, unit = "mean_{t=2..N}|x(t)-x(t-1)|", "uV/sample"
    elif base_family == "hjorth":
        if "activity" in metric:
            base_formula, unit = "var_t[x(t)]", "uV^2"
        elif "mobility" in metric:
            base_formula, unit = "sqrt(var_t[Delta x(t)] / var_t[x(t)])", "sample^-1"
        else:
            base_formula, unit = "sqrt(var_t[Delta^2 x(t)] / var_t[Delta x(t)]) / mobility", "unitless"
    elif base_family == "psd":
        if "_rel" in metric:
            base_formula, unit = "band_power / total_power_1-40Hz", "unitless"
        else:
            base_formula, unit = "integral_{f in [low,high)} P_Welch(f) df", "uV^2"
    elif base_family == "de":
        base_formula, unit = "0.5 * ln(2*pi*e*band_power)", "nats"
    else:
        base_formula, unit = "derived feature value", "see variable name"

    formula = base_formula
    spatial = "single channel; no spatial averaging"
    return {
        "feature": feature_name,
        "scope": scope,
        "feature_family": base_family,
        "formula": formula,
        "unit": unit,
        "band_hz": band_hz,
        "spatial_aggregation": spatial,
        "channels_used": members_text,
        "preprocessing": _PREPROCESSING,
        "window_definition": _WINDOW_DEFINITION,
        "spectral_estimator": _WELCH_DEFINITION if base_family in {"psd", "de"} else "not applicable",
    }


def extract_features(config: dict[str, Any], preprocessed_manifest_path: Path, output_root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(preprocessed_manifest_path, dtype={"subject_id": str})
    windows_per_lap = int(config["features"]["windows_per_lap"])
    bands = config["features"]["bands"]
    exclude_channels = set(config["features"].get("exclude_channels", []))
    rows: list[dict[str, Any]] = []

    for _, row in manifest.iterrows():
        path = row.get("preprocessed_file")
        if not isinstance(path, str) or not path:
            continue
        raw = read_raw_signal(path, preload=True)
        keep_channels = [ch for ch in raw.ch_names if ch not in exclude_channels]
        raw.pick(keep_channels)
        ch_names = list(raw.ch_names)
        sfreq = float(raw.info["sfreq"])
        data = raw.get_data()

        for window_index, (start, stop) in enumerate(split_indices(raw.n_times, windows_per_lap), start=1):
            misc_score = int(row["misc_score"]) if pd.notna(row["misc_score"]) else -1
            misc_class, misc_label = misc_three_class(misc_score) if misc_score >= 0 else (-1, "missing")
            sample_id = f"sub-{row['subject_id']}_{row['condition']}_lap{int(row['lap_index']):02d}_win{window_index:02d}"
            item = {
                "sample_id": sample_id,
                "subject_id": str(row["subject_id"]),
                "susceptibility_group": row["susceptibility_group"],
                "susceptibility_id": row["susceptibility_id"],
                "condition": row["condition"],
                "condition_label": row["condition_label"],
                "session_id": row["session_id"],
                "lap_index": int(row["lap_index"]),
                "window_index": window_index,
                "lap_group": row["lap_group"],
                "misc_score": misc_score,
                "misc_3class_id": misc_class,
                "misc_3class_label": misc_label,
                "misc_binary_2plus": 1 if misc_score >= 2 else 0 if misc_score >= 0 else -1,
                "misc_binary_6plus": 1 if misc_score >= 6 else 0 if misc_score >= 0 else -1,
                "window_start_sec": start / sfreq,
                "window_end_sec": stop / sfreq,
            }
            item.update(_extract_window_features(data[:, start:stop], ch_names, sfreq, bands))
            rows.append(item)

    features = pd.DataFrame(rows)
    feature_dir = output_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_dir / "features.csv", index=False, encoding="utf-8-sig")
    metadata_cols = metadata_columns()
    feature_cols = [c for c in features.columns if c not in metadata_cols]
    model_channels = []
    for feature_name in feature_cols:
        scope = feature_name.split("__", 1)[0]
        if scope not in {"qc"} and scope not in model_channels:
            model_channels.append(scope)
    dictionary = pd.DataFrame(
        [describe_feature(c, modeling_channels=model_channels) for c in feature_cols]
    )
    dictionary_path = feature_dir / "feature_dictionary.csv"
    try:
        dictionary.to_csv(dictionary_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        dictionary.to_csv(feature_dir / "feature_dictionary_reconstruction.csv", index=False, encoding="utf-8-sig")
    _write_channel_quality_summary(features, feature_dir)
    return features


def _write_channel_quality_summary(features: pd.DataFrame, feature_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    rms_cols = [c for c in features.columns if c.endswith("__time__rms_uV")]
    for col in rms_cols:
        channel = col.split("__", 1)[0]
        metrics = {
            "rms_uV": f"{channel}__time__rms_uV",
            "ptp_uV": f"{channel}__time__ptp_uV",
            "line_length_uV_per_sample": f"{channel}__time__line_length_uV_per_sample",
            "low_gamma_rel": f"{channel}__psd__low_gamma_rel",
        }
        for condition, group in features.groupby("condition_label", dropna=False):
            item: dict[str, Any] = {"condition_label": condition, "channel": channel, "n_windows": len(group)}
            for metric, metric_col in metrics.items():
                if metric_col not in group:
                    continue
                values = group[metric_col].replace([np.inf, -np.inf], np.nan).dropna()
                item[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
                item[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
                item[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            rows.append(item)
    pd.DataFrame(rows).to_csv(feature_dir / "channel_quality_summary.csv", index=False, encoding="utf-8-sig")


def metadata_columns() -> set[str]:
    return {
        "sample_id",
        "subject_id",
        "susceptibility_group",
        "susceptibility_id",
        "condition",
        "condition_label",
        "session_id",
        "lap_index",
        "window_index",
        "lap_group",
        "misc_score",
        "misc_3class_id",
        "misc_3class_label",
        "misc_binary_2plus",
        "misc_binary_6plus",
        "window_start_sec",
        "window_end_sec",
    }
