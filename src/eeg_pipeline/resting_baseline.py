from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from .crop import _make_lap_raw
from .events import read_neuracle_events
from .features import _extract_window_features, metadata_columns
from .io import ensure_eeg_channels, export_edf, read_raw_signal
from .modeling import _feature_columns, _make_pipeline, evaluate_label, evaluate_ordinal_misc
from .preprocess import _safe_filter, detect_bad_channels, run_ica
from .statistics import _parse_channel_feature

MISC_TASKS = ["misc_3class_id", "misc_binary_2plus", "misc_binary_6plus"]


def _preprocess_like_laps(raw: mne.io.BaseRaw, config: dict[str, Any]) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    cfg = config["preprocess"]
    montage_name = cfg.get("montage")
    if montage_name:
        raw.set_montage(mne.channels.make_standard_montage(montage_name), on_missing="ignore", verbose="ERROR")
    _safe_filter(raw, float(cfg["l_freq"]), float(cfg["h_freq"]))
    bads = detect_bad_channels(raw, float(cfg["bad_channel_z"]), float(cfg["flat_std_uV"]))
    raw.info["bads"] = bads
    if bads and raw.get_montage() is not None:
        raw.interpolate_bads(reset_bads=False, verbose="ERROR")
    ref_channels = [ch for ch in cfg["reference_channels"] if ch in raw.ch_names]
    if len(ref_channels) == len(cfg["reference_channels"]):
        raw.set_eeg_reference(ref_channels=ref_channels, projection=False, verbose="ERROR")
    target_sfreq = float(cfg["target_sfreq"])
    if target_sfreq and raw.info["sfreq"] > target_sfreq:
        raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    ica_status, excluded = "disabled", []
    if cfg["ica"].get("enabled", True):
        try:
            raw, _, excluded, ica_status = run_ica(raw, cfg["ica"])
        except Exception as exc:
            ica_status = f"failed: {exc}"
    return raw, {
        "bad_channels": ",".join(bads),
        "reference_channels": ",".join(ref_channels),
        "ica_status": ica_status,
        "ica_excluded_count": len(excluded),
    }


def build_resting_baselines(config: dict[str, Any], workspace: Path, out_dir: Path, window_sec: float) -> pd.DataFrame:
    data_root = (workspace / config["data_root"]).resolve()
    exclude_channels = set(config["features"].get("exclude_channels", []))
    bands = config["features"]["bands"]
    target_sfreq = float(config["preprocess"]["target_sfreq"])
    window_samples = int(round(window_sec * target_sfreq))
    rows, log_rows = [], []
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for condition in ("normal_driving", "fast_driving"):
            rest_dir = subject_dir / f"resting_before_{condition}"
            signal, evt = rest_dir / "data.bdf", rest_dir / "evt.bdf"
            log = {"subject_id": subject_dir.name, "condition": condition, "resting_dir": rest_dir.name}
            if not signal.exists():
                log_rows.append({**log, "status": "missing_resting_signal"})
                continue
            raw = ensure_eeg_channels(read_raw_signal(signal, preload=True))
            sfreq = float(raw.info["sfreq"])
            markers = read_neuracle_events(evt, sfreq) if evt.exists() else []
            if len(markers) >= 2:
                start, stop = min(m.sample for m in markers), min(max(m.sample for m in markers), raw.n_times)
                segment = "first_to_last_marker"
            else:
                start, stop, segment = 0, raw.n_times, "whole_recording"
            cropped_file = out_dir / "cropped" / f"sub-{subject_dir.name}_resting_before_{condition}_raw.edf"
            export_edf(_make_lap_raw(raw, start, stop), cropped_file, overwrite=True)
            rest = ensure_eeg_channels(read_raw_signal(cropped_file, preload=True))
            rest, info = _preprocess_like_laps(rest, config)
            rest.pick([ch for ch in rest.ch_names if ch not in exclude_channels])
            data = rest.get_data()
            n_windows = rest.n_times // window_samples
            windows = [
                _extract_window_features(data[:, i * window_samples : (i + 1) * window_samples], list(rest.ch_names), float(rest.info["sfreq"]), bands)
                for i in range(n_windows)
            ]
            frame = pd.DataFrame(windows)
            baseline = frame[[c for c in frame.columns if not c.startswith("qc__")]].median()
            rows.append({"subject_id": subject_dir.name, "condition": condition, **baseline.to_dict()})
            log_rows.append(
                {
                    **log,
                    "status": "ok",
                    "segment": segment,
                    "n_markers": len(markers),
                    "segment_sec": (stop - start) / sfreq,
                    "window_samples": window_samples,
                    "n_windows": n_windows,
                    **info,
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(out_dir / "resting_baseline_log.csv", index=False, encoding="utf-8-sig")
    baselines = pd.DataFrame(rows)
    baselines.to_csv(out_dir / "resting_baselines.csv", index=False, encoding="utf-8-sig")
    return baselines


def _log_ratio(feature: str) -> bool:
    info = _parse_channel_feature(feature)
    return info["feature_family"] == "time" or info["measure"].startswith("activity") or info["measure"] == "absolute_power"


def normalise_features(features: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in _feature_columns(features)]
    keys = features[["subject_id", "condition"]].astype(str)
    base = keys.merge(baselines.astype({"subject_id": str}), on=["subject_id", "condition"], how="left")
    missing = base[feature_cols].isna().all(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} windows have no resting baseline")
    out = features.copy()
    for col in feature_cols:
        x = features[col].to_numpy(dtype=float)
        b = base[col].to_numpy(dtype=float)
        if _log_ratio(col):
            with np.errstate(divide="ignore", invalid="ignore"):
                out[col] = np.where((x > 0) & (b > 0), np.log10(x) - np.log10(b), np.nan)
        else:
            out[col] = x - b
    meta = [c for c in features.columns if c in metadata_columns()]
    return out[meta + feature_cols]


def _loso_lap_macro_auc(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, laps: np.ndarray, select_k: int) -> float:
    classes = np.unique(y)
    prob = np.zeros((len(y), len(classes)))
    for subject in np.unique(subjects):
        test = subjects == subject
        pipe = _make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced"), select_k, X.shape[1])
        pipe.fit(X[~test], y[~test])
        fold_prob = pipe.predict_proba(X[test])
        for j, cls in enumerate(pipe.named_steps["model"].classes_):
            prob[test, int(np.flatnonzero(classes == cls)[0])] = fold_prob[:, j]
    frame = pd.DataFrame(prob, columns=classes).assign(lap=laps, y=y).groupby("lap").mean()
    return float(roc_auc_score(label_binarize(frame["y"].astype(int), classes=classes), frame[classes].to_numpy(), average="macro", multi_class="ovr"))


def permutation_test(norm: pd.DataFrame, out_dir: Path, select_k: int, n_perm: int, random_state: int) -> pd.DataFrame:
    X = norm[_feature_columns(norm)].to_numpy(dtype=float)
    y = norm["misc_3class_id"].to_numpy(dtype=int)
    subjects = norm["subject_id"].astype(str).to_numpy()
    laps = norm["lap_group"].astype(str).to_numpy()
    observed = _loso_lap_macro_auc(X, y, subjects, laps, select_k)
    lap_table = norm.groupby("lap_group").agg(subject=("subject_id", "first"), y=("misc_3class_id", "first"))
    rng = np.random.default_rng(random_state)
    null = []
    for _ in range(n_perm):
        shuffled = lap_table["y"].copy()
        for _, idx in lap_table.groupby("subject").groups.items():
            shuffled.loc[idx] = rng.permutation(lap_table.loc[idx, "y"].to_numpy())
        null.append(_loso_lap_macro_auc(X, norm["lap_group"].map(shuffled).to_numpy(dtype=int), subjects, laps, select_k))
    null_arr = np.asarray(null)
    pd.DataFrame({"null_lap_macro_auc": null_arr}).to_csv(out_dir / "permutation_null.csv", index=False, encoding="utf-8-sig")
    result = pd.DataFrame(
        [
            {
                "model": "LogisticRegression",
                "statistic": "LOSO MISC 3-class lap-averaged macro-AUC",
                "observed": observed,
                "n_permutations": n_perm,
                "null_mean": float(null_arr.mean()),
                "null_95th": float(np.quantile(null_arr, 0.95)),
                "p_permutation": float((np.sum(null_arr >= observed) + 1) / (n_perm + 1)),
            }
        ]
    )
    result.to_csv(out_dir / "permutation_test.csv", index=False, encoding="utf-8-sig")
    return result


def run_models(norm: pd.DataFrame, out_dir: Path, select_k: int, random_state: int) -> pd.DataFrame:
    frames = []
    for group_level, sub in [("subject", "models_loso"), ("lap", "models_lolo")]:
        for label in MISC_TASKS:
            frames.append(evaluate_label(norm, label, out_dir / sub, group_level, select_k, random_state).assign(validation=group_level))
        frames.append(evaluate_ordinal_misc(norm, out_dir / sub, group_level, select_k).assign(validation=group_level))
    result = pd.concat(frames, ignore_index=True, sort=False)
    result.to_csv(out_dir / "metrics_resting_normalised.csv", index=False, encoding="utf-8-sig")
    return result
