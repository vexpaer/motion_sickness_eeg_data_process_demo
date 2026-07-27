from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis

from .io import ensure_eeg_channels, export_edf, read_raw_signal


def _robust_z(values: np.ndarray) -> np.ndarray:
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - med) / mad


def detect_bad_channels(raw: mne.io.BaseRaw, bad_channel_z: float, flat_std_uV: float) -> list[str]:
    data_uV = raw.get_data() * 1e6
    stds = np.std(data_uV, axis=1)
    z = np.abs(_robust_z(stds))
    bads = []
    for ch, std, zval in zip(raw.ch_names, stds, z):
        if std < flat_std_uV or zval > bad_channel_z:
            bads.append(ch)
    return bads


def _safe_filter(raw: mne.io.BaseRaw, l_freq: float, h_freq: float) -> None:
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="iir", verbose="ERROR")


def _integrate(values: np.ndarray, freqs: np.ndarray) -> float:
    integrator = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if integrator is None:
        raise AttributeError("NumPy provides neither trapezoid nor trapz")
    return float(integrator(values, freqs))


def _signal_quality_metrics(raw: mne.io.BaseRaw) -> dict[str, float]:
    """Return compact lap-level metrics for the before/after ICA audit."""
    data_uV = raw.copy().pick("eeg").get_data() * 1e6
    if data_uV.size == 0:
        return {
            "rms_median_uV": np.nan,
            "ptp_median_uV": np.nan,
            "low_gamma_relative_median": np.nan,
        }
    rms = np.sqrt(np.mean(data_uV**2, axis=1))
    ptp = np.ptp(data_uV, axis=1)
    sfreq = float(raw.info["sfreq"])
    ratios = []
    for channel in data_uV:
        freqs, psd = welch(channel, fs=sfreq, nperseg=min(int(sfreq * 2), len(channel)))
        total_idx = (freqs >= 1.0) & (freqs <= 40.0)
        low_idx = (freqs >= 30.0) & (freqs <= 40.0)
        total = _integrate(psd[total_idx], freqs[total_idx]) if np.any(total_idx) else 0.0
        low = _integrate(psd[low_idx], freqs[low_idx]) if np.any(low_idx) else 0.0
        ratios.append(low / total if total > 0 else np.nan)
    return {
        "rms_median_uV": float(np.nanmedian(rms)),
        "ptp_median_uV": float(np.nanmedian(ptp)),
        "low_gamma_relative_median": float(np.nanmedian(ratios)),
    }


def _ica_component_metrics(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA, cfg: dict[str, Any]) -> pd.DataFrame:
    sources = ica.get_sources(raw).get_data()
    sfreq = float(raw.info["sfreq"])
    rows = []
    eye_channels = [ch for ch in cfg["eye_channels"] if ch in raw.ch_names]
    eye_data = raw.copy().pick(eye_channels).get_data() if eye_channels else np.empty((0, raw.n_times))
    total_low, total_high = cfg["total_band"]
    muscle_low, muscle_high = cfg["muscle_band"]

    for idx, src in enumerate(sources):
        eye_corr = 0.0
        if eye_data.size:
            corrs = []
            for eye in eye_data:
                if np.std(eye) > 0 and np.std(src) > 0:
                    corrs.append(abs(float(np.corrcoef(src, eye)[0, 1])))
            eye_corr = max(corrs) if corrs else 0.0

        freqs, psd = welch(src, fs=sfreq, nperseg=min(int(sfreq * 2), len(src)))
        total_idx = (freqs >= total_low) & (freqs <= total_high)
        muscle_idx = (freqs >= muscle_low) & (freqs <= muscle_high)
        total_power = _integrate(psd[total_idx], freqs[total_idx]) if np.any(total_idx) else 0.0
        muscle_power = _integrate(psd[muscle_idx], freqs[muscle_idx]) if np.any(muscle_idx) else 0.0
        muscle_ratio = muscle_power / total_power if total_power > 0 else 0.0
        kurt = float(abs(kurtosis(src, fisher=True, bias=False))) if len(src) > 3 else 0.0
        rows.append(
            {
                "component": idx,
                "eye_corr": eye_corr,
                "muscle_ratio": muscle_ratio,
                "abs_kurtosis": kurt,
            }
        )
    return pd.DataFrame(rows)


def run_ica(raw: mne.io.BaseRaw, cfg: dict[str, Any]) -> tuple[mne.io.BaseRaw, pd.DataFrame, list[int], str]:
    requested_components = cfg["n_components"]
    min_components = max(2, int(cfg.get("min_components", 2)))
    ica_status = "ok"

    def make_ica(n_components: int | float) -> mne.preprocessing.ICA:
        return mne.preprocessing.ICA(
            n_components=n_components,
            method=cfg["method"],
            max_iter=cfg["max_iter"],
            random_state=cfg["random_state"],
        )

    ica = make_ica(requested_components)
    try:
        ica.fit(raw, picks="eeg", verbose="ERROR")
    except (RuntimeError, ValueError) as exc:
        # With highly low-rank, rereferenced laps, the 99% target can resolve
        # to one PCA component, which MNE correctly refuses for ICA. Retry
        # with the configured minimum so ICA remains defined and auditable.
        if isinstance(requested_components, float) and "One PCA component" in str(exc):
            n_eeg = len(mne.pick_types(raw.info, eeg=True, exclude="bads"))
            fallback_components = min(n_eeg, min_components)
            if fallback_components < 2:
                raise
            ica = make_ica(fallback_components)
            ica.fit(raw, picks="eeg", verbose="ERROR")
            ica_status = f"ok_fallback_n_components_{fallback_components}"
        else:
            raise
    metrics = _ica_component_metrics(raw, ica, cfg)
    metrics["n_components_requested"] = requested_components
    metrics["n_components_used"] = int(ica.n_components_)
    exclude: list[int] = []
    reasons: list[str] = []
    for _, row in metrics.iterrows():
        reason_parts = []
        if row["eye_corr"] >= cfg["eye_corr_threshold"]:
            reason_parts.append("eye_corr")
        if row["muscle_ratio"] >= cfg["muscle_ratio_threshold"] and row["abs_kurtosis"] >= cfg["kurtosis_threshold"]:
            reason_parts.append("muscle_like")
        reasons.append(";".join(reason_parts))
        if reason_parts:
            exclude.append(int(row["component"]))
    exclude = exclude[: int(cfg["max_auto_exclude"])]
    metrics["auto_reason"] = reasons
    metrics["excluded"] = metrics["component"].isin(exclude)
    ica.exclude = exclude
    cleaned = ica.apply(raw.copy(), verbose="ERROR")
    return cleaned, metrics, exclude, ica_status


def preprocess_laps(config: dict[str, Any], cropped_manifest_path: Path, output_root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(cropped_manifest_path, dtype={"subject_id": str})
    out_root = output_root / "preprocessed_laps"
    report_root = output_root / "preprocess_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    rows = []
    reports = []
    pre_post_quality = []
    preprocess_cfg = config["preprocess"]
    ica_cfg = preprocess_cfg["ica"]

    for _, row in manifest.iterrows():
        cropped_file = row.get("cropped_file")
        if not isinstance(cropped_file, str) or not cropped_file:
            rows.append({**row.to_dict(), "preprocess_status": "skipped", "preprocessed_file": None})
            continue

        raw = ensure_eeg_channels(read_raw_signal(cropped_file, preload=True))
        montage_name = preprocess_cfg.get("montage")
        if montage_name:
            montage = mne.channels.make_standard_montage(montage_name)
            raw.set_montage(montage, on_missing="ignore", verbose="ERROR")

        # The preprocessing order is deliberate: filtering is applied before
        # bad-channel statistics/interpolation, rereferencing, resampling, and
        # per-lap ICA. No additional per-lap DC-offset subtraction is applied;
        # the 1--40 Hz band-pass removes the DC component.
        _safe_filter(raw, float(preprocess_cfg["l_freq"]), float(preprocess_cfg["h_freq"]))

        bads = detect_bad_channels(raw, float(preprocess_cfg["bad_channel_z"]), float(preprocess_cfg["flat_std_uV"]))
        raw.info["bads"] = bads
        if bads and raw.get_montage() is not None:
            raw.interpolate_bads(reset_bads=False, verbose="ERROR")

        ref_channels = [ch for ch in preprocess_cfg["reference_channels"] if ch in raw.ch_names]
        if len(ref_channels) == len(preprocess_cfg["reference_channels"]):
            raw.set_eeg_reference(ref_channels=ref_channels, projection=False, verbose="ERROR")

        target_sfreq = float(preprocess_cfg["target_sfreq"])
        if target_sfreq and raw.info["sfreq"] > target_sfreq:
            raw.resample(target_sfreq, npad="auto", verbose="ERROR")

        pre_ica_quality = _signal_quality_metrics(raw)
        ica_excluded: list[int] = []
        ica_status = "disabled"
        if ica_cfg.get("enabled", True):
            try:
                raw, component_metrics, ica_excluded, ica_status = run_ica(raw, ica_cfg)
                component_metrics.insert(0, "sample_id", f"sub-{row['subject_id']}_{row['condition']}_lap{int(row['lap_index']):02d}")
                reports.append(component_metrics)
            except Exception as exc:
                ica_status = f"failed: {exc}"

        post_ica_quality = _signal_quality_metrics(raw)
        pre_post_quality.append(
            {
                "subject_id": str(row["subject_id"]),
                "condition": str(row["condition"]),
                "lap_index": int(row["lap_index"]),
                "ica_status": ica_status,
                "ica_excluded_count": len(ica_excluded),
                "pre_ica_rms_median_uV": pre_ica_quality["rms_median_uV"],
                "post_ica_rms_median_uV": post_ica_quality["rms_median_uV"],
                "pre_ica_ptp_median_uV": pre_ica_quality["ptp_median_uV"],
                "post_ica_ptp_median_uV": post_ica_quality["ptp_median_uV"],
                "pre_ica_low_gamma_relative_median": pre_ica_quality["low_gamma_relative_median"],
                "post_ica_low_gamma_relative_median": post_ica_quality["low_gamma_relative_median"],
            }
        )

        subject = str(row["subject_id"])
        condition = str(row["condition"])
        lap_index = int(row["lap_index"])
        output_file = out_root / f"sub-{subject}" / condition / f"sub-{subject}_{condition}_lap{lap_index:02d}_preprocessed.edf"
        export_edf(raw, output_file, overwrite=True)
        rows.append(
            {
                **row.to_dict(),
                "preprocess_status": "written",
                "preprocessed_file": str(output_file.resolve()),
                "bad_channels": ",".join(bads),
                "dc_offset_correction": "not_applied",
                "reference_channels": ",".join(ref_channels),
                "ica_status": ica_status,
                "ica_excluded": ",".join(str(x) for x in ica_excluded),
                "preprocessed_sfreq": float(raw.info["sfreq"]),
                "preprocessed_n_times": int(raw.n_times),
            }
        )

    updated = pd.DataFrame(rows)
    out_path = output_root / "manifests" / "lap_manifest_preprocessed.csv"
    updated.to_csv(out_path, index=False, encoding="utf-8-sig")
    if reports:
        pd.concat(reports, ignore_index=True).to_csv(report_root / "ica_component_metrics.csv", index=False, encoding="utf-8-sig")
    if pre_post_quality:
        pd.DataFrame(pre_post_quality).to_csv(
            report_root / "ica_pre_post_quality.csv", index=False, encoding="utf-8-sig"
        )
    return updated
