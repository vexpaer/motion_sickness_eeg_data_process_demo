from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _bad_channel_count(value: Any) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return len([part for part in text.split(",") if part.strip()])


def _summary(values: pd.Series, prefix: str) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {f"{prefix}_{name}": np.nan for name in ("median", "p05", "p95", "iqr")}
    return {
        f"{prefix}_median": float(x.median()),
        f"{prefix}_p05": float(x.quantile(0.05)),
        f"{prefix}_p95": float(x.quantile(0.95)),
        f"{prefix}_iqr": float(x.quantile(0.75) - x.quantile(0.25)),
    }


def run_quality_report(config: dict[str, Any], output_root: Path) -> dict[str, pd.DataFrame]:
    manifest_path = output_root / "manifests" / "lap_manifest_preprocessed.csv"
    features_path = output_root / "features" / "features.csv"
    ica_path = output_root / "preprocess_reports" / "ica_component_metrics.csv"
    if not manifest_path.exists() or not features_path.exists():
        return {}

    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    features = pd.read_csv(features_path, dtype={"subject_id": str})
    manifest["bad_channel_count"] = manifest.get("bad_channels", pd.Series(index=manifest.index)).map(_bad_channel_count)
    manifest["usable_lap"] = manifest["status"].isin(["ok", "truncated_but_accepted"])
    manifest["rejected_lap"] = ~manifest["usable_lap"]
    manifest["ica_excluded_count"] = pd.to_numeric(manifest.get("ica_excluded", ""), errors="coerce").fillna(0).map(
        lambda value: len([x for x in str(value).split(",") if x.strip()]) if value else 0
    )

    if ica_path.exists():
        ica = pd.read_csv(ica_path)
        ica["condition_label"] = ica["sample_id"].map(
            manifest.set_index(manifest["lap_group"].astype(str).map(lambda x: f"sub-{x}"))["condition_label"].to_dict()
        )
        ica["excluded_bool"] = ica["excluded"].astype(str).str.lower().isin(["true", "1", "yes"])
        ica_summary = (
            ica.groupby("condition_label", dropna=False)
            .agg(ica_components_total=("component", "size"), ica_components_excluded=("excluded_bool", "sum"))
            .reset_index()
        )
        ica_summary["ica_exclusion_rate"] = ica_summary["ica_components_excluded"] / ica_summary["ica_components_total"]
    else:
        ica_summary = pd.DataFrame(columns=["condition_label", "ica_components_total", "ica_components_excluded", "ica_exclusion_rate"])

    rms = features.get("qc__median_rms_uV", pd.Series(dtype=float))
    max_abs = features.get("qc__max_abs_uV", pd.Series(dtype=float))
    low_gamma_cols = [c for c in features.columns if "__psd__low_gamma_rel" in c]
    if low_gamma_cols:
        spectral = features[low_gamma_cols].apply(pd.to_numeric, errors="coerce").median(axis=1)
    else:
        spectral = pd.Series(np.nan, index=features.index)

    rows: list[dict[str, Any]] = []
    expected_subjects = [str(x) for x in config.get("expected_subjects", [])]
    expected_per_condition = len(expected_subjects) * int(config.get("crop", {}).get("laps_per_session", 6))
    for condition, laps in manifest.groupby("condition_label", dropna=False):
        condition = str(condition)
        lap_count = int(len(laps))
        usable_count = int(laps["usable_lap"].sum())
        feature_mask = features["condition_label"].astype(str).eq(condition)
        row: dict[str, Any] = {
            "condition_label": condition,
            "expected_laps": expected_per_condition,
            "recorded_laps": lap_count,
            "usable_laps": usable_count,
            "rejected_laps": int(laps["rejected_lap"].sum()),
            "missing_laps": max(expected_per_condition - lap_count, 0),
            "missing_lap_rate": max(expected_per_condition - lap_count, 0) / expected_per_condition if expected_per_condition else np.nan,
            "lap_rejection_rate": float(laps["rejected_lap"].mean()) if lap_count else np.nan,
            "laps_with_bad_channels": int((laps["bad_channel_count"] > 0).sum()),
            "bad_channel_instance_total": int(laps["bad_channel_count"].sum()),
            "bad_channel_lap_rate": float((laps["bad_channel_count"] > 0).mean()) if lap_count else np.nan,
            "bad_channels_per_lap_mean": float(laps["bad_channel_count"].mean()) if lap_count else np.nan,
            "windows_total": int(feature_mask.sum()),
            "windows_rejected": 0,
            "window_rejection_rate": 0.0,
        }
        row.update(_summary(rms[feature_mask], "median_rms_uV"))
        row.update(_summary(max_abs[feature_mask], "max_abs_uV"))
        row.update(_summary(spectral[feature_mask], "median_channel_low_gamma_relative_power"))
        if not ica_summary.empty:
            match = ica_summary[ica_summary["condition_label"].astype(str).eq(condition)]
            if not match.empty:
                row.update(match.iloc[0].drop(labels=["condition_label"]).to_dict())
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("condition_label")
    out_dir = output_root / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "signal_quality_condition_summary.csv", index=False, encoding="utf-8-sig")

    feature_quality = pd.DataFrame(
        {
            "subject_id": features["subject_id"].astype(str),
            "condition_label": features["condition_label"].astype(str),
            "median_rms_uV": rms,
            "max_abs_uV": max_abs,
            "median_channel_low_gamma_relative_power": spectral,
        }
    )
    lap_quality = (
        manifest.groupby(["subject_id", "condition_label"], dropna=False)
        .agg(bad_channels_per_lap=("bad_channel_count", "mean"), ica_excluded_per_lap=("ica_excluded_count", "mean"))
        .reset_index()
    )
    feature_subject = feature_quality.groupby(["subject_id", "condition_label"], dropna=False).mean(numeric_only=True).reset_index()
    paired_base = feature_subject.merge(lap_quality, on=["subject_id", "condition_label"], how="outer")
    paired_rows: list[dict[str, Any]] = []
    metrics = [
        "median_rms_uV",
        "max_abs_uV",
        "median_channel_low_gamma_relative_power",
        "bad_channels_per_lap",
        "ica_excluded_per_lap",
    ]
    for metric in metrics:
        wide = paired_base.pivot(index="subject_id", columns="condition_label", values=metric)
        if "normal" not in wide.columns or "fast" not in wide.columns:
            continue
        paired = wide[["normal", "fast"]].dropna()
        diff = paired["fast"] - paired["normal"]
        if len(diff) > 0:
            try:
                p_value = float(wilcoxon(diff, zero_method="wilcox").pvalue) if np.any(diff != 0) else 1.0
            except Exception:
                p_value = np.nan
        else:
            p_value = np.nan
        paired_rows.append(
            {
                "metric": metric,
                "n_subjects": int(len(diff)),
                "normal_median": float(paired["normal"].median()) if len(diff) else np.nan,
                "fast_median": float(paired["fast"].median()) if len(diff) else np.nan,
                "median_fast_minus_normal": float(diff.median()) if len(diff) else np.nan,
                "wilcoxon_p": p_value,
            }
        )
    paired_tests = pd.DataFrame(paired_rows)
    paired_tests.to_csv(out_dir / "signal_quality_condition_paired_tests.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(output_root / "manifests" / "lap_manifest_preprocessed.csv", index=False, encoding="utf-8-sig")
    return {"condition_summary": summary, "paired_tests": paired_tests}
