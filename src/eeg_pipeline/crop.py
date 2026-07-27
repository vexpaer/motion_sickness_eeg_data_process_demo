from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import pandas as pd

from .io import ensure_eeg_channels, export_edf, read_raw_signal


def _make_lap_raw(raw: mne.io.BaseRaw, start: int, stop: int) -> mne.io.RawArray:
    data = raw.get_data(start=start, stop=stop)
    info = mne.create_info(raw.ch_names, raw.info["sfreq"], ch_types=["eeg"] * len(raw.ch_names))
    lap_raw = mne.io.RawArray(data, info, verbose="ERROR")
    if raw.get_montage() is not None:
        lap_raw.set_montage(raw.get_montage(), on_missing="ignore", verbose="ERROR")
    return lap_raw


def crop_laps(config: dict[str, Any], lap_manifest_path: Path, output_root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(lap_manifest_path, dtype={"subject_id": str})
    cropped_root = output_root / "cropped_laps"
    current_signal_file: str | None = None
    current_raw: mne.io.BaseRaw | None = None
    cropped_files: list[str | None] = []
    crop_status: list[str] = []

    for _, row in manifest.iterrows():
        if row["status"] not in {"ok", "truncated_but_accepted"}:
            cropped_files.append(None)
            crop_status.append("skipped")
            continue

        signal_file = str(row["source_signal_file"])
        if signal_file != current_signal_file:
            current_raw = ensure_eeg_channels(read_raw_signal(signal_file, preload=True))
            current_signal_file = signal_file
        if current_raw is None:
            cropped_files.append(None)
            crop_status.append("missing_loaded_raw")
            continue
        raw = current_raw

        start = int(row["start_sample"])
        stop = int(row["actual_end_sample"])
        if stop <= start:
            cropped_files.append(None)
            crop_status.append("invalid_range")
            continue

        subject = str(row["subject_id"])
        condition = str(row["condition"])
        lap_index = int(row["lap_index"])
        output_file = cropped_root / f"sub-{subject}" / condition / f"sub-{subject}_{condition}_lap{lap_index:02d}_raw.edf"
        lap_raw = _make_lap_raw(raw, start, stop)
        export_edf(lap_raw, output_file, overwrite=True)
        cropped_files.append(str(output_file.resolve()))
        crop_status.append("written")

    updated = manifest.copy()
    updated["cropped_file"] = cropped_files
    updated["crop_status"] = crop_status
    out_path = output_root / "manifests" / "lap_manifest_cropped.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(out_path, index=False, encoding="utf-8-sig")
    return updated
