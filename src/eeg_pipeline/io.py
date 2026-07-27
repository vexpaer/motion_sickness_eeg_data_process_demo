from __future__ import annotations

from pathlib import Path

import mne


def read_raw_signal(path: str | Path, preload: bool = False) -> mne.io.BaseRaw:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(str(path), preload=preload, verbose="ERROR")
    if suffix == ".edf":
        return mne.io.read_raw_edf(str(path), preload=preload, verbose="ERROR")
    raise ValueError(f"Unsupported signal file: {path}")


def ensure_eeg_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    mapping = {ch: "eeg" for ch in raw.ch_names}
    raw.set_channel_types(mapping, verbose="ERROR")
    return raw


def export_edf(raw: mne.io.BaseRaw, output_path: str | Path, overwrite: bool = True) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.export(str(output_path), fmt="edf", overwrite=overwrite, verbose="ERROR")
