from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np


@dataclass(frozen=True)
class EventMarker:
    sample: int
    onset_sec: float
    duration_sec: float
    code: int


def _read_tal_data(evt_path: Path) -> np.ndarray:
    raw_evt = mne.io.read_raw_bdf(str(evt_path), preload=False, verbose="ERROR")
    try:
        tal_data = raw_evt._read_segment_file([], [], 0, 0, int(raw_evt.n_times), None, None)
    except Exception:
        idx = np.empty(0, int)
        tal_data = raw_evt._read_segment_file(
            np.empty((0, raw_evt.n_times)),
            idx,
            0,
            0,
            int(raw_evt.n_times),
            np.ones((len(idx), 1)),
            None,
        )
    return tal_data[0]


def _decode_tal_bytes(tal_channels: np.ndarray) -> str:
    tals = bytearray()
    for chan in tal_channels:
        this_chan = chan.ravel()
        if this_chan.dtype == np.int32:
            this_chan.dtype = np.uint8
            this_chan = this_chan.reshape(-1, 4)[:, :3].ravel()
            tals.extend(this_chan)
        else:
            for value in this_chan:
                i = int(value)
                tals.extend(np.uint8([i % 256, i // 256]))
    return tals.decode("latin-1")


def read_neuracle_events(evt_path: str | Path, sfreq: float) -> list[EventMarker]:
    evt_path = Path(evt_path)
    tal_text = _decode_tal_bytes(_read_tal_data(evt_path))
    pattern = r"([+-]\d+\.?\d*)(\x15(\d+\.?\d*))?(\x14.*?)\x14\x00"
    markers: list[EventMarker] = []
    for match in re.findall(pattern, tal_text):
        onset = float(match[0])
        duration = float(match[2]) if match[2] else 0.0
        for desc in match[3].split("\x14")[1:]:
            if desc and desc.isdigit():
                markers.append(
                    EventMarker(
                        sample=int(round(onset * sfreq)),
                        onset_sec=onset,
                        duration_sec=duration,
                        code=int(desc),
                    )
                )
    return markers
