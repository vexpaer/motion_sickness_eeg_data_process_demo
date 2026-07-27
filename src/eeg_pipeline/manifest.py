from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .events import read_neuracle_events
from .io import read_raw_signal


@dataclass
class SourceInfo:
    subject_id: str
    condition: str
    condition_label: str
    source_dir: str
    segment_label: str
    source_signal_file: str | None
    source_event_file: str | None
    n_times: int | None
    sfreq: float | None
    event_count: int
    lap_offset: int
    lap_count: int
    status: str
    notes: str


def read_group_map(data_root: Path) -> dict[str, str]:
    path = data_root / "group_divide.txt"
    groups: dict[str, str] = {}
    current: str | None = None
    if not path.exists():
        return groups
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        lower = text.lower()
        if lower.startswith("susceptible"):
            current = "susceptible"
            continue
        if lower.startswith("non-susceptible"):
            current = "non_susceptible"
            continue
        if text.isdigit() and current:
            groups[text] = current
    return groups


def read_misc_scores(data_root: Path) -> dict[tuple[str, str, int], Any]:
    path = data_root / "MISC_score.xlsx"
    if not path.exists():
        return {}
    df = pd.read_excel(path)
    scores: dict[tuple[str, str, int], Any] = {}
    for _, row in df.iterrows():
        subject_id = str(int(row["sub_id"]))
        for condition in ["normal_driving", "fast_driving"]:
            for lap in range(1, 7):
                col = f"{condition}_{lap}"
                if col in row and pd.notna(row[col]):
                    scores[(subject_id, condition, lap)] = int(row[col])
    return scores


def _accepted_truncation(config: dict[str, Any], subject_id: str, condition: str, lap_index: int) -> str | None:
    for item in config["crop"].get("accept_truncated_laps", []):
        if (
            str(item.get("subject_id")) == subject_id
            and item.get("condition") == condition
            and int(item.get("lap_index")) == int(lap_index)
        ):
            return str(item.get("reason", "accepted by configuration"))
    return None


def _signal_candidates(source_dir: Path) -> list[Path]:
    names = ["data.bdf", "data.edf", "data_clean.edf"]
    return [source_dir / name for name in names if (source_dir / name).exists()]


def _choose_signal_file(source_dir: Path, max_event_sample: int | None) -> tuple[Path | None, int | None, float | None, str]:
    candidates = _signal_candidates(source_dir)
    if not candidates:
        return None, None, None, "no signal file found"

    inspected: list[tuple[Path, int, float]] = []
    for candidate in candidates:
        raw = read_raw_signal(candidate, preload=False)
        inspected.append((candidate, int(raw.n_times), float(raw.info["sfreq"])))

    if max_event_sample is not None:
        for candidate, n_times, sfreq in inspected:
            if n_times >= max_event_sample:
                return candidate, n_times, sfreq, "first candidate covering all events"

    chosen, n_times, sfreq = max(inspected, key=lambda item: item[1])
    return chosen, n_times, sfreq, "longest candidate; event range may exceed file length"


def _source_dirs(session_dir: Path, subject_id: str, condition: str) -> list[tuple[Path, str]]:
    top_has_raw = (session_dir / "evt.bdf").exists() and bool(_signal_candidates(session_dir))
    if top_has_raw:
        return [(session_dir, "session")]
    segment_dirs = sorted(p for p in session_dir.glob("data segment *") if p.is_dir())
    return [(p, p.name) for p in segment_dirs]


def build_manifests(config: dict[str, Any], data_root: Path, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_manifest_dir = output_root / "manifests"
    output_manifest_dir.mkdir(parents=True, exist_ok=True)
    group_map = read_group_map(data_root)
    misc_scores = read_misc_scores(data_root)

    source_rows: list[dict[str, Any]] = []
    lap_rows: list[dict[str, Any]] = []
    resting_rows: list[dict[str, Any]] = []
    expected_subjects = [str(x) for x in config["expected_subjects"]]
    events_per_lap = int(config["crop"]["events_per_lap"])
    driving_conditions: dict[str, str] = config["driving_conditions"]

    for subject_id in expected_subjects:
        subject_dir = data_root / subject_id
        for condition, condition_label in driving_conditions.items():
            session_dir = subject_dir / condition
            if not session_dir.exists():
                note_file = session_dir / "data_broken.txt"
                notes = note_file.read_text(encoding="utf-8").strip() if note_file.exists() else "session directory missing"
                source_rows.append(
                    asdict(
                        SourceInfo(
                            subject_id=subject_id,
                            condition=condition,
                            condition_label=condition_label,
                            source_dir=str(session_dir),
                            segment_label="missing",
                            source_signal_file=None,
                            source_event_file=None,
                            n_times=None,
                            sfreq=None,
                            event_count=0,
                            lap_offset=0,
                            lap_count=0,
                            status="missing_raw",
                            notes=notes,
                        )
                    )
                )
                continue

            source_dirs = _source_dirs(session_dir, subject_id, condition)
            if not source_dirs:
                note_file = session_dir / "data_broken.txt"
                notes = note_file.read_text(encoding="utf-8").strip() if note_file.exists() else "no readable source files"
                source_rows.append(
                    asdict(
                        SourceInfo(
                            subject_id=subject_id,
                            condition=condition,
                            condition_label=condition_label,
                            source_dir=str(session_dir),
                            segment_label="missing",
                            source_signal_file=None,
                            source_event_file=None,
                            n_times=None,
                            sfreq=None,
                            event_count=0,
                            lap_offset=0,
                            lap_count=0,
                            status="missing_raw",
                            notes=notes,
                        )
                    )
                )
                continue

            lap_offset = 0
            for source_dir, segment_label in source_dirs:
                evt_path = source_dir / "evt.bdf"
                if not evt_path.exists():
                    source_rows.append(
                        asdict(
                            SourceInfo(
                                subject_id=subject_id,
                                condition=condition,
                                condition_label=condition_label,
                                source_dir=str(source_dir),
                                segment_label=segment_label,
                                source_signal_file=None,
                                source_event_file=None,
                                n_times=None,
                                sfreq=None,
                                event_count=0,
                                lap_offset=lap_offset,
                                lap_count=0,
                                status="missing_evt",
                                notes="evt.bdf missing",
                            )
                        )
                    )
                    continue

                provisional_signal, provisional_n, provisional_sfreq, _ = _choose_signal_file(source_dir, None)
                if provisional_signal is None or provisional_sfreq is None:
                    source_rows.append(
                        asdict(
                            SourceInfo(
                                subject_id=subject_id,
                                condition=condition,
                                condition_label=condition_label,
                                source_dir=str(source_dir),
                                segment_label=segment_label,
                                source_signal_file=None,
                                source_event_file=str(evt_path),
                                n_times=None,
                                sfreq=None,
                                event_count=0,
                                lap_offset=lap_offset,
                                lap_count=0,
                                status="missing_signal",
                                notes="no signal file found",
                            )
                        )
                    )
                    continue

                events = read_neuracle_events(evt_path, provisional_sfreq)
                max_event_sample = max((ev.sample for ev in events), default=None)
                signal_path, n_times, sfreq, choose_note = _choose_signal_file(source_dir, max_event_sample)
                if sfreq != provisional_sfreq:
                    events = read_neuracle_events(evt_path, sfreq)

                lap_count = len(events) // events_per_lap
                event_remainder = len(events) % events_per_lap
                source_status = "ok" if event_remainder == 0 and lap_count > 0 else "bad_event_count"
                source_notes = choose_note
                if event_remainder:
                    source_notes += f"; event_count_not_multiple_of_{events_per_lap}"

                source_rows.append(
                    asdict(
                        SourceInfo(
                            subject_id=subject_id,
                            condition=condition,
                            condition_label=condition_label,
                            source_dir=str(source_dir),
                            segment_label=segment_label,
                            source_signal_file=str(signal_path) if signal_path else None,
                            source_event_file=str(evt_path),
                            n_times=int(n_times) if n_times is not None else None,
                            sfreq=float(sfreq) if sfreq is not None else None,
                            event_count=len(events),
                            lap_offset=lap_offset,
                            lap_count=lap_count,
                            status=source_status,
                            notes=source_notes,
                        )
                    )
                )

                for local_lap in range(1, lap_count + 1):
                    global_lap = lap_offset + local_lap
                    start_marker = events[(local_lap - 1) * events_per_lap]
                    end_marker = events[local_lap * events_per_lap - 1]
                    requested_end = int(end_marker.sample)
                    actual_end = min(requested_end, int(n_times)) if n_times is not None else requested_end
                    status = "ok"
                    notes = ""
                    if n_times is not None and requested_end > n_times:
                        accepted_reason = _accepted_truncation(config, subject_id, condition, global_lap)
                        if accepted_reason:
                            status = "truncated_but_accepted"
                            notes = accepted_reason
                        else:
                            status = "event_end_after_data_end"
                            notes = "requested end sample exceeds signal length"

                    misc_score = misc_scores.get((subject_id, condition, global_lap))
                    group = group_map.get(subject_id, "unknown")
                    lap_rows.append(
                        {
                            "subject_id": subject_id,
                            "susceptibility_group": group,
                            "susceptibility_id": 1 if group == "susceptible" else 0 if group == "non_susceptible" else None,
                            "condition": condition,
                            "condition_label": condition_label,
                            "session_id": f"{subject_id}_{condition}",
                            "segment_label": segment_label,
                            "source_dir": str(source_dir),
                            "source_signal_file": str(signal_path) if signal_path else None,
                            "source_event_file": str(evt_path),
                            "lap_index": global_lap,
                            "local_lap_index": local_lap,
                            "event_count_for_lap": events_per_lap,
                            "start_sample": int(start_marker.sample),
                            "requested_end_sample": requested_end,
                            "actual_end_sample": actual_end,
                            "sfreq": float(sfreq),
                            "n_times": int(n_times),
                            "duration_sec": (actual_end - int(start_marker.sample)) / float(sfreq),
                            "requested_duration_sec": (requested_end - int(start_marker.sample)) / float(sfreq),
                            "status": status,
                            "notes": notes,
                            "misc_score": misc_score,
                            "lap_group": f"{subject_id}_{condition}_lap{global_lap:02d}",
                        }
                    )
                lap_offset += lap_count

        for resting_condition in config.get("resting_conditions", []):
            resting_dir = subject_dir / resting_condition
            signal_path = resting_dir / "data.bdf"
            evt_path = resting_dir / "evt.bdf"
            rec_path = resting_dir / "recordInformation.json"
            status = "ok"
            n_times = None
            sfreq = None
            event_count = 0
            notes = ""
            if not resting_dir.exists():
                status = "missing_dir"
                notes = "resting session directory missing"
            elif not signal_path.exists():
                status = "missing_signal"
                notes = "data.bdf missing"
            elif not evt_path.exists():
                status = "missing_evt"
                notes = "evt.bdf missing"
            else:
                raw = read_raw_signal(signal_path, preload=False)
                n_times = int(raw.n_times)
                sfreq = float(raw.info["sfreq"])
                event_count = len(read_neuracle_events(evt_path, sfreq))
            resting_rows.append(
                {
                    "subject_id": subject_id,
                    "condition": resting_condition,
                    "source_dir": str(resting_dir),
                    "source_signal_file": str(signal_path) if signal_path.exists() else None,
                    "source_event_file": str(evt_path) if evt_path.exists() else None,
                    "record_information_file": str(rec_path) if rec_path.exists() else None,
                    "n_times": n_times,
                    "sfreq": sfreq,
                    "event_count": event_count,
                    "has_final_dir": (resting_dir / "final").exists(),
                    "status": status,
                    "notes": notes,
                }
            )

    raw_sessions = pd.DataFrame(source_rows)
    lap_manifest = pd.DataFrame(lap_rows)
    resting_sessions = pd.DataFrame(resting_rows)
    raw_sessions.to_csv(output_manifest_dir / "raw_sessions.csv", index=False, encoding="utf-8-sig")
    lap_manifest.to_csv(output_manifest_dir / "lap_manifest.csv", index=False, encoding="utf-8-sig")
    resting_sessions.to_csv(output_manifest_dir / "resting_sessions.csv", index=False, encoding="utf-8-sig")
    return raw_sessions, lap_manifest
