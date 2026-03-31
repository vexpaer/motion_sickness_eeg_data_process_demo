import argparse
import itertools
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import hilbert
from tqdm import tqdm

# Purpose:
# Unified EEG feature extraction demo for files in subject folder 25009 under the current directory.
# Supports time-domain, Hjorth, PSD, DE, PLV, and WPLI features with optional interactive selection.


BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 40),
}

FEATURE_OPTIONS = {
    "1": "time",      # variance / RMS
    "2": "hjorth",
    "3": "psd",
    "4": "de",
    "5": "plv",
    "6": "wpli",
}

VALID_FILE_PATTERN = re.compile(r"^(YG|FYG)_(JS|ZC)_\d+_final\d+\.(edf|bdf)$", re.IGNORECASE)


def assign_brain_region(ch_name: str) -> str:
    name = ch_name.upper()
    if name.startswith(("FP", "F", "AF", "FT")):
        return "frontal"
    if name.startswith(("FC", "C", "CZ", "CP")):
        return "central"
    if name.startswith(("P", "PO", "PZ")):
        return "parietal"
    if name.startswith(("O", "OZ")):
        return "occipital"
    return "unknown"


def build_regions(ch_names: Sequence[str]) -> Dict[str, List[int]]:
    regions = {
        "frontal": [],
        "central": [],
        "parietal": [],
        "occipital": [],
        "whole": list(range(len(ch_names))),
    }
    for idx, ch in enumerate(ch_names):
        region = assign_brain_region(ch)
        if region in regions:
            regions[region].append(idx)
    return regions


def zscore_channel_feature(values: Dict[str, float]) -> Dict[str, float]:
    arr = np.array(list(values.values()), dtype=np.float64)
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    if std_val == 0:
        return {k: 0.0 for k in values}
    return {k: (v - mean_val) / std_val for k, v in values.items()}


def region_mean_from_channel_dict(ch_values: Dict[str, float]) -> Dict[str, float]:
    channels = list(ch_values.keys())
    regions = {
        "frontal": [ch for ch in channels if assign_brain_region(ch) == "frontal"],
        "central": [ch for ch in channels if assign_brain_region(ch) == "central"],
        "parietal": [ch for ch in channels if assign_brain_region(ch) == "parietal"],
        "occipital": [ch for ch in channels if assign_brain_region(ch) == "occipital"],
        "whole": channels,
    }
    out = {}
    for region, chs in regions.items():
        out[region] = float(np.mean([ch_values[ch] for ch in chs])) if chs else np.nan
    return out


def bandpass_array(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    nyq = sfreq / 2
    low = max(fmin / nyq, 1e-5)
    high = min(fmax / nyq, 0.999)
    b, a = signal.butter(4, [low, high], btype="band")
    return signal.filtfilt(b, a, data, axis=1)


def calc_time_features(seg_data: np.ndarray, ch_names: Sequence[str]) -> Dict[str, float]:
    variance = {ch: float(np.var(seg_data[i])) for i, ch in enumerate(ch_names)}
    rms = {ch: float(np.sqrt(np.mean(seg_data[i] ** 2))) for i, ch in enumerate(ch_names)}

    variance_norm = zscore_channel_feature(variance)
    rms_norm = zscore_channel_feature(rms)

    out = {}
    for ch in ch_names:
        out[f"{ch}_variance_norm"] = variance_norm[ch]
        out[f"{ch}_rms_norm"] = rms_norm[ch]

    variance_region = region_mean_from_channel_dict(variance_norm)
    rms_region = region_mean_from_channel_dict(rms_norm)
    for region in ["frontal", "central", "parietal", "occipital", "whole"]:
        out[f"{region}_variance"] = variance_region[region]
        out[f"{region}_rms"] = rms_region[region]
    return out


def calc_hjorth_features(seg_data: np.ndarray, ch_names: Sequence[str]) -> Dict[str, float]:
    activity = {}
    mobility = {}
    complexity = {}

    for i, ch in enumerate(ch_names):
        sig = seg_data[i]
        d1 = np.gradient(sig)
        d2 = np.gradient(d1)

        var_sig = np.var(sig)
        var_d1 = np.var(d1)
        var_d2 = np.var(d2)

        act = float(var_sig)
        mob = float(np.sqrt(var_d1 / var_sig)) if var_sig > 0 else 0.0
        comp = float(np.sqrt(var_d2 / var_d1) / mob) if var_d1 > 0 and mob > 0 else 0.0

        activity[ch] = act
        mobility[ch] = mob
        complexity[ch] = comp

    activity_norm = zscore_channel_feature(activity)
    mobility_norm = zscore_channel_feature(mobility)
    complexity_norm = zscore_channel_feature(complexity)

    out = {}
    for ch in ch_names:
        out[f"{ch}_activity_norm"] = activity_norm[ch]
        out[f"{ch}_mobility_norm"] = mobility_norm[ch]
        out[f"{ch}_complexity_norm"] = complexity_norm[ch]

    for feature_name, values in [
        ("activity", activity_norm),
        ("mobility", mobility_norm),
        ("complexity", complexity_norm),
    ]:
        region_values = region_mean_from_channel_dict(values)
        for region in ["frontal", "central", "parietal", "occipital", "whole"]:
            out[f"{region}_{feature_name}_norm"] = region_values[region]
    return out


def calc_psd_features(seg_data: np.ndarray, ch_names: Sequence[str], sfreq: float) -> Dict[str, float]:
    ch_band_values = {ch: {} for ch in ch_names}

    for i, ch in enumerate(ch_names):
        freqs, psd = signal.welch(seg_data[i], fs=sfreq, window="hann", nperseg=min(1024, seg_data.shape[1]))
        for band_name, (fmin, fmax) in BANDS.items():
            idx = (freqs >= fmin) & (freqs <= fmax)
            ch_band_values[ch][band_name] = float(np.mean(psd[idx])) if np.any(idx) else 0.0

    out = {}
    for band_name in BANDS:
        raw_band = {ch: ch_band_values[ch][band_name] for ch in ch_names}
        norm_band = zscore_channel_feature(raw_band)
        for ch in ch_names:
            out[f"{ch}_{band_name}"] = norm_band[ch]
        region_values = region_mean_from_channel_dict(norm_band)
        for region in ["frontal", "central", "parietal", "occipital", "whole"]:
            out[f"{region}_{band_name}"] = region_values[region]
    return out


def calc_de_features(seg_data: np.ndarray, ch_names: Sequence[str], sfreq: float) -> Dict[str, float]:
    de_values = {}
    for i, ch in enumerate(ch_names):
        freqs, psd = signal.welch(seg_data[i], fs=sfreq, window="hann", nperseg=min(1024, seg_data.shape[1]))
        idx = (freqs >= 0.5) & (freqs <= 45)
        power = float(np.mean(psd[idx])) if np.any(idx) else 0.0
        de_values[ch] = float(0.5 * np.log(2 * np.pi * np.e * power)) if power > 0 else 0.0

    de_norm = zscore_channel_feature(de_values)
    out = {}
    for ch in ch_names:
        out[f"{ch}_de_norm"] = de_norm[ch]

    region_values = region_mean_from_channel_dict(de_norm)
    for region in ["frontal", "central", "parietal", "occipital", "whole"]:
        out[f"{region}_de"] = region_values[region]
    return out


def calc_plv_matrix(filtered_data: np.ndarray) -> np.ndarray:
    n_channels = filtered_data.shape[0]
    phases = np.angle(hilbert(filtered_data, axis=1))
    matrix = np.zeros((n_channels, n_channels), dtype=np.float64)
    for i in range(n_channels):
        for j in range(i, n_channels):
            phase_diff = phases[i] - phases[j]
            val = np.abs(np.mean(np.exp(1j * phase_diff)))
            matrix[i, j] = val
            matrix[j, i] = val
    return matrix


def calc_wpli_matrix(filtered_data: np.ndarray) -> np.ndarray:
    n_channels = filtered_data.shape[0]
    phases = np.angle(hilbert(filtered_data, axis=1))
    matrix = np.zeros((n_channels, n_channels), dtype=np.float64)
    for i in range(n_channels):
        for j in range(i, n_channels):
            phase_diff = phases[i] - phases[j]
            weights = np.abs(np.sin(phase_diff))
            num = np.abs(np.sum(weights * np.exp(1j * phase_diff)))
            den = np.sum(weights)
            val = float(num / den) if den > 0 else 0.0
            matrix[i, j] = val
            matrix[j, i] = val
    return matrix


def matrix_to_connectivity_features(
    matrix: np.ndarray,
    ch_names: Sequence[str],
    prefix: str,
    regions: Dict[str, List[int]],
) -> Dict[str, float]:
    out = {}

    for i, j in itertools.combinations(range(len(ch_names)), 2):
        out[f"{prefix}_{ch_names[i]}_{ch_names[j]}"] = float(matrix[i, j])

    region_names = ["frontal", "central", "parietal", "occipital"]
    for region in region_names:
        idxs = regions[region]
        if len(idxs) > 1:
            sub = matrix[np.ix_(idxs, idxs)]
            upper = sub[np.triu_indices(len(idxs), k=1)]
            out[f"{prefix}_region_{region}"] = float(np.mean(upper)) if upper.size else np.nan

    for reg1, reg2 in itertools.combinations(region_names, 2):
        idx1, idx2 = regions[reg1], regions[reg2]
        if idx1 and idx2:
            inter = matrix[np.ix_(idx1, idx2)]
            out[f"{prefix}_region_{reg1}-{reg2}"] = float(np.mean(inter)) if inter.size else np.nan

    return out


def calc_plv_features(seg_data: np.ndarray, ch_names: Sequence[str], sfreq: float) -> Dict[str, float]:
    out = {}
    regions = build_regions(ch_names)
    for band_name, (fmin, fmax) in BANDS.items():
        filtered = bandpass_array(seg_data, sfreq, fmin, fmax)
        matrix = calc_plv_matrix(filtered)
        out.update(matrix_to_connectivity_features(matrix, ch_names, f"plv_{band_name}", regions))
    return out


def calc_wpli_features(seg_data: np.ndarray, ch_names: Sequence[str], sfreq: float) -> Dict[str, float]:
    out = {}
    regions = build_regions(ch_names)
    for band_name, (fmin, fmax) in BANDS.items():
        filtered = bandpass_array(seg_data, sfreq, fmin, fmax)
        matrix = calc_wpli_matrix(filtered)
        out.update(matrix_to_connectivity_features(matrix, ch_names, f"wpli_{band_name}", regions))
    return out


def split_to_segments(data: np.ndarray, segments: int) -> List[np.ndarray]:
    total = data.shape[1]
    if total < segments:
        return [data]
    boundaries = np.linspace(0, total, segments + 1, dtype=int)
    return [data[:, boundaries[i]:boundaries[i + 1]] for i in range(segments)]


def parse_selection_indices(text: str, max_index: int) -> List[int]:
    text = text.strip().lower()
    if text == "all":
        return list(range(max_index))

    selected = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            for idx in range(int(start), int(end) + 1):
                if 1 <= idx <= max_index:
                    selected.add(idx - 1)
        else:
            idx = int(part)
            if 1 <= idx <= max_index:
                selected.add(idx - 1)
    return sorted(selected)


def choose_features_interactively(default_features: List[str]) -> List[str]:
    print("\n可选特征:")
    print("1) 时域(方差+均方根)")
    print("2) Hjorth")
    print("3) PSD")
    print("4) 微分熵(DE)")
    print("5) PLV")
    print("6) WPLI")
    text = input(f"选择要提取的特征(如1,3,6 或 all，默认 {','.join(default_features)}): ").strip()
    if not text:
        return default_features
    if text.lower() == "all":
        return list(FEATURE_OPTIONS.values())

    picked = []
    for key in [x.strip() for x in text.split(",") if x.strip()]:
        if key in FEATURE_OPTIONS:
            picked.append(FEATURE_OPTIONS[key])
    return picked or default_features


def choose_files_interactively(candidates: List[Path]) -> List[Path]:
    print("\n可选文件(仅展示前30个):")
    for i, p in enumerate(candidates[:30], start=1):
        print(f"{i}) {p.name}")
    if len(candidates) > 30:
        print(f"... 共 {len(candidates)} 个文件")

    text = input("选择文件序号(如1,3-8 或 all，默认all): ").strip()
    if not text or text.lower() == "all":
        return candidates

    idxs = parse_selection_indices(text, len(candidates))
    return [candidates[i] for i in idxs] if idxs else candidates


def discover_data_files(data_dir: Path) -> List[Path]:
    files = []
    for p in sorted(data_dir.iterdir()):
        if p.is_file() and VALID_FILE_PATTERN.match(p.name):
            files.append(p)
    return files


def extract_features_from_segment(
    seg_data: np.ndarray,
    ch_names: Sequence[str],
    sfreq: float,
    selected_features: Sequence[str],
) -> Dict[str, float]:
    features = {}
    if "time" in selected_features:
        features.update(calc_time_features(seg_data, ch_names))
    if "hjorth" in selected_features:
        features.update(calc_hjorth_features(seg_data, ch_names))
    if "psd" in selected_features:
        features.update(calc_psd_features(seg_data, ch_names, sfreq))
    if "de" in selected_features:
        features.update(calc_de_features(seg_data, ch_names, sfreq))
    if "plv" in selected_features:
        features.update(calc_plv_features(seg_data, ch_names, sfreq))
    if "wpli" in selected_features:
        features.update(calc_wpli_features(seg_data, ch_names, sfreq))
    return features


def read_raw_file(file_path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)


def run_extraction(
    files: Sequence[Path],
    output_csv: Path,
    selected_features: Sequence[str],
    selected_channels: Sequence[str],
    segments: int,
    lfreq: float,
    hfreq: float,
    target_sfreq: float,
) -> None:
    rows = []
    total_steps = len(files) * max(segments, 1)

    with tqdm(total=total_steps, desc="特征提取进度", unit="segment") as pbar:
        for file_path in files:
            try:
                raw = read_raw_file(file_path)
                if target_sfreq and raw.info["sfreq"] > target_sfreq:
                    raw.resample(target_sfreq, verbose=False)
                raw.filter(lfreq, hfreq, method="fir", verbose=False)

                if selected_channels:
                    missing = [ch for ch in selected_channels if ch not in raw.ch_names]
                    if missing:
                        raise ValueError(f"文件 {file_path.name} 缺少通道: {missing}")
                    raw.pick(selected_channels)

                ch_names = raw.ch_names
                sfreq = float(raw.info["sfreq"])
                data = raw.get_data().astype(np.float64)

                segments_data = split_to_segments(data, segments)
                for seg_idx, seg_data in enumerate(segments_data, start=1):
                    if seg_data.shape[1] < 8:
                        pbar.update(1)
                        continue
                    row = {
                        "sample_id": f"{file_path.stem}.{seg_idx}",
                    }
                    row.update(extract_features_from_segment(seg_data, ch_names, sfreq, selected_features))
                    rows.append(row)
                    pbar.update(1)

            except Exception as exc:
                print(f"\n[跳过] {file_path.name}: {exc}")
                pbar.update(segments)

    if not rows:
        raise RuntimeError("没有成功提取到任何特征，请检查文件与参数设置。")

    df = pd.DataFrame(rows)
    ordered_cols = ["sample_id"] + sorted([c for c in df.columns if c != "sample_id"])
    df = df[ordered_cols]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n提取完成，共 {len(df)} 行，{len(df.columns) - 1} 个特征")
    print(f"输出文件: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整合版EEG特征提取（支持EDF/BDF、6段切分、交互选择）")
    parser.add_argument("--data-dir", default="25009", help="Data directory (default: 25009 in current directory)")
    parser.add_argument("--output", default="features_25009.csv", help="Output CSV path")
    parser.add_argument("--features", default="", help="特征选择，逗号分隔: time,hjorth,psd,de,plv,wpli")
    parser.add_argument("--files", default="", help="指定文件名（逗号分隔，不带路径）")
    parser.add_argument("--channels", default="", help="指定通道名（逗号分隔），默认全部通道")
    parser.add_argument("--segments", type=int, default=6, help="每个文件切分段数（默认6）")
    parser.add_argument("--lfreq", type=float, default=0.5, help="带通低频")
    parser.add_argument("--hfreq", type=float, default=45.0, help="带通高频")
    parser.add_argument("--target-sfreq", type=float, default=250.0, help="目标重采样率(仅在原采样率更高时生效)")
    parser.add_argument("--no-interactive", action="store_true", help="关闭交互模式")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path.cwd()
    data_dir = workspace / args.data_dir
    output_csv = workspace / args.output

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    candidates = discover_data_files(data_dir)
    if not candidates:
        raise RuntimeError("在命名数据目录下未发现符合命名规则的EDF/BDF文件")

    selected_features = [x.strip() for x in args.features.split(",") if x.strip()]
    if not selected_features:
        selected_features = ["time", "hjorth", "psd", "de", "plv", "wpli"]

    files = candidates
    if args.files:
        allow = {x.strip() for x in args.files.split(",") if x.strip()}
        files = [p for p in candidates if p.name in allow]

    if not args.no_interactive:
        files = choose_files_interactively(files)
        selected_features = choose_features_interactively(selected_features)

    selected_channels = [x.strip() for x in args.channels.split(",") if x.strip()]

    print("\n=== 本次提取配置 ===")
    print(f"文件数: {len(files)}")
    print(f"特征: {selected_features}")
    print(f"切分段数: {args.segments}")
    print(f"滤波范围: {args.lfreq}-{args.hfreq} Hz")
    print(f"重采样目标: {args.target_sfreq} Hz")
    print(f"输出: {output_csv}")

    run_extraction(
        files=files,
        output_csv=output_csv,
        selected_features=selected_features,
        selected_channels=selected_channels,
        segments=max(1, args.segments),
        lfreq=args.lfreq,
        hfreq=args.hfreq,
        target_sfreq=args.target_sfreq,
    )


if __name__ == "__main__":
    main()
