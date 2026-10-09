from __future__ import annotations

import argparse
from pathlib import Path

from src.eeg_pipeline.config import load_config, resolve_path
from src.eeg_pipeline.crop import crop_laps
from src.eeg_pipeline.features import extract_features
from src.eeg_pipeline.figures import make_figures
from src.eeg_pipeline.manifest import build_manifests
from src.eeg_pipeline.modeling import run_modeling
from src.eeg_pipeline.preprocess import preprocess_laps
from src.eeg_pipeline.statistics import run_statistics


def _paths(args: argparse.Namespace, config: dict) -> tuple[Path, Path]:
    base = Path(__file__).resolve().parent
    data_root = resolve_path(args.data_root or config["data_root"], base)
    output_root = resolve_path(args.output_root or config["output_root"], base)
    output_root.mkdir(parents=True, exist_ok=True)
    return data_root, output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG motion-sickness processing and analysis pipeline")
    parser.add_argument("command", choices=["scan", "crop", "preprocess", "features", "statistics", "model", "figures", "all"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    config = load_config(config_path)
    data_root, output_root = _paths(args, config)

    if args.command in {"scan", "all"}:
        build_manifests(config, data_root, output_root)
    if args.command in {"crop", "all"}:
        crop_laps(config, output_root / "manifests" / "lap_manifest.csv", output_root)
    if args.command in {"preprocess", "all"}:
        preprocess_laps(config, output_root / "manifests" / "lap_manifest_cropped.csv", output_root)
    if args.command in {"features", "all"}:
        extract_features(config, output_root / "manifests" / "lap_manifest_preprocessed.csv", output_root)
    if args.command in {"statistics", "all"}:
        run_statistics(config, output_root / "features" / "features.csv", output_root)
    if args.command in {"model", "all"}:
        run_modeling(config, output_root / "features" / "features.csv", output_root)
    if args.command in {"figures", "all"}:
        make_figures(output_root)


if __name__ == "__main__":
    main()
