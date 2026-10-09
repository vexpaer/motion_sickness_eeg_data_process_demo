from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.modules.setdefault("mne", types.ModuleType("mne"))

import pandas as pd

from eeg_pipeline import revision2 as r2
from eeg_pipeline.config import load_config

STEPS = ["composition", "loso", "uncertainty", "without", "stats", "auc", "labelonly", "permutation", "rawperm"]
SLOW_STEPS = {"permutation", "rawperm"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=[s for s in STEPS if s not in SLOW_STEPS])
    parser.add_argument("--n-perm", type=int, default=1000, help="susceptibility LOSO label permutations")
    parser.add_argument("--n-raw-perm", type=int, default=500, help="LOSO MISC 3-class within-participant label permutations")
    parser.add_argument("--n-stat-perm", type=int, default=10000, help="within-participant permutations")
    parser.add_argument("--exclude-participant", default="25017")
    args = parser.parse_args()

    config = load_config(ROOT / "config.yaml")
    model_cfg = config["model"]
    select_k = int(model_cfg["select_k"])
    seed = int(model_cfg["random_state"])
    outputs = ROOT / "outputs"
    out_dir = outputs / "revision2"
    df = pd.read_csv(outputs / "features" / "features.csv", dtype={"subject_id": str})

    for step in args.steps:
        start = time.time()
        print(f"[{step}] running", flush=True)
        if step == "composition":
            print(r2.misc_stratum_composition(df, out_dir)[0].to_string())
        elif step == "loso":
            print(r2.run_loso(df, out_dir, select_k, seed)[["label", "model", "f1_macro", "auc_macro_ovr", "auc_macro_ovr_lap"]].to_string())
        elif step == "uncertainty":
            print(r2.lolo_cluster_uncertainty(outputs / "models", out_dir, seed).to_string())
        elif step == "without":
            print(r2.lolo_without_participant(df, out_dir, args.exclude_participant, select_k, seed)[["label", "model", "f1_macro", "auc_macro_ovr", "auc_macro_ovr_lap"]].to_string())
        elif step == "stats":
            print(r2.participant_aware_statistics(df, out_dir, args.n_stat_perm, seed, args.exclude_participant).to_string())
        elif step == "auc":
            print(r2.auc_aggregation_table(outputs / "models", out_dir / "models_loso", out_dir).to_string())
        elif step == "labelonly":
            print(r2.label_only_baselines(df, outputs / "models", out_dir).round(3).to_string())
        elif step == "rawperm":
            from eeg_pipeline import resting_baseline as rb

            raw_dir = out_dir / "models_loso" / "raw_feature_permutation"
            raw_dir.mkdir(parents=True, exist_ok=True)
            features = df[[c for c in df.columns if not c.startswith("qc__")]]
            print(rb.permutation_test(features, raw_dir, select_k, args.n_raw_perm, seed).to_string())
        elif step == "permutation":
            print(r2.susceptibility_permutation_test(df, out_dir, select_k, args.n_perm, seed).to_string())
        print(f"[{step}] done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
