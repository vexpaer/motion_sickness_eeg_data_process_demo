from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from eeg_pipeline import resting_baseline as rb
from eeg_pipeline.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", choices=["baselines", "models", "permutation"], default=["baselines", "models", "permutation"])
    parser.add_argument("--n-perm", type=int, default=500)
    args = parser.parse_args()

    config = load_config(ROOT / "config.yaml")
    select_k = int(config["model"]["select_k"])
    seed = int(config["model"]["random_state"])
    out_dir = ROOT / "outputs" / "revision2" / "resting_baseline"
    features = pd.read_csv(ROOT / "outputs" / "features" / "features.csv", dtype={"subject_id": str})
    window_sec = float((features["window_end_sec"] - features["window_start_sec"]).median())

    for step in args.steps:
        start = time.time()
        print(f"[{step}] running", flush=True)
        if step == "baselines":
            baselines = rb.build_resting_baselines(config, ROOT, out_dir, window_sec)
            print(pd.read_csv(out_dir / "resting_baseline_log.csv").to_string(), flush=True)
            norm = rb.normalise_features(features, baselines)
            norm.to_csv(out_dir / "features_resting_normalised.csv", index=False, encoding="utf-8-sig")
        else:
            norm = pd.read_csv(out_dir / "features_resting_normalised.csv", dtype={"subject_id": str})
            if step == "models":
                metrics = rb.run_models(norm, out_dir, select_k, seed)
                cols = ["validation", "label", "model", "balanced_accuracy", "f1_macro", "auc_macro_ovr_lap", "auc_macro_ovr_ci_lower", "auc_macro_ovr_ci_upper", "mae", "spearman_r"]
                print(metrics[[c for c in cols if c in metrics]].to_string(), flush=True)
            else:
                print(rb.permutation_test(norm, out_dir, select_k, args.n_perm, seed).to_string(), flush=True)
        print(f"[{step}] done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
