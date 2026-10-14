from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _add_box(ax, xy, width, height, title, lines, facecolor, edgecolor="#36536b"):
    x, y = xy
    rect = plt.Rectangle((x, y), width, height, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + 0.02, y + height - 0.06, title, fontsize=12, fontweight="bold", va="top", color="#102a3a")
    ax.text(
        x + 0.02,
        y + height - 0.13,
        "\n".join(lines),
        fontsize=9.5,
        va="top",
        color="#1f2933",
        linespacing=1.25,
    )


def _make_data_record_figure(fig_dir: Path) -> None:
    tree = r"""EEG_data/
├── README.txt
├── dataset_manifest.csv
├── data_dictionary.csv
├── group_divide.txt
├── MISC_score.xlsx
├── MSSQ_score.xlsx
├── 25009/                         # representative participant
│   ├── normal_driving/
│   │   ├── data.bdf               # raw continuous EEG
│   │   ├── evt.bdf                # synchronized event markers
│   │   ├── recordInformation.json # acquisition metadata
│   │   └── final/
│   │       ├── final1.edf
│   │       ├── final2.edf
│   │       ├── final3.edf
│   │       ├── final4.edf
│   │       ├── final5.edf
│   │       └── final6.edf
│   ├── fast_driving/
│   │   ├── data.bdf
│   │   ├── evt.bdf
│   │   ├── recordInformation.json
│   │   └── final/
│   │       ├── final1.edf
│   │       ├── ...
│   │       └── final6.edf
│   ├── resting_before_normal_driving/
│   │   ├── data.bdf
│   │   ├── evt.bdf
│   │   └── recordInformation.json
│   ├── resting_after_normal_driving/
│   │   ├── data.bdf
│   │   ├── evt.bdf
│   │   └── recordInformation.json
│   ├── resting_before_fast_driving/
│   │   ├── data.bdf
│   │   ├── evt.bdf
│   │   └── recordInformation.json
│   └── resting_after_fast_driving/
│       ├── data.bdf
│       ├── evt.bdf
│       └── recordInformation.json
├── 25010/ ... 25023/              # same nominal structure
├── 25013/
│   └── normal_driving/
│       └── data_broken.txt        # no valid raw EEG; missing_raw
└── 25019/
    └── fast_driving/
        ├── note.txt
        ├── data segment 1/        # laps 1–4
        │   ├── data.bdf
        │   ├── evt.bdf
        │   └── recordInformation.json
        ├── data segment 2/        # laps 5–6
        │   ├── data.bdf
        │   ├── evt.bdf
        │   └── recordInformation.json
        └── final/
            ├── final1.edf
            ├── ...
            └── final6.edf"""
    fig, ax = plt.subplots(figsize=(8.5, 14))
    ax.set_axis_off()
    ax.text(
        0.008,
        0.995,
        tree,
        va="top",
        ha="left",
        fontsize=9.5,
        family="DejaVu Sans Mono",
        color="#18232d",
        linespacing=1.18,
    )
    fig.subplots_adjust(left=0.008, right=0.995, top=0.998, bottom=0.005)
    fig.savefig(fig_dir / "data_record.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_figures(output_root: Path) -> None:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _make_data_record_figure(fig_dir)

    lap_manifest = output_root / "manifests" / "lap_manifest_preprocessed.csv"
    if lap_manifest.exists():
        laps = pd.read_csv(lap_manifest, dtype={"subject_id": str})
        usable = laps[laps["status"].isin(["ok", "truncated_but_accepted"])]
        fig, ax = plt.subplots(figsize=(6, 4))
        usable.boxplot(column="duration_sec", by="condition_label", ax=ax)
        ax.set_title("Lap duration by condition")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Duration (s)")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(fig_dir / "lap_duration_by_condition.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        counts = usable.groupby(["condition_label", "misc_score"]).size().unstack(fill_value=0).T
        counts = counts.rename(columns={"normal": "Normal driving", "fast": "Fast driving"})
        counts = counts.reindex(columns=["Normal driving", "Fast driving"], fill_value=0)
        counts.plot(kind="bar", ax=ax, rot=0, color=["#4C78A8", "#F58518"])
        ax.legend(title="Condition", frameon=False)
        ax.set_title("Lap-level MISC distribution")
        ax.set_xlabel("MISC score")
        ax.set_ylabel("Lap count")
        fig.tight_layout()
        fig.savefig(fig_dir / "misc_distribution.png", dpi=300)
        plt.close(fig)

    features_path = output_root / "features" / "features.csv"
    if features_path.exists():
        features = pd.read_csv(features_path, dtype={"subject_id": str})
        if {"qc__median_rms_uV", "condition_label"}.issubset(features.columns):
            fig, ax = plt.subplots(figsize=(6, 4))
            features.boxplot(column="qc__median_rms_uV", by="condition_label", ax=ax)
            ax.set_title("Window RMS quality summary")
            ax.set_xlabel("Condition")
            ax.set_ylabel("Median channel RMS (uV)")
            fig.suptitle("")
            fig.tight_layout()
            fig.savefig(fig_dir / "window_rms_by_condition.png", dpi=300)
            plt.close(fig)

    quality_path = output_root / "features" / "signal_quality_condition_summary.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        if not quality.empty:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            quality.plot.bar(x="condition_label", y="median_rms_uV_median", ax=axes[0], legend=False, color="#5b8ff9")
            axes[0].set_title("Median RMS amplitude")
            axes[0].set_xlabel("Condition")
            axes[0].set_ylabel("uV")
            quality.plot.bar(x="condition_label", y="median_channel_low_gamma_relative_power_median", ax=axes[1], legend=False, color="#61dDAA")
            axes[1].set_title("Low-gamma relative power")
            axes[1].set_xlabel("Condition")
            axes[1].set_ylabel("Relative power")
            fig.tight_layout()
            fig.savefig(fig_dir / "signal_quality_by_condition.png", dpi=300)
            plt.close(fig)

    metrics_path = output_root / "models" / "metrics_all.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        cls = metrics[metrics["model"].notna()] if "model" in metrics else metrics.iloc[0:0]
        if not cls.empty and "f1_macro" in cls.columns:
            task_order = ["misc_3class_id", "misc_binary_2plus", "misc_binary_6plus", "susceptibility_id"]
            task_labels = {
                "misc_3class_id": "MISC 3-class",
                "misc_binary_2plus": "MISC \u2265 2",
                "misc_binary_6plus": "MISC \u2265 6",
                "susceptibility_id": "Susceptibility",
            }
            model_labels = {
                "LogisticRegression": "Logistic regression",
                "SVM_RBF": "SVM (RBF)",
                "RandomForest": "Random forest",
            }
            plot_data = cls[cls["label"].isin(task_order)].copy()
            plot_data["label"] = pd.Categorical(plot_data["label"], categories=task_order, ordered=True)
            plot_data["model"] = plot_data["model"].map(model_labels).fillna(plot_data["model"])
            pivot = plot_data.pivot_table(index="label", columns="model", values="f1_macro", aggfunc="mean")
            pivot = pivot.reindex(task_order)
            pivot.index = [task_labels.get(str(label), str(label)) for label in pivot.index]
            pivot = pivot.reindex(columns=["Logistic regression", "SVM (RBF)", "Random forest"])
            fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=150)
            x = np.arange(len(pivot.index))
            width = 0.23
            colors = ["#4C78A8", "#F58518", "#54A24B"]
            for i, (model, color) in enumerate(zip(pivot.columns, colors)):
                values = pivot[model].to_numpy(dtype=float)
                bars = ax.bar(x + (i - 1) * width, values, width=width, label=model, color=color)
                ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)

            ax.set_title("Macro-F1 across classification tasks", pad=12, fontsize=12, fontweight="bold")
            ax.set_xlabel("Classification task")
            ax.set_ylabel("Macro-F1")
            ax.set_xticks(x, pivot.index)
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(title="Model", frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
            fig.tight_layout(pad=1.2)
            fig.savefig(fig_dir / "model_macro_f1.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
