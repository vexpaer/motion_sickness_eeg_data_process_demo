from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
R2 = OUT / "revision2"
MODEL = {"LogisticRegression": "Logistic regression", "SVM_RBF": "SVM (RBF)", "RandomForest": "Random forest"}
TASK = {
    "misc_3class_id": "MISC 3-class",
    "misc_binary_2plus": r"MISC $\geq$2",
    "misc_binary_6plus": r"MISC $\geq$6",
    "susceptibility_id": "Susceptibility",
}


def f(x: float, digits: int = 3) -> str:
    return "--" if pd.isna(x) else f"{x:.{digits}f}"


def fq(q: float) -> str:
    return "$<$0.001" if q < 0.001 else f(q)


def ci(x: float, lo: float, hi: float) -> str:
    return f"{f(x)} ({f(lo, 2)}--{f(hi, 2)})"


def perm(path: Path) -> pd.Series:
    return pd.read_csv(path).iloc[0]


def perm_text(r: pd.Series, what: str, how: str) -> str:
    return (
        f"{what} {r['observed']:.3f} versus null mean {r['null_mean']:.3f} "
        f"($p={r['p_permutation']:.2f}$, {int(r['n_permutations']):,} {how})"
    )


RAW_PERM = R2 / "models_loso" / "raw_feature_permutation" / "permutation_test.csv"
SUSC_PERM = R2 / "models_loso" / "susceptibility_permutation_test.csv"
REST_PERM = R2 / "resting_baseline" / "permutation_test.csv"


def table(label: str, caption: str, spec: str, header: str, rows: list[str], size: str = r"\scriptsize") -> str:
    body = "\n".join(f"        {r} \\\\" for r in rows)
    return (
        "\\begin{table}[!htbp]\n    \\centering\n"
        f"    \\caption{{{caption}}}\n    \\label{{{label}}}\n    {size}\n"
        f"    \\begin{{tabular}}{{{spec}}}\n        \\toprule\n        {header} \\\\\n        \\midrule\n{body}\n"
        "        \\bottomrule\n    \\end{tabular}\n\\end{table}\n"
    )


def table_s1() -> str:
    lolo = pd.read_csv(OUT / "models" / "metrics_all.csv")
    loso = pd.read_csv(R2 / "models_loso" / "metrics_all_loso.csv")
    subj = pd.read_csv(R2 / "models_loso" / "susceptibility_subject_level_loso.csv").set_index("model")
    rows = []
    for scheme, frame in [("LOLO", lolo), ("LOSO", loso)]:
        for task in TASK:
            for model in MODEL:
                r = frame[(frame["label"] == task) & (frame["model"] == model)].iloc[0]
                part = f(subj.loc[model, "subject_auc"]) if scheme == "LOSO" and task == "susceptibility_id" else "--"
                rows.append(
                    f"{scheme} & {TASK[task]} & {MODEL[model]} & {f(r['balanced_accuracy'])} & {f(r['f1_macro'])} & "
                    f"{f(r['auc_macro_ovr'])} & {f(r['auc_macro_ovr_lap'])} & {part}"
                )
    ordinal = []
    for scheme, frame in [("LOLO", lolo), ("LOSO", loso)]:
        r = frame[frame["label"] == "ordinal_misc_score"].iloc[0]
        ordinal.append(f"{scheme}: MAE {r['mae']:.2f}, Spearman $\\rho$ {r['spearman_r']:.2f}")
    return table(
        "tab:s1",
        "Benchmark results under leave-one-lap-out (LOLO) and leave-one-subject-out (LOSO) validation. "
        "Balanced accuracy and macro-F1 are computed on windows; AUC is macro one-versus-rest, window-pooled or lap-averaged; "
        "participant-level AUC averages all windows of each held-out participant. Ordinal MISC model: "
        + "; ".join(ordinal)
        + ". Permutation tests (logistic regression): "
        + perm_text(perm(RAW_PERM), "LOSO MISC 3-class lap-averaged AUC", "within-participant permutations")
        + "; "
        + perm_text(perm(SUSC_PERM), "LOSO susceptibility participant-level AUC", "permutations")
        + ". Pooled LOSO AUC is not centred on 0.5 because participants differ in label composition.",
        "@{}lllccccc@{}",
        "Validation & Task & Model & Bal. acc. & Macro-F1 & AUC (window) & AUC (lap) & AUC (participant)",
        rows,
    )


def table_s2() -> str:
    boot = pd.read_csv(R2 / "lolo_3class_cluster_bootstrap.csv").set_index("model")
    infl = pd.read_csv(R2 / "lolo_3class_participant_influence.csv", dtype={"excluded_participant": str})
    wo = pd.read_csv(R2 / "models_lolo_without_25017" / "metrics_without_participant.csv")
    rows = []
    for model in MODEL:
        b = boot.loc[model]
        i = infl[infl["model"] == model]
        w3 = wo[(wo["label"] == "misc_3class_id") & (wo["model"] == model)].iloc[0]
        w6 = wo[(wo["label"] == "misc_binary_6plus") & (wo["model"] == model)].iloc[0]
        rows.append(
            f"{MODEL[model]} & {ci(b['macro_f1'], b['macro_f1_ci_lower'], b['macro_f1_ci_upper'])} & "
            f"{ci(b['high_recall'], b['high_recall_ci_lower'], b['high_recall_ci_upper'])} & "
            f"{ci(b['high_f1'], b['high_f1_ci_lower'], b['high_f1_ci_upper'])} & "
            f"{f(i['macro_f1'].min())}--{f(i['macro_f1'].max())} & "
            f"{f(w3['f1_macro'])} / {f(w3['recall_class_2'])} & {f(w6['recall_class_1'])}"
        )
    return table(
        "tab:s2",
        "Stability of the LOLO three-class MISC benchmark with respect to participants. Intervals are 95\\% participant-cluster bootstrap "
        "percentile intervals (2,000 resamples of the 15 participants; resamples without a high-stratum window, 3.75\\%, were dropped). "
        "Influence: range of macro-F1 when each participant is excluded from evaluation in turn. Without 25017: models retrained after "
        "removing participant 25017 (high stratum then contains 4 laps from 2 participants).",
        "@{}lcccccc@{}",
        "Model & Macro-F1 (95\\% CI) & High recall (95\\% CI) & High F1 (95\\% CI) & Influence macro-F1 & Without 25017: F1 / high recall & Without 25017: MISC$\\geq$6 recall",
        rows,
    )


def table_s3() -> str:
    stats = R2 / "statistics"
    comp = pd.read_csv(stats / "misc_channel_participant_aware_comparison.csv").set_index("entity")
    adj = pd.read_csv(stats / "misc_lmm_adjusted_feature_tests.csv")
    comp["min_q_lmm_adjusted"] = adj.groupby("entity")["q_lmm_bh"].min()
    order = pd.read_csv(stats / "misc_channel_statistics_table.csv")["entity"]
    rows = [
        f"\\texttt{{{r.entity}}} & {r.n_sig_lmm} & {r.n_sig_lmm_adjusted} & {fq(r.min_q_lmm_adjusted)} & {r.n_sig_kw} & "
        f"{r.n_sig_perm_participant} & {fq(r.q_channel_perm_participant)}"
        for r in comp.loc[order].reset_index().itertuples()
    ]
    return table(
        "tab:s3",
        "Channel-level results of the MISC-stratified statistics, including the sensitivity analysis. Count columns give the number of the "
        "channel's 21 features with $q<0.05$ (BH correction across 630 features): mixed model (LMM, primary analysis), mixed model with "
        "driving condition and lap order as additional fixed effects (LMM adj.), lap-level Kruskal--Wallis test (KW), and participant-stratified "
        "permutation of the Kruskal--Wallis $H$ statistic (10,000 within-participant permutations; Perm.). Channel $q$: Fisher combination of "
        "the channel's 21 feature-level $p$ values calibrated against the within-participant permutations (BH across 30 channels). Channels "
        "are ordered as in Table 7 of the main text.",
        "@{}lcccccc@{}",
        "Channel & LMM & LMM adj. & LMM adj. min. $q$ & KW & Perm. & Perm. channel $q$",
        rows,
    )


def table_s5() -> str:
    stats = R2 / "statistics"
    sel = pd.read_csv(stats / "psd_validation_features.csv")
    adj = pd.read_csv(stats / "misc_lmm_adjusted_feature_tests.csv").set_index("feature")["q_lmm_bh"]
    kw = pd.read_csv(OUT / "statistics" / "misc_kruskal_tests.csv").set_index("feature")["q_kruskal_bh"]
    band = {"theta": "$\\theta$ (4--8 Hz)", "alpha": "$\\alpha$ (8--13 Hz)", "beta": "$\\beta$ (13--30 Hz)"}
    rows = [
        f"\\texttt{{{r.entity}}} & {band[r.band]} & {'absolute' if r.measure == 'absolute_power' else 'relative'} & "
        f"{r.beta_mild_vs_none_sd:+.2f} & {r.beta_high_vs_none_sd:+.2f} & {fq(r.q_lmm_bh)} & {fq(adj[r.feature])} & "
        f"{fq(kw[r.feature])} & {fq(r.q_perm_participant)}"
        for r in sel.itertuples()
    ]
    return table(
        "tab:s5",
        "Theta, alpha and beta PSD features with $q<0.05$ in the mixed model. Effects are fixed-effect differences from the none-to-low "
        "stratum in standard deviations (absolute power log10-transformed). All $q$ values are BH-adjusted across 630 features: mixed model "
        "(LMM), mixed model adjusted for driving condition and lap order (LMM adj.), lap-level Kruskal--Wallis test (KW), and "
        "participant-stratified permutation (Perm.). The three PO3 absolute-power features are shown in Fig. 7 of the main text.",
        "@{}lllcccccc@{}",
        "Channel & Band & Power & Mild vs none & High vs none & LMM $q$ & LMM adj. $q$ & KW $q$ & Perm. $q$",
        rows,
    )


def table_s4() -> str:
    rb = pd.read_csv(R2 / "resting_baseline" / "metrics_resting_normalised.csv")
    rows = []
    for scheme, name in [("subject", "LOSO"), ("lap", "LOLO")]:
        for task in ["misc_3class_id", "misc_binary_2plus", "misc_binary_6plus"]:
            for model in MODEL:
                r = rb[(rb["validation"] == scheme) & (rb["label"] == task) & (rb["model"] == model)].iloc[0]
                rows.append(
                    f"{name} & {TASK[task]} & {MODEL[model]} & {f(r['balanced_accuracy'])} & {f(r['f1_macro'])} & {f(r['auc_macro_ovr_lap'])}"
                )
    return table(
        "tab:s4",
        "Pre-specified resting-baseline normalization. Each driving window was expressed relative to the "
        "median of the same session's pre-drive resting windows (log ratio for time-domain RMS, peak-to-peak amplitude and line length, Hjorth activity and absolute band power; difference otherwise). "
        "Its primary test (logistic regression, "
        + perm_text(perm(REST_PERM), "LOSO MISC 3-class lap-averaged AUC", "within-participant permutations")
        + ") did not meet the pre-specified criterion. Pooled LOSO AUC is not centred on 0.5 when participants differ in "
        "label composition, so the remaining LOSO values are descriptive and should not be read as above or below chance.",
        "@{}lllccc@{}",
        "Validation & Task & Model & Bal. acc. & Macro-F1 & AUC (lap)",
        rows,
    )


def figure_s1(out_dir: Path) -> None:
    panels = [
        (R2 / "models_loso" / "raw_feature_permutation" / "permutation_null.csv", R2 / "models_loso" / "raw_feature_permutation" / "permutation_test.csv",
         "MISC 3-class, LOSO\n(lap-averaged macro-AUC)"),
        (R2 / "models_loso" / "susceptibility_permutation_null.csv", R2 / "models_loso" / "susceptibility_permutation_test.csv",
         "Susceptibility, LOSO\n(participant-level AUC)"),
        (R2 / "resting_baseline" / "permutation_null.csv", R2 / "resting_baseline" / "permutation_test.csv",
         "MISC 3-class, LOSO, resting-normalized\n(lap-averaged macro-AUC)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)
    for ax, (null_path, test_path, title) in zip(axes, panels):
        null = pd.read_csv(null_path).iloc[:, 0]
        test = pd.read_csv(test_path).iloc[0]
        ax.hist(null, bins=30, color="#9aa7b8", edgecolor="white")
        ax.axvline(test["observed"], color="#c0392b", lw=2, label=f"observed {test['observed']:.2f}")
        ax.axvline(null.mean(), color="#2c3e50", lw=1, ls="--", label=f"null mean {null.mean():.2f}")
        ax.set_title(f"{title}\n$p$ = {test['p_permutation']:.2f}", fontsize=9)
        ax.set_xlabel("AUC under permuted labels", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("Permutations", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_s1_permutation_nulls.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Supplementary Tables S1-S5 (LaTeX) and Fig. S1.")
    parser.add_argument("output_dir", help="directory for supplementary_tables.tex and figure_s1_permutation_nulls.png")
    out_dir = Path(parser.parse_args().output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = "\n".join([table_s1(), table_s2(), table_s3(), table_s4(), table_s5()])
    (out_dir / "supplementary_tables.tex").write_text(tables, encoding="utf-8")
    figure_s1(out_dir)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
