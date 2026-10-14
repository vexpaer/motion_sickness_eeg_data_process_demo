from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from .modeling import _feature_columns, _make_pipeline, evaluate_label, evaluate_ordinal_misc
from .statistics import (
    _bh_fdr,
    _lap_level,
    _parse_channel_feature,
    _stat_columns,
    misc_fisher_entity_ranking,
    misc_kruskal_dunn_tests,
)

MISC_ORDER = ["none_to_low", "mild_to_moderate", "high"]
MODEL_LABELS = ["misc_3class_id", "misc_binary_2plus", "misc_binary_6plus", "susceptibility_id"]
MODEL_NAMES = ["LogisticRegression", "SVM_RBF", "RandomForest"]


def _write(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def misc_stratum_composition(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    laps = df.groupby("lap_group", as_index=False).agg(
        subject_id=("subject_id", "first"),
        condition_label=("condition_label", "first"),
        misc_3class_label=("misc_3class_label", "first"),
        n_windows=("sample_id", "count"),
    )
    rows: list[dict[str, Any]] = []
    for condition in ["all", "normal", "fast"]:
        subset = laps if condition == "all" else laps[laps["condition_label"] == condition]
        for label in MISC_ORDER:
            part = subset[subset["misc_3class_label"] == label]
            counts = part["subject_id"].value_counts().sort_values(ascending=False)
            rows.append(
                {
                    "condition": condition,
                    "misc_stratum": label,
                    "n_laps": int(len(part)),
                    "n_windows": int(part["n_windows"].sum()),
                    "n_participants": int(counts.size),
                    "max_laps_one_participant": int(counts.iloc[0]) if counts.size else 0,
                    "max_participant_share": float(counts.iloc[0] / len(part)) if counts.size else np.nan,
                    "laps_by_participant": ";".join(f"{sid}:{n}" for sid, n in counts.items()),
                }
            )
    summary = pd.DataFrame(rows)
    by_participant = (
        pd.crosstab(laps["subject_id"], laps["misc_3class_label"])
        .reindex(columns=MISC_ORDER, fill_value=0)
        .reset_index()
    )
    by_participant["n_laps"] = by_participant[MISC_ORDER].sum(axis=1)
    by_participant["n_strata_observed"] = (by_participant[MISC_ORDER] > 0).sum(axis=1)
    _write(summary, out_dir / "misc_stratum_composition.csv")
    _write(by_participant, out_dir / "misc_stratum_by_participant.csv")
    return summary, by_participant


def run_loso(df: pd.DataFrame, out_dir: Path, select_k: int, random_state: int) -> pd.DataFrame:
    model_dir = out_dir / "models_loso"
    metrics = [
        evaluate_label(df, label, model_dir, group_level="subject", select_k=select_k, random_state=random_state)
        for label in MODEL_LABELS
    ]
    metrics.append(evaluate_ordinal_misc(df, model_dir, group_level="subject", select_k=select_k))
    result = pd.concat(metrics, ignore_index=True, sort=False)
    _write(result, model_dir / "metrics_all_loso.csv")
    subject_level_susceptibility(model_dir)
    return result


def subject_level_susceptibility(model_dir: Path) -> pd.DataFrame:
    rows = []
    per_subject = []
    for model in MODEL_NAMES:
        path = model_dir / f"oof_susceptibility_id_{model}.csv"
        if not path.exists():
            continue
        oof = pd.read_csv(path, dtype={"subject_id": str})
        subj = oof.groupby("subject_id", as_index=False).agg(y_true=("y_true", "first"), prob=("prob_class_1", "mean"))
        subj["y_pred"] = (subj["prob"] >= 0.5).astype(int)
        subj["model"] = model
        per_subject.append(subj)
        rows.append(
            {
                "model": model,
                "n_participants": int(len(subj)),
                "subject_auc": float(roc_auc_score(subj["y_true"], subj["prob"])),
                "subject_accuracy": float((subj["y_true"] == subj["y_pred"]).mean()),
                "subject_balanced_accuracy": float(balanced_accuracy_score(subj["y_true"], subj["y_pred"])),
                "n_correct": int((subj["y_true"] == subj["y_pred"]).sum()),
            }
        )
    if per_subject:
        _write(pd.concat(per_subject, ignore_index=True), model_dir / "susceptibility_subject_predictions_loso.csv")
    return _write(pd.DataFrame(rows), model_dir / "susceptibility_subject_level_loso.csv")


def _loso_subject_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, select_k: int) -> float:
    prob = np.zeros(len(y))
    for subject in np.unique(groups):
        test = groups == subject
        train = ~test
        if len(np.unique(y[train])) < 2:
            return np.nan
        pipe = _make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced"), select_k, X.shape[1])
        pipe.fit(X[train], y[train])
        prob[test] = pipe.predict_proba(X[test])[:, 1]
    frame = pd.DataFrame({"g": groups, "y": y, "p": prob}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    return float(roc_auc_score(frame["y"], frame["p"]))


def susceptibility_permutation_test(
    df: pd.DataFrame,
    out_dir: Path,
    select_k: int,
    n_perm: int,
    random_state: int,
) -> pd.DataFrame:
    work = df[df["susceptibility_id"].notna() & (df["susceptibility_id"] >= 0)].copy()
    X = work[_feature_columns(work)].to_numpy(dtype=float)
    y = work["susceptibility_id"].to_numpy(dtype=int)
    groups = work["subject_id"].astype(str).to_numpy()
    observed = _loso_subject_auc(X, y, groups, select_k)

    subject_labels = work.groupby("subject_id")["susceptibility_id"].first()
    subjects = subject_labels.index.to_numpy().astype(str)
    rng = np.random.default_rng(random_state)
    null = []
    for _ in range(n_perm):
        mapping = dict(zip(subjects, rng.permutation(subject_labels.to_numpy())))
        y_perm = np.array([mapping[g] for g in groups], dtype=int)
        null.append(_loso_subject_auc(X, y_perm, groups, select_k))
    null_arr = np.asarray(null, dtype=float)
    null_arr = null_arr[np.isfinite(null_arr)]
    result = pd.DataFrame(
        [
            {
                "model": "LogisticRegression",
                "statistic": "LOSO subject-level AUC",
                "observed": observed,
                "n_permutations": int(len(null_arr)),
                "null_mean": float(null_arr.mean()),
                "null_95th": float(np.quantile(null_arr, 0.95)),
                "p_permutation": float((np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1)),
            }
        ]
    )
    _write(pd.DataFrame({"null_subject_auc": null_arr}), out_dir / "models_loso" / "susceptibility_permutation_null.csv")
    return _write(result, out_dir / "models_loso" / "susceptibility_permutation_test.csv")


def _threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "high_recall": float(recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)),
        "high_f1": float(f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)),
    }


def lolo_cluster_uncertainty(model_dir: Path, out_dir: Path, random_state: int, n_bootstrap: int = 2000) -> pd.DataFrame:
    rows = []
    influence_rows = []
    for model in MODEL_NAMES:
        oof = pd.read_csv(model_dir / f"oof_misc_3class_id_{model}.csv", dtype={"subject_id": str})
        y = oof["y_true"].to_numpy(dtype=int)
        pred = oof["y_pred"].to_numpy(dtype=int)
        subjects = oof["subject_id"].to_numpy()
        point = _threshold_metrics(y, pred)
        index = {s: np.flatnonzero(subjects == s) for s in np.unique(subjects)}
        keys = np.array(sorted(index))
        rng = np.random.default_rng(random_state)
        boot = []
        for _ in range(n_bootstrap):
            idx = np.concatenate([index[s] for s in rng.choice(keys, size=len(keys), replace=True)])
            if np.any(y[idx] == 2):
                boot.append(_threshold_metrics(y[idx], pred[idx]))
        boot_df = pd.DataFrame(boot)
        row: dict[str, Any] = {"model": model, "n_valid_replicates": int(len(boot_df)), "n_bootstrap": n_bootstrap}
        for key, value in point.items():
            row[key] = value
            row[f"{key}_ci_lower"] = float(boot_df[key].quantile(0.025))
            row[f"{key}_ci_upper"] = float(boot_df[key].quantile(0.975))
        rows.append(row)
        for subject in keys:
            keep = subjects != subject
            influence_rows.append(
                {"model": model, "excluded_participant": subject, **_threshold_metrics(y[keep], pred[keep])}
            )
    _write(pd.DataFrame(influence_rows), out_dir / "lolo_3class_participant_influence.csv")
    return _write(pd.DataFrame(rows), out_dir / "lolo_3class_cluster_bootstrap.csv")


def lolo_without_participant(
    df: pd.DataFrame, out_dir: Path, participant: str, select_k: int, random_state: int
) -> pd.DataFrame:
    subset = df[df["subject_id"].astype(str) != participant].copy()
    model_dir = out_dir / f"models_lolo_without_{participant}"
    result = pd.concat(
        [
            evaluate_label(subset, label, model_dir, group_level="lap", select_k=select_k, random_state=random_state)
            for label in ["misc_3class_id", "misc_binary_6plus"]
        ],
        ignore_index=True,
    )
    return _write(result, model_dir / "metrics_without_participant.csv")


def _log_scale(feature: str) -> bool:
    info = _parse_channel_feature(feature)
    return info["feature_family"] == "time" or info["measure"].startswith("activity") or info["measure"] == "absolute_power"


def _significant_channels(frame: pd.DataFrame, q_col: str, alpha: float = 0.05) -> int:
    return int(frame.loc[frame[q_col] < alpha, "entity"].nunique())


LMM_TERM = 'C(stratum, Treatment("none_to_low"))'
LMM_COVARIATES = "C(condition) + lap_index"


def _fit_ml(formula: str, data: pd.DataFrame) -> Any:
    import statsmodels.formula.api as smf

    model = smf.mixedlm(formula, data, groups=data["subject"])
    for method in ("nm", "powell"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(reml=False, method=method, maxiter=10000)
        if np.isfinite(fit.llf):
            return fit
    raise RuntimeError("no optimiser produced a finite log-likelihood")


def mixed_model_tests(lap_df: pd.DataFrame, stat_cols: list[str], adjust: bool) -> pd.DataFrame:
    work = lap_df[lap_df["misc_3class_label"].isin(MISC_ORDER)].copy()
    covariates = f" + {LMM_COVARIATES}" if adjust else ""
    rows = []
    for feature in stat_cols:
        values = pd.to_numeric(work[feature], errors="coerce").to_numpy(dtype=float)
        transformed = _log_scale(feature) and np.all(values[np.isfinite(values)] > 0)
        if transformed:
            values = np.log10(values)
        sd = np.nanstd(values)
        data = pd.DataFrame(
            {
                "y": (values - np.nanmean(values)) / sd if sd > 0 else values,
                "stratum": work["misc_3class_label"].to_numpy(),
                "condition": work["condition_label"].astype(str).to_numpy(),
                "lap_index": pd.to_numeric(work["lap_index"], errors="coerce").to_numpy(dtype=float),
                "subject": work["subject_id"].astype(str).to_numpy(),
            }
        ).dropna()
        item = _parse_channel_feature(feature)
        try:
            full = _fit_ml(f"y ~ {LMM_TERM}{covariates}", data)
            null = _fit_ml(f"y ~ 1{covariates}", data)
            lr = max(2.0 * (full.llf - null.llf), 0.0)
            item.update(
                {
                    "adjusted_for": LMM_COVARIATES if adjust else "",
                    "log10_transformed": bool(transformed),
                    "n_laps": int(len(data)),
                    "beta_mild_vs_none_sd": float(full.fe_params.get(f"{LMM_TERM}[T.mild_to_moderate]", np.nan)),
                    "beta_high_vs_none_sd": float(full.fe_params.get(f"{LMM_TERM}[T.high]", np.nan)),
                    "participant_variance": float(full.cov_re.iloc[0, 0]),
                    "residual_variance": float(full.scale),
                    "lr_statistic": float(lr),
                    "p_lmm": float(chi2.sf(lr, df=2)),
                    "converged": bool(full.converged and null.converged),
                }
            )
        except Exception as exc:
            item.update({"log10_transformed": bool(transformed), "p_lmm": np.nan, "converged": False, "error": str(exc)})
        rows.append(item)
    out = pd.DataFrame(rows)
    out["q_lmm_bh"] = _bh_fdr(out["p_lmm"])
    return out.sort_values(["q_lmm_bh", "feature"]).reset_index(drop=True)


def _kw_h(ranks: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
    n = ranks.shape[0]
    if not np.allclose(np.sort(ranks, axis=0), np.arange(1, n + 1)[:, None]):
        raise ValueError("tied values: _kw_h has no tie correction")
    one_hot = np.eye(n_groups)[labels]
    sizes = one_hot.sum(axis=0)
    rank_sums = one_hot.T @ ranks
    return 12.0 / (n * (n + 1.0)) * np.sum(rank_sums**2 / sizes[:, None], axis=0) - 3.0 * (n + 1.0)


def kw_permutation_tests(
    lap_df: pd.DataFrame,
    stat_cols: list[str],
    n_perm: int,
    random_state: int,
    block_cols: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = lap_df[lap_df["misc_3class_label"].isin(MISC_ORDER)].reset_index(drop=True)
    labels = work["misc_3class_label"].map({k: i for i, k in enumerate(MISC_ORDER)}).to_numpy(dtype=int)
    values = work[stat_cols].to_numpy(dtype=float)
    ranks = np.column_stack([rankdata(values[:, j]) for j in range(values.shape[1])])
    entities = np.array([_parse_channel_feature(c)["entity"] for c in stat_cols])
    channel_names = np.unique(entities)
    channel_masks = [entities == ch for ch in channel_names]

    def channel_fisher(h: np.ndarray) -> np.ndarray:
        log_p = chi2.logsf(np.maximum(h, 0.0), df=len(MISC_ORDER) - 1)
        return np.array([-2.0 * log_p[mask].sum() for mask in channel_masks])

    if block_cols is None:
        blocks = [np.arange(len(work))]
    else:
        keys = work[block_cols].astype(str).agg("|".join, axis=1).to_numpy()
        blocks = [np.flatnonzero(keys == key) for key in pd.unique(keys)]

    observed_h = _kw_h(ranks, labels, len(MISC_ORDER))
    observed_f = channel_fisher(observed_h)
    rng = np.random.default_rng(random_state)
    exceed_h = np.zeros(len(stat_cols))
    exceed_f = np.zeros(len(channel_names))
    for _ in range(n_perm):
        permuted = labels.copy()
        for block in blocks:
            permuted[block] = rng.permutation(labels[block])
        h = _kw_h(ranks, permuted, len(MISC_ORDER))
        exceed_h += h >= observed_h * (1.0 - 1e-12)
        exceed_f += channel_fisher(h) >= observed_f * (1.0 - 1e-12)

    scheme = "free" if block_cols is None else "within_" + "_x_".join(block_cols)
    informative = sum(len(np.unique(labels[b])) > 1 for b in blocks)
    features = pd.DataFrame([_parse_channel_feature(c) for c in stat_cols])
    features["scheme"] = scheme
    features["h_statistic"] = observed_h
    features["p_perm"] = (exceed_h + 1.0) / (n_perm + 1.0)
    features["q_perm_bh"] = _bh_fdr(features["p_perm"])
    features["n_blocks"] = len(blocks)
    features["n_informative_blocks"] = int(informative)
    features["n_permutations"] = n_perm
    channels = pd.DataFrame(
        {
            "entity": channel_names,
            "scheme": scheme,
            "fisher_statistic": observed_f,
            "p_channel_perm": (exceed_f + 1.0) / (n_perm + 1.0),
        }
    )
    channels["q_channel_perm_bh"] = _bh_fdr(channels["p_channel_perm"])
    return (
        features.sort_values(["q_perm_bh", "feature"]).reset_index(drop=True),
        channels.sort_values(["q_channel_perm_bh", "entity"]).reset_index(drop=True),
    )


def participant_aware_statistics(
    df: pd.DataFrame, out_dir: Path, n_perm: int, random_state: int, exclude_participant: str
) -> pd.DataFrame:
    stat_cols = _stat_columns(df)
    lap_df = _lap_level(df, stat_cols)
    stats_dir = out_dir / "statistics"

    kw, _ = misc_kruskal_dunn_tests(lap_df, stat_cols)
    kw_fisher = misc_fisher_entity_ranking(kw)
    lmm = _write(mixed_model_tests(lap_df, stat_cols, adjust=False), stats_dir / "misc_lmm_feature_tests.csv")
    lmm_adj = _write(mixed_model_tests(lap_df, stat_cols, adjust=True), stats_dir / "misc_lmm_adjusted_feature_tests.csv")
    perms = {}
    for name, blocks in [("free", None), ("participant", ["subject_id"]), ("session", ["subject_id", "condition_label"])]:
        feat, chan = kw_permutation_tests(lap_df, stat_cols, n_perm, random_state, blocks)
        perms[name] = (_write(feat, stats_dir / f"misc_kw_permutation_{name}_features.csv"), _write(chan, stats_dir / f"misc_kw_permutation_{name}_channels.csv"))
    lap_wo = lap_df[lap_df["subject_id"].astype(str) != exclude_participant]
    kw_wo, _ = misc_kruskal_dunn_tests(lap_wo, stat_cols)
    _write(kw_wo, stats_dir / f"misc_kruskal_tests_without_{exclude_participant}.csv")
    perm_wo_feat, perm_wo_chan = kw_permutation_tests(lap_wo, stat_cols, n_perm, random_state, ["subject_id"])
    _write(perm_wo_feat, stats_dir / f"misc_kw_permutation_participant_without_{exclude_participant}_features.csv")
    _write(perm_wo_chan, stats_dir / f"misc_kw_permutation_participant_without_{exclude_participant}_channels.csv")

    def counts(frame: pd.DataFrame, q_col: str, name: str) -> pd.DataFrame:
        return frame.assign(sig=frame[q_col] < 0.05).groupby("entity")["sig"].sum().astype(int).rename(name).reset_index()

    channels = kw_fisher[["entity", "combined_fdr_bh"]].rename(columns={"combined_fdr_bh": "q_kw_naive_fisher"})
    for frame, q_col, name in [
        (kw, "q_kruskal_bh", "n_sig_kw"),
        (lmm, "q_lmm_bh", "n_sig_lmm"),
        (lmm_adj, "q_lmm_bh", "n_sig_lmm_adjusted"),
        (perms["participant"][0], "q_perm_bh", "n_sig_perm_participant"),
        (perms["session"][0], "q_perm_bh", "n_sig_perm_session"),
        (kw_wo, "q_kruskal_bh", f"n_sig_kw_without_{exclude_participant}"),
    ]:
        channels = channels.merge(counts(frame, q_col, name), on="entity", how="left")
    for name in ["free", "participant", "session"]:
        channels = channels.merge(
            perms[name][1][["entity", "q_channel_perm_bh"]].rename(columns={"q_channel_perm_bh": f"q_channel_perm_{name}"}),
            on="entity",
            how="left",
        )
    _write(channels, stats_dir / "misc_channel_participant_aware_comparison.csv")

    def beta(frame: pd.DataFrame, q_col: str, name: str, extra: list[str] | None = None) -> pd.DataFrame:
        mask = (frame["feature_family"] == "psd") & (frame["band"] == "beta") & (frame["measure"] == "absolute_power")
        return frame.loc[mask, ["entity", q_col, *(extra or [])]].rename(columns={q_col: name})

    beta_table = (
        beta(kw, "q_kruskal_bh", "q_kw")
        .merge(beta(lmm, "q_lmm_bh", "q_lmm", ["beta_high_vs_none_sd"]), on="entity")
        .merge(beta(lmm_adj, "q_lmm_bh", "q_lmm_adjusted"), on="entity")
        .merge(beta(perms["participant"][0], "q_perm_bh", "q_perm_participant"), on="entity")
        .merge(beta(perms["session"][0], "q_perm_bh", "q_perm_session"), on="entity")
        .sort_values("q_kw")
    )
    _write(beta_table, stats_dir / "misc_beta_participant_aware_comparison.csv")

    def row(analysis: str, frame: pd.DataFrame, q_col: str, beta_col: str | None, calibrated: pd.DataFrame | None = None, **extra: Any) -> dict[str, Any]:
        return {
            "analysis": analysis,
            "features_q_lt_0_05": int((frame[q_col] < 0.05).sum()),
            "channels_with_sig_feature": _significant_channels(frame, q_col),
            "channels_calibrated_q_lt_0_05": int((calibrated["q_channel_perm_bh"] < 0.05).sum()) if calibrated is not None else np.nan,
            "beta_channels_q_lt_0_05": int((beta_table[beta_col] < 0.05).sum()) if beta_col else np.nan,
            **extra,
        }

    summary = pd.DataFrame(
        [
            row(
                "lap-level Kruskal-Wallis (auxiliary)", kw, "q_kruskal_bh", "q_kw", perms["free"][1],
                channels_naive_fisher_q_lt_0_05=int((kw_fisher["combined_fdr_bh"] < 0.05).sum()),
            ),
            row("LMM, participant random intercept", lmm, "q_lmm_bh", "q_lmm", lmm_converged=int(lmm["converged"].sum())),
            row("LMM, participant random intercept + condition + lap order", lmm_adj, "q_lmm_bh", "q_lmm_adjusted", lmm_converged=int(lmm_adj["converged"].sum())),
            row("KW permutation within participant", perms["participant"][0], "q_perm_bh", "q_perm_participant", perms["participant"][1], informative_blocks=int(perms["participant"][0]["n_informative_blocks"].iloc[0])),
            row("KW permutation within participant x condition", perms["session"][0], "q_perm_bh", "q_perm_session", perms["session"][1], informative_blocks=int(perms["session"][0]["n_informative_blocks"].iloc[0])),
            row(f"lap-level Kruskal-Wallis without participant {exclude_participant}", kw_wo, "q_kruskal_bh", None),
            row(
                f"KW permutation within participant, without participant {exclude_participant}",
                perm_wo_feat, "q_perm_bh", None, perm_wo_chan,
                informative_blocks=int(perm_wo_feat["n_informative_blocks"].iloc[0]),
            ),
        ]
    )
    return _write(summary, stats_dir / "participant_aware_statistics_summary.csv")


def label_only_baselines(df: pd.DataFrame, primary_model_dir: Path, out_dir: Path) -> pd.DataFrame:
    laps = df.groupby("lap_group", as_index=False).agg(
        subject_id=("subject_id", "first"),
        session_id=("session_id", "first"),
        lap_index=("lap_index", "first"),
        y=("misc_3class_id", "first"),
    )
    laps = laps[laps["y"] >= 0].reset_index(drop=True)
    classes = np.array([0, 1, 2])

    def predict(group_col: str, nearest: bool) -> np.ndarray:
        scores = []
        for i, row in laps.iterrows():
            others = laps[(laps[group_col] == row[group_col]) & (laps.index != i)]
            if nearest and len(others):
                distance = (others["lap_index"] - row["lap_index"]).abs()
                others = others[distance == distance.min()]
            counts = np.array([(others["y"] == c).sum() for c in classes], dtype=float) + 0.01
            scores.append(counts / counts.sum())
        return np.asarray(scores)

    y = laps["y"].to_numpy(dtype=int)
    rows = []
    for name, col, nearest in [
        ("same participant, other laps", "subject_id", False),
        ("same session, other laps", "session_id", False),
        ("same session, nearest lap", "session_id", True),
    ]:
        score = predict(col, nearest)
        pred = classes[score.argmax(axis=1)]
        rows.append(
            {
                "predictor": f"label-only: {name}",
                "uses_eeg": False,
                "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "macro_auc_lap": float(roc_auc_score(np.eye(3)[y], score, average="macro")),
            }
        )

    for model in MODEL_NAMES:
        oof = pd.read_csv(primary_model_dir / f"oof_misc_3class_id_{model}.csv", dtype={"subject_id": str})
        lap = oof.groupby("lap_group", as_index=False).agg(
            subject_id=("subject_id", "first"),
            session_id=("session_id", "first"),
            y=("y_true", "first"),
            p0=("prob_class_0", "mean"),
            p2=("prob_class_2", "mean"),
        )
        lap["severity_score"] = lap["p2"] - lap["p0"]

        def within(key: str) -> tuple[float, int]:
            values = []
            for _, group in lap.groupby(key):
                if group["y"].nunique() > 1:
                    values.append(roc_auc_score(group["y"] > group["y"].min(), group["severity_score"]))
            return float(np.mean(values)), len(values)

        participant_auc, n_participants = within("subject_id")
        session_auc, n_sessions = within("session_id")
        metrics = pd.read_csv(primary_model_dir / f"metrics_misc_3class_id.csv")
        metric = metrics[metrics["model"] == model].iloc[0]
        rows.append(
            {
                "predictor": f"EEG leave-one-lap-out: {model}",
                "uses_eeg": True,
                "macro_f1": float(metric["f1_macro"]),
                "balanced_accuracy": float(metric["balanced_accuracy"]),
                "macro_auc_lap": float(metric["auc_macro_ovr_lap"]),
                "within_participant_auc": participant_auc,
                "n_participants_with_variation": n_participants,
                "within_session_auc": session_auc,
                "n_sessions_with_variation": n_sessions,
            }
        )
    return _write(pd.DataFrame(rows), out_dir / "lolo_label_only_baselines.csv")


def auc_aggregation_table(primary_dir: Path, loso_dir: Path, out_dir: Path) -> pd.DataFrame:
    frames = []
    for scheme, path in [("leave-one-lap-out", primary_dir / "metrics_all.csv"), ("leave-one-subject-out", loso_dir / "metrics_all_loso.csv")]:
        if not path.exists():
            continue
        metrics = pd.read_csv(path)
        metrics = metrics[metrics["label"].isin(MODEL_LABELS)]
        frames.append(
            metrics[
                ["label", "model", "accuracy", "balanced_accuracy", "f1_macro", "auc_macro_ovr", "auc_macro_ovr_lap"]
            ]
            .rename(
                columns={
                    "accuracy": "accuracy_window",
                    "balanced_accuracy": "balanced_accuracy_window",
                    "f1_macro": "macro_f1_window",
                    "auc_macro_ovr": "macro_auc_window_pooled",
                    "auc_macro_ovr_lap": "macro_auc_lap_averaged",
                }
            )
            .assign(validation=scheme)
        )
    subject_path = loso_dir / "susceptibility_subject_level_loso.csv"
    if subject_path.exists():
        subject = pd.read_csv(subject_path)
        frames.append(
            subject.rename(
                columns={
                    "subject_auc": "auc_participant_level",
                    "subject_accuracy": "accuracy_participant",
                    "subject_balanced_accuracy": "balanced_accuracy_participant",
                }
            )[["model", "accuracy_participant", "balanced_accuracy_participant", "auc_participant_level"]].assign(
                label="susceptibility_id", validation="leave-one-subject-out (participant-averaged)"
            )
        )
    return _write(pd.concat(frames, ignore_index=True), out_dir / "benchmark_auc_by_aggregation_level.csv")


BOOTSTRAP_FEATURES = ["PO3__psd__theta_abs_uV2", "PO3__psd__alpha_abs_uV2", "PO3__psd__beta_abs_uV2"]


def parametric_bootstrap_lrt(
    df: pd.DataFrame, out_dir: Path, features: list[str], n_boot: int, random_state: int
) -> pd.DataFrame:
    lap_df = _lap_level(df, features)
    work = lap_df[lap_df["misc_3class_label"].isin(MISC_ORDER)].reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    rows = []
    for feature in features:
        values = pd.to_numeric(work[feature], errors="coerce").to_numpy(dtype=float)
        if _log_scale(feature) and np.all(values[np.isfinite(values)] > 0):
            values = np.log10(values)
        data = pd.DataFrame(
            {
                "y": (values - np.nanmean(values)) / np.nanstd(values),
                "stratum": work["misc_3class_label"].to_numpy(),
                "subject": work["subject_id"].astype(str).to_numpy(),
            }
        ).dropna()
        full = _fit_ml(f"y ~ {LMM_TERM}", data)
        null = _fit_ml("y ~ 1", data)
        observed = max(2.0 * (full.llf - null.llf), 0.0)
        subjects, codes = np.unique(data["subject"], return_inverse=True)
        intercept = float(null.fe_params.iloc[0])
        sd_subject = float(np.sqrt(max(null.cov_re.iloc[0, 0], 0.0)))
        sd_resid = float(np.sqrt(null.scale))
        simulated = []
        for _ in range(n_boot):
            sim = data.copy()
            sim["y"] = intercept + rng.normal(0.0, sd_subject, len(subjects))[codes] + rng.normal(0.0, sd_resid, len(sim))
            simulated.append(max(2.0 * (_fit_ml(f"y ~ {LMM_TERM}", sim).llf - _fit_ml("y ~ 1", sim).llf), 0.0))
        simulated = np.asarray(simulated)
        rows.append(
            {
                "feature": feature,
                "lr_statistic": observed,
                "p_chi2": float(chi2.sf(observed, df=2)),
                "p_parametric_bootstrap": float((np.sum(simulated >= observed) + 1) / (n_boot + 1)),
                "n_bootstrap": n_boot,
                "beta_high_vs_none_sd": float(full.fe_params.get(f"{LMM_TERM}[T.high]", np.nan)),
            }
        )
    return _write(pd.DataFrame(rows), out_dir / "statistics" / "lmm_parametric_bootstrap.csv")


ALTERNATIVE_PARTITIONS = {
    "0-2_3-5_6-10": [(0, 2), (3, 5), (6, 10)],
    "0-1_2-4_5-10": [(0, 1), (2, 4), (5, 10)],
}


def channel_statistics_table(out_dir: Path) -> pd.DataFrame:
    stats_dir = out_dir / "statistics"
    lmm = pd.read_csv(stats_dir / "misc_lmm_feature_tests.csv")
    kw = pd.read_csv(Path(out_dir).parent / "statistics" / "misc_kruskal_tests.csv")
    perm = pd.read_csv(stats_dir / "misc_kw_permutation_participant_features.csv")
    perm_ch = pd.read_csv(stats_dir / "misc_kw_permutation_participant_channels.csv")

    def counts(frame: pd.DataFrame, q_col: str) -> pd.Series:
        return (frame[q_col] < 0.05).groupby(frame["entity"]).sum().astype(int)

    table = pd.DataFrame(
        {
            "n_sig_lmm": counts(lmm, "q_lmm_bh"),
            "min_q_lmm": lmm.groupby("entity")["q_lmm_bh"].min(),
            "n_sig_kw": counts(kw, "q_kruskal_bh"),
            "n_sig_perm_participant": counts(perm, "q_perm_bh"),
            "q_channel_perm_participant": perm_ch.set_index("entity")["q_channel_perm_bh"],
        }
    )
    table = table.rename_axis("entity").reset_index().sort_values(["n_sig_lmm", "min_q_lmm"], ascending=[False, True])
    return _write(table, stats_dir / "misc_channel_statistics_table.csv")


def alternative_partition_tests(df: pd.DataFrame, out_dir: Path, n_perm: int, random_state: int) -> pd.DataFrame:
    stat_cols = _stat_columns(df)
    lap_df = _lap_level(df, stat_cols)
    score = pd.to_numeric(lap_df["misc_score"], errors="coerce")
    rows = []
    for scheme, bins in ALTERNATIVE_PARTITIONS.items():
        work = lap_df.copy()
        work["misc_3class_label"] = None
        for (low, high), label in zip(bins, MISC_ORDER):
            work.loc[(score >= low) & (score <= high), "misc_3class_label"] = label
        lmm = mixed_model_tests(work, stat_cols, adjust=False)
        kw, _ = misc_kruskal_dunn_tests(work, stat_cols)
        perm, perm_ch = kw_permutation_tests(work, stat_cols, n_perm, random_state, ["subject_id"])
        rows.append(
            {
                "scheme": scheme,
                "lmm_features": int((lmm["q_lmm_bh"] < 0.05).sum()),
                "lmm_channels": _significant_channels(lmm, "q_lmm_bh"),
                "kw_features": int((kw["q_kruskal_bh"] < 0.05).sum()),
                "kw_channels": _significant_channels(kw, "q_kruskal_bh"),
                "perm_features": int((perm["q_perm_bh"] < 0.05).sum()),
                "perm_channels": _significant_channels(perm, "q_perm_bh"),
                "perm_channels_calibrated": int((perm_ch["q_channel_perm_bh"] < 0.05).sum()),
                "perm_calibrated_channel_names": ",".join(perm_ch.loc[perm_ch["q_channel_perm_bh"] < 0.05, "entity"]),
                "po3_lmm_features": int(((lmm["q_lmm_bh"] < 0.05) & (lmm["entity"] == "PO3")).sum()),
            }
        )
    return _write(pd.DataFrame(rows), out_dir / "statistics" / "misc_alternative_partitions.csv")


PSD_BANDS = ["theta", "alpha", "beta"]
BAND_SYMBOLS = {"theta": "θ", "alpha": "α", "beta": "β"}
BAND_RANGES = {"theta": "4–8 Hz", "alpha": "8–13 Hz", "beta": "13–30 Hz"}
PSD_FIG_FONT = 9


def psd_validation_figure(df: pd.DataFrame, out_dir: Path, alpha: float = 0.05) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats_dir = out_dir / "statistics"
    lmm = pd.read_csv(stats_dir / "misc_lmm_feature_tests.csv")
    perm = pd.read_csv(stats_dir / "misc_kw_permutation_participant_features.csv").set_index("feature")["q_perm_bh"]
    selected = lmm[(lmm["q_lmm_bh"] < alpha) & (lmm["feature_family"] == "psd") & lmm["band"].isin(PSD_BANDS)].copy()
    selected["q_perm_participant"] = selected["feature"].map(perm)
    adjusted = pd.read_csv(stats_dir / "misc_lmm_adjusted_feature_tests.csv").set_index("feature")["q_lmm_bh"]
    kw = pd.read_csv(Path(out_dir).parent / "statistics" / "misc_kruskal_tests.csv").set_index("feature")["q_kruskal_bh"]
    selected["q_lmm_adjusted"] = selected["feature"].map(adjusted)
    selected["q_kw"] = selected["feature"].map(kw)
    selected["band_order"] = selected["band"].map(PSD_BANDS.index)
    selected["measure_order"] = (selected["measure"] != "absolute_power").astype(int)
    selected = selected.sort_values(["band_order", "measure_order", "q_lmm_bh"]).reset_index(drop=True)
    _write(
        selected[["feature", "entity", "band", "measure", "q_lmm_bh", "q_lmm_adjusted", "q_kw", "q_perm_participant", "beta_mild_vs_none_sd", "beta_high_vs_none_sd"]],
        stats_dir / "psd_validation_features.csv",
    )
    shown = selected[selected["q_perm_participant"] < alpha].reset_index(drop=True)

    stat_cols = _stat_columns(df)
    laps = _lap_level(df, stat_cols)
    laps = laps[laps["misc_3class_label"].isin(MISC_ORDER)]
    subjects = sorted(laps["subject_id"].astype(str).unique())
    cmap = plt.get_cmap("tab20")
    colours = {s: cmap(i % 20) for i, s in enumerate(subjects)}
    tick_labels = ["None–low\n(0–1)", "Mild–mod.\n(2–5)", "High\n(6–10)"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": PSD_FIG_FONT, "axes.titlesize": PSD_FIG_FONT,
                         "axes.labelsize": PSD_FIG_FONT, "xtick.labelsize": PSD_FIG_FONT, "ytick.labelsize": PSD_FIG_FONT,
                         "legend.fontsize": PSD_FIG_FONT, "legend.title_fontsize": PSD_FIG_FONT})
    fig, axes = plt.subplots(1, len(shown), figsize=(7.5, 4.6), squeeze=False, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (_, row) in zip(axes[0], shown.iterrows()):
        feature = row["feature"]
        groups = [laps.loc[laps["misc_3class_label"] == s, [feature, "subject_id"]] for s in MISC_ORDER]
        ax.boxplot(
            [g[feature].to_numpy(dtype=float) for g in groups],
            positions=[1, 2, 3],
            widths=0.55,
            showfliers=False,
            medianprops={"color": "black"},
            boxprops={"color": "#555555"},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
        )
        for pos, g in enumerate(groups, start=1):
            ax.scatter(
                pos + rng.uniform(-0.18, 0.18, len(g)),
                g[feature].to_numpy(dtype=float),
                s=8,
                c=[colours[str(s)] for s in g["subject_id"]],
                alpha=0.85,
                linewidths=0,
                zorder=3,
            )
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 3], tick_labels)
        ax.set_xlabel("MISC stratum")
        band = row["band"]
        ax.set_title(
            f"{row['entity']} {BAND_SYMBOLS[band]} ({BAND_RANGES[band]})\n"
            f"High vs none: {row['beta_high_vs_none_sd']:+.2f} SD\n"
            f"LMM q={row['q_lmm_bh']:.3f}; adj. q={row['q_lmm_adjusted']:.3f}\n"
            f"KW q={row['q_kw']:.3f}; perm. q={row['q_perm_participant']:.3f}",
        )
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Absolute power (µV², log scale)")
    handles = [plt.Line2D([], [], marker="o", markersize=4, linestyle="", color=colours[s], label=s) for s in subjects]
    fig.legend(handles=handles, title="Participant", loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=8,
               frameon=False, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "psd_validation_theta_alpha_beta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return selected
