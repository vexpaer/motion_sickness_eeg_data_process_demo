from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, kruskal, mannwhitneyu, norm, rankdata, spearmanr, wilcoxon

from .features import metadata_columns
from .quality import run_quality_report


def _stat_columns(df: pd.DataFrame) -> list[str]:
    meta = metadata_columns()
    cols: list[str] = []
    for col in df.columns:
        if col in meta or col.startswith("qc__") or "__" not in col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _parse_channel_feature(feature: str) -> dict[str, str]:
    parts = feature.split("__")
    scope = parts[0] if parts else ""
    family = parts[1] if len(parts) > 1 else ""
    metric = parts[2] if len(parts) > 2 else ""
    base_family = family
    band = ""
    measure = base_family
    if base_family == "psd":
        if metric.endswith("_rel_mean"):
            band = metric.removesuffix("_rel_mean")
            measure = "relative_power"
        elif metric.endswith("_abs_uV2_mean"):
            band = metric.removesuffix("_abs_uV2_mean")
            measure = "absolute_power"
        elif metric.endswith("_rel"):
            band = metric.removesuffix("_rel")
            measure = "relative_power"
        elif metric.endswith("_abs_uV2"):
            band = metric.removesuffix("_abs_uV2")
            measure = "absolute_power"
    elif base_family == "de":
        band = metric.removesuffix("_mean")
        measure = "differential_entropy"
    elif base_family in {"time", "hjorth"}:
        measure = metric
    else:
        measure = metric or base_family
    return {
        "feature": feature,
        "entity": scope,
        "level": "channel",
        "channel": scope,
        "feature_family": base_family,
        "band": band,
        "measure": measure,
    }


def _bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(q, index=p_values.index)
    idx = np.where(valid)[0]
    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q[idx[order]] = adjusted
    return pd.Series(q, index=p_values.index)


def _paired_sign_flip_p(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return np.nan
    observed = abs(float(np.mean(diff)))
    if n <= 20:
        signs = np.array(list(product([-1.0, 1.0], repeat=n)), dtype=float)
        null = np.abs((signs * diff).mean(axis=1))
        return float((np.sum(null >= observed - 1e-15)) / len(null))
    rng = np.random.default_rng(97)
    signs = rng.choice([-1.0, 1.0], size=(100000, n))
    null = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(null >= observed - 1e-15) + 1) / (len(null) + 1))


def _safe_wilcoxon(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if len(diff) == 0:
        return np.nan
    try:
        return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def _rank_biserial_from_u(u_stat: float, n_a: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0 or not np.isfinite(u_stat):
        return np.nan
    return float((2.0 * u_stat / (n_a * n_b)) - 1.0)


def _lap_level(df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    meta_cols = [
        "subject_id",
        "susceptibility_group",
        "susceptibility_id",
        "condition",
        "condition_label",
        "session_id",
        "lap_index",
        "lap_group",
        "misc_score",
        "misc_3class_id",
        "misc_3class_label",
    ]
    present_meta = [c for c in meta_cols if c in df.columns]
    grouped = df.groupby("lap_group", dropna=False)
    meta = grouped[present_meta].first().reset_index(drop=True)
    values = grouped[stat_cols].mean(numeric_only=True).reset_index(drop=True)
    return pd.concat([meta, values], axis=1)


def condition_paired_tests(lap_df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    subject_condition = (
        lap_df.groupby(["subject_id", "condition_label"], dropna=False)[stat_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for feature in stat_cols:
        wide = subject_condition.pivot(index="subject_id", columns="condition_label", values=feature)
        if "normal" not in wide or "fast" not in wide:
            continue
        paired = wide[["normal", "fast"]].dropna()
        normal = paired["normal"].to_numpy(dtype=float)
        fast = paired["fast"].to_numpy(dtype=float)
        diff = fast - normal
        item = _parse_channel_feature(feature)
        item.update(
            {
                "analysis": "condition_paired_fast_minus_normal",
                "n_subjects": int(len(diff)),
                "normal_mean": float(np.mean(normal)) if len(normal) else np.nan,
                "fast_mean": float(np.mean(fast)) if len(fast) else np.nan,
                "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
                "median_diff": float(np.median(diff)) if len(diff) else np.nan,
                "cohens_dz": float(np.mean(diff) / np.std(diff, ddof=1)) if len(diff) > 1 and np.std(diff, ddof=1) > 0 else np.nan,
                "p_signflip": _paired_sign_flip_p(diff),
                "p_wilcoxon": _safe_wilcoxon(diff),
            }
        )
        rows.append(item)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_signflip_bh"] = _bh_fdr(out["p_signflip"])
        out["q_wilcoxon_bh"] = _bh_fdr(out["p_wilcoxon"])
        out = out.sort_values(["q_signflip_bh", "p_signflip", "feature"], na_position="last")
    return out


def misc_spearman_tests(lap_df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    work = lap_df[pd.to_numeric(lap_df["misc_score"], errors="coerce").notna()].copy()
    work["misc_score"] = pd.to_numeric(work["misc_score"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for feature in stat_cols:
        values = pd.to_numeric(work[feature], errors="coerce")
        valid = values.notna() & work["misc_score"].notna()
        item = _parse_channel_feature(feature)
        if valid.sum() >= 3 and values[valid].nunique() > 1 and work.loc[valid, "misc_score"].nunique() > 1:
            rho, p_value = spearmanr(values[valid], work.loc[valid, "misc_score"])
        else:
            rho, p_value = np.nan, np.nan
        item.update(
            {
                "analysis": "lap_level_misc_spearman",
                "n_laps": int(valid.sum()),
                "rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_spearman": float(p_value) if np.isfinite(p_value) else np.nan,
            }
        )
        rows.append(item)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_spearman_bh"] = _bh_fdr(out["p_spearman"])
        out = out.sort_values(["q_spearman_bh", "p_spearman", "feature"], na_position="last")
    return out


def _dunn_pairwise(values: pd.Series, groups: pd.Series) -> list[dict[str, Any]]:
    work = pd.DataFrame({"value": values, "group": groups}).dropna()
    work = work[work["group"].isin(["none_to_low", "mild_to_moderate", "high"])]
    if work["group"].nunique() < 2:
        return []
    values_arr = work["value"].to_numpy(dtype=float)
    ranks = rankdata(values_arr, method="average")
    work = work.assign(rank=ranks)
    n_total = len(work)
    _, tie_counts = np.unique(values_arr, return_counts=True)
    if n_total > 1:
        tie_correction = 1.0 - float(np.sum(tie_counts**3 - tie_counts)) / float(n_total**3 - n_total)
    else:
        tie_correction = np.nan
    if not np.isfinite(tie_correction) or tie_correction <= 0:
        tie_correction = 1.0
    base_var = n_total * (n_total + 1.0) / 12.0 * tie_correction
    order = ["none_to_low", "mild_to_moderate", "high"]
    rows = []
    for i, group_a in enumerate(order):
        for group_b in order[i + 1 :]:
            a = work[work["group"] == group_a]
            b = work[work["group"] == group_b]
            if len(a) == 0 or len(b) == 0:
                continue
            mean_rank_a = float(a["rank"].mean())
            mean_rank_b = float(b["rank"].mean())
            se = np.sqrt(base_var * (1.0 / len(a) + 1.0 / len(b)))
            z_value = (mean_rank_a - mean_rank_b) / se if se > 0 else np.nan
            p_value = float(2.0 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
            rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": int(len(a)),
                    "n_b": int(len(b)),
                    "median_a": float(a["value"].median()),
                    "median_b": float(b["value"].median()),
                    "mean_rank_a": mean_rank_a,
                    "mean_rank_b": mean_rank_b,
                    "z_dunn": float(z_value) if np.isfinite(z_value) else np.nan,
                    "p_dunn": p_value,
                }
            )
    return rows


def misc_kruskal_dunn_tests(lap_df: pd.DataFrame, stat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = lap_df[lap_df["misc_3class_label"].isin(["none_to_low", "mild_to_moderate", "high"])].copy()
    kruskal_rows: list[dict[str, Any]] = []
    dunn_rows: list[dict[str, Any]] = []
    for feature in stat_cols:
        item = _parse_channel_feature(feature)
        groups = []
        counts: dict[str, int] = {}
        medians: dict[str, float] = {}
        for label in ["none_to_low", "mild_to_moderate", "high"]:
            values = pd.to_numeric(work.loc[work["misc_3class_label"] == label, feature], errors="coerce").dropna()
            counts[label] = int(len(values))
            medians[label] = float(values.median()) if len(values) else np.nan
            if len(values):
                groups.append(values.to_numpy(dtype=float))
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            try:
                test = kruskal(*groups)
                h_stat = float(test.statistic)
                p_value = float(test.pvalue)
            except Exception:
                h_stat, p_value = np.nan, np.nan
        else:
            h_stat, p_value = np.nan, np.nan
        row = {
            **item,
            "analysis": "lap_level_misc_3class_kruskal",
            "n_laps": int(sum(counts.values())),
            "n_none_to_low": counts.get("none_to_low", 0),
            "n_mild_to_moderate": counts.get("mild_to_moderate", 0),
            "n_high": counts.get("high", 0),
            "median_none_to_low": medians.get("none_to_low", np.nan),
            "median_mild_to_moderate": medians.get("mild_to_moderate", np.nan),
            "median_high": medians.get("high", np.nan),
            "h_statistic": h_stat,
            "p_kruskal": p_value,
        }
        kruskal_rows.append(row)
        for posthoc in _dunn_pairwise(pd.to_numeric(work[feature], errors="coerce"), work["misc_3class_label"]):
            dunn_rows.append({**item, "analysis": "lap_level_misc_3class_dunn", **posthoc})

    kruskal_out = pd.DataFrame(kruskal_rows)
    if not kruskal_out.empty:
        kruskal_out["q_kruskal_bh"] = _bh_fdr(kruskal_out["p_kruskal"])
        kruskal_out = kruskal_out.sort_values(["q_kruskal_bh", "p_kruskal", "feature"], na_position="last")

    dunn_out = pd.DataFrame(dunn_rows)
    if not dunn_out.empty:
        dunn_out["q_dunn_bh"] = _bh_fdr(dunn_out["p_dunn"])
        dunn_out = dunn_out.sort_values(["q_dunn_bh", "p_dunn", "feature", "group_a", "group_b"], na_position="last")
    return kruskal_out, dunn_out


def misc_fisher_entity_ranking(kruskal_tests: pd.DataFrame) -> pd.DataFrame:
    if kruskal_tests.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for entity, group in kruskal_tests.groupby("entity", dropna=False):
        p_values = pd.to_numeric(group["p_kruskal"], errors="coerce").dropna().to_numpy(dtype=float)
        p_values = p_values[np.isfinite(p_values)]
        if len(p_values) == 0:
            continue
        fisher_stat, combined_p = combine_pvalues(p_values, method="fisher")
        n_laps = pd.to_numeric(group["n_laps"], errors="coerce")
        h_values = pd.to_numeric(group["h_statistic"], errors="coerce")
        effect_proxy = (h_values / (n_laps - 1.0)).replace([np.inf, -np.inf], np.nan)
        q_feature = pd.to_numeric(group["q_kruskal_bh"], errors="coerce")
        rows.append(
            {
                "entity": entity,
                "level": "channel",
                "feature_count": int(len(group)),
                "mean_p_value": float(np.mean(p_values)),
                "fisher_statistic": float(fisher_stat),
                "combined_p_fisher": float(combined_p),
                "mean_h_statistic": float(h_values.mean()),
                "mean_kw_effect_proxy": float(effect_proxy.mean()),
                "median_kw_effect_proxy": float(effect_proxy.median()),
                "features_q_lt_0_05": int((q_feature < 0.05).sum()),
                "mean_neglog10_feature_p": float(
                    (-np.log10(np.clip(p_values, 1e-300, 1.0))).mean()
                ),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["combined_fdr_bh"] = _bh_fdr(out["combined_p_fisher"])
    out["neglog10_mean_p"] = -np.log10(np.clip(out["mean_p_value"].to_numpy(dtype=float), 1e-300, 1.0))
    out["neglog10_combined_p"] = -np.log10(np.clip(out["combined_p_fisher"].to_numpy(dtype=float), 1e-300, 1.0))
    out["neglog10_combined_bh_p"] = -np.log10(np.clip(out["combined_fdr_bh"].to_numpy(dtype=float), 1e-300, 1.0))
    out["fisher_evidence_score"] = out["neglog10_combined_p"]
    out = out.sort_values(
        ["combined_fdr_bh", "combined_p_fisher", "entity"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    out.insert(0, "overall_rank", np.arange(1, len(out) + 1))
    return out


def misc_threshold_sensitivity_entities(lap_df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    score = pd.to_numeric(lap_df["misc_score"], errors="coerce")
    schemes = {
        "0-2_3-5_6-10": [(-np.inf, 2, "low"), (3, 5, "mid"), (6, np.inf, "high")],
        "0-1_2-4_5-10": [(-np.inf, 1, "low"), (2, 4, "mid"), (5, np.inf, "high")],
    }
    entity_rows: list[dict[str, Any]] = []
    for scheme, bins in schemes.items():
        labels = pd.Series(np.nan, index=lap_df.index, dtype=object)
        for lower, upper, label in bins:
            labels.loc[(score >= lower) & (score <= upper)] = label
        feature_rows: list[dict[str, Any]] = []
        for feature in stat_cols:
            groups = [
                pd.to_numeric(lap_df.loc[labels == label, feature], errors="coerce").dropna().to_numpy(dtype=float)
                for _, _, label in bins
            ]
            if not all(len(group) > 0 for group in groups):
                continue
            try:
                result = kruskal(*groups)
                p_value = float(result.pvalue)
                h_stat = float(result.statistic)
            except Exception:
                p_value, h_stat = np.nan, np.nan
            item = _parse_channel_feature(feature)
            item.update({"feature": feature, "p_kruskal": p_value, "h_statistic": h_stat, "n_laps": int(sum(len(g) for g in groups))})
            feature_rows.append(item)
        feature_df = pd.DataFrame(feature_rows)
        if feature_df.empty:
            continue
        feature_df["q_kruskal_bh"] = _bh_fdr(feature_df["p_kruskal"])
        for entity, group in feature_df.groupby("entity", dropna=False):
            p_values = pd.to_numeric(group["p_kruskal"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(p_values) == 0:
                continue
            fisher_stat, combined_p = combine_pvalues(p_values, method="fisher")
            entity_rows.append(
                {
                    "scheme": scheme,
                    "entity": entity,
                    "level": "channel",
                    "feature_count": int(len(group)),
                    "mean_p_value": float(np.mean(p_values)),
                    "fisher_statistic": float(fisher_stat),
                    "combined_p_fisher": float(combined_p),
                    "features_q_lt_0_05": int((group["q_kruskal_bh"] < 0.05).sum()),
                }
            )
    out = pd.DataFrame(entity_rows)
    if out.empty:
        return out
    out["combined_fdr_bh"] = out.groupby("scheme", group_keys=False)["combined_p_fisher"].transform(_bh_fdr)
    out["neglog10_mean_p"] = -np.log10(np.clip(out["mean_p_value"].to_numpy(dtype=float), 1e-300, 1.0))
    out["neglog10_combined_p"] = -np.log10(np.clip(out["combined_p_fisher"].to_numpy(dtype=float), 1e-300, 1.0))
    out["neglog10_combined_bh_p"] = -np.log10(np.clip(out["combined_fdr_bh"].to_numpy(dtype=float), 1e-300, 1.0))
    out["fisher_evidence_score"] = out["neglog10_combined_p"]
    out = out.sort_values(["scheme", "fisher_evidence_score", "combined_p_fisher"], ascending=[True, False, True]).reset_index(drop=True)
    out.insert(0, "scheme_rank", out.groupby("scheme").cumcount() + 1)
    return out


def beta_validation_representative_channels(
    kruskal_tests: pd.DataFrame,
    entity_ranking: pd.DataFrame,
    n: int = 3,
) -> pd.DataFrame:
    if kruskal_tests.empty or entity_ranking.empty:
        return pd.DataFrame()
    if not {"band", "feature_family", "measure"}.issubset(kruskal_tests.columns):
        return pd.DataFrame()
    required_rank_cols = {"entity", "level", "combined_p_fisher", "combined_fdr_bh"}
    if not required_rank_cols.issubset(entity_ranking.columns):
        return pd.DataFrame()

    channels = entity_ranking[entity_ranking["level"].eq("channel")].copy()
    channels["combined_p_fisher"] = pd.to_numeric(channels["combined_p_fisher"], errors="coerce")
    channels["combined_fdr_bh"] = pd.to_numeric(channels["combined_fdr_bh"], errors="coerce")
    channels = channels.sort_values(
        ["combined_fdr_bh", "combined_p_fisher", "entity"],
        na_position="last",
    ).drop_duplicates("entity")
    channels = channels.head(n).copy()
    if channels.empty:
        return pd.DataFrame()
    channels.insert(0, "channel_rank_fisher", np.arange(1, len(channels) + 1))

    beta_psd = kruskal_tests[
        kruskal_tests["level"].eq("channel")
        & kruskal_tests["band"].eq("beta")
        & kruskal_tests["feature_family"].eq("psd")
        & kruskal_tests["measure"].eq("absolute_power")
    ].copy()
    if beta_psd.empty:
        return pd.DataFrame()
    beta_psd = beta_psd.drop_duplicates("entity")
    out = channels[["entity", "channel_rank_fisher", "combined_p_fisher", "combined_fdr_bh"]].merge(
        beta_psd,
        on="entity",
        how="inner",
        suffixes=("_channel", ""),
    )
    if out.empty:
        return out
    out = out.sort_values("channel_rank_fisher").reset_index(drop=True)
    out.insert(0, "beta_rank", np.arange(1, len(out) + 1))
    return out


def susceptibility_group_tests(lap_df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    subject_df = (
        lap_df.groupby(["subject_id", "susceptibility_group"], dropna=False)[stat_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for feature in stat_cols:
        susceptible = subject_df.loc[subject_df["susceptibility_group"] == "susceptible", feature].dropna().to_numpy(dtype=float)
        non_susceptible = subject_df.loc[subject_df["susceptibility_group"] == "non_susceptible", feature].dropna().to_numpy(dtype=float)
        item = _parse_channel_feature(feature)
        if len(susceptible) and len(non_susceptible):
            try:
                test = mannwhitneyu(susceptible, non_susceptible, alternative="two-sided")
                u_stat = float(test.statistic)
                p_value = float(test.pvalue)
            except Exception:
                u_stat, p_value = np.nan, np.nan
        else:
            u_stat, p_value = np.nan, np.nan
        item.update(
            {
                "analysis": "subject_level_susceptibility_mannwhitney",
                "n_susceptible": int(len(susceptible)),
                "n_non_susceptible": int(len(non_susceptible)),
                "susceptible_median": float(np.median(susceptible)) if len(susceptible) else np.nan,
                "non_susceptible_median": float(np.median(non_susceptible)) if len(non_susceptible) else np.nan,
                "u_statistic": u_stat,
                "rank_biserial": _rank_biserial_from_u(u_stat, len(susceptible), len(non_susceptible)),
                "p_mannwhitney": p_value,
            }
        )
        rows.append(item)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_mannwhitney_bh"] = _bh_fdr(out["p_mannwhitney"])
        out = out.sort_values(["q_mannwhitney_bh", "p_mannwhitney", "feature"], na_position="last")
    return out


def run_statistics(config: dict[str, Any], features_path: Path, output_root: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(features_path, dtype={"subject_id": str})
    stat_cols = _stat_columns(df)
    out_dir = output_root / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    lap_df = _lap_level(df, stat_cols)
    lap_df.to_csv(out_dir / "lap_level_channel_features.csv", index=False, encoding="utf-8-sig")

    subject_condition = (
        lap_df.groupby(["subject_id", "susceptibility_group", "condition_label"], dropna=False)[stat_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    subject_condition.to_csv(out_dir / "subject_condition_channel_features.csv", index=False, encoding="utf-8-sig")

    subject_features = (
        lap_df.groupby(["subject_id", "susceptibility_group"], dropna=False)[stat_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    subject_features.to_csv(out_dir / "subject_level_channel_features.csv", index=False, encoding="utf-8-sig")

    condition_tests = condition_paired_tests(lap_df, stat_cols)
    misc_tests = misc_spearman_tests(lap_df, stat_cols)
    misc_kruskal_tests, misc_dunn_tests = misc_kruskal_dunn_tests(lap_df, stat_cols)
    misc_entity_ranking = misc_fisher_entity_ranking(misc_kruskal_tests)
    misc_threshold_sensitivity = misc_threshold_sensitivity_entities(lap_df, stat_cols)
    beta_psd_all_entities = misc_kruskal_tests[
        misc_kruskal_tests["level"].eq("channel")
        & misc_kruskal_tests["band"].eq("beta")
        & misc_kruskal_tests["feature_family"].eq("psd")
        & misc_kruskal_tests["measure"].eq("absolute_power")
    ].copy()
    if not beta_psd_all_entities.empty:
        beta_psd_all_entities = beta_psd_all_entities.sort_values(
            ["q_kruskal_bh", "p_kruskal", "entity"],
            na_position="last",
        ).reset_index(drop=True)
    beta_representative = beta_validation_representative_channels(
        misc_kruskal_tests,
        misc_entity_ranking,
        n=3,
    )
    susceptibility_tests = susceptibility_group_tests(lap_df, stat_cols)
    quality_reports = run_quality_report(config, output_root)

    condition_tests.to_csv(out_dir / "condition_paired_tests.csv", index=False, encoding="utf-8-sig")
    misc_tests.to_csv(out_dir / "misc_spearman_tests.csv", index=False, encoding="utf-8-sig")
    misc_kruskal_tests.to_csv(out_dir / "misc_kruskal_tests.csv", index=False, encoding="utf-8-sig")
    misc_dunn_tests.to_csv(out_dir / "misc_dunn_posthoc_tests.csv", index=False, encoding="utf-8-sig")
    misc_entity_ranking.to_csv(out_dir / "misc_fisher_entity_ranking.csv", index=False, encoding="utf-8-sig")
    misc_threshold_sensitivity.to_csv(out_dir / "misc_threshold_sensitivity_entities.csv", index=False, encoding="utf-8-sig")
    beta_psd_all_entities.to_csv(out_dir / "misc_beta_psd_all_entities.csv", index=False, encoding="utf-8-sig")
    beta_representative.to_csv(out_dir / "misc_beta_validation_representative.csv", index=False, encoding="utf-8-sig")
    susceptibility_tests.to_csv(out_dir / "susceptibility_group_tests.csv", index=False, encoding="utf-8-sig")

    evidence = misc_kruskal_tests[["feature", "entity", "level", "feature_family", "band", "measure", "h_statistic", "p_kruskal", "q_kruskal_bh"]].copy()
    if not misc_dunn_tests.empty:
        dunn_min = misc_dunn_tests.groupby("feature", as_index=False)["q_dunn_bh"].min().rename(columns={"q_dunn_bh": "best_dunn_q"})
        evidence = evidence.merge(dunn_min, on="feature", how="left")
    evidence = evidence.merge(misc_tests[["feature", "rho", "p_spearman", "q_spearman_bh"]], on="feature", how="left")
    evidence = evidence.merge(susceptibility_tests[["feature", "rank_biserial", "p_mannwhitney", "q_mannwhitney_bh"]], on="feature", how="left")
    evidence = evidence.merge(condition_tests[["feature", "mean_diff", "cohens_dz", "p_signflip", "q_signflip_bh"]], on="feature", how="left")
    evidence = evidence.sort_values(["q_kruskal_bh", "best_dunn_q", "feature"], na_position="last")
    evidence.to_csv(out_dir / "representative_feature_evidence.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "analysis": "condition_paired_fast_minus_normal",
                "unit": "subject_condition_mean",
                "features_tested": len(stat_cols),
                "p_value": "p_signflip",
                "fdr_value": "q_signflip_bh",
                "significant_q_lt_0_05": int((condition_tests.get("q_signflip_bh", pd.Series(dtype=float)) < 0.05).sum()),
            },
            {
                "analysis": "lap_level_misc_spearman",
                "unit": "lap_mean",
                "features_tested": len(stat_cols),
                "p_value": "p_spearman",
                "fdr_value": "q_spearman_bh",
                "significant_q_lt_0_05": int((misc_tests.get("q_spearman_bh", pd.Series(dtype=float)) < 0.05).sum()),
            },
            {
                "analysis": "subject_level_susceptibility_mannwhitney",
                "unit": "subject_mean",
                "features_tested": len(stat_cols),
                "p_value": "p_mannwhitney",
                "fdr_value": "q_mannwhitney_bh",
                "significant_q_lt_0_05": int((susceptibility_tests.get("q_mannwhitney_bh", pd.Series(dtype=float)) < 0.05).sum()),
            },
            {
                "analysis": "lap_level_misc_3class_kruskal",
                "unit": "lap_mean",
                "features_tested": len(stat_cols),
                "p_value": "p_kruskal",
                "fdr_value": "q_kruskal_bh",
                "significant_q_lt_0_05": int((misc_kruskal_tests.get("q_kruskal_bh", pd.Series(dtype=float)) < 0.05).sum()),
            },
            {
                "analysis": "lap_level_misc_3class_fisher_entity",
                "unit": "lap_mean_entity_aggregate",
                "features_tested": len(stat_cols),
                "entities_tested": int(len(misc_entity_ranking)),
                "p_value": "combined_p_fisher",
                "fdr_value": "combined_fdr_bh",
                "significant_q_lt_0_05": int((misc_entity_ranking.get("combined_fdr_bh", pd.Series(dtype=float)) < 0.05).sum()),
            },
            {
                "analysis": "lap_level_misc_3class_beta_absolute_psd",
                "unit": "lap_mean_entity_feature",
                "features_tested": int(len(beta_psd_all_entities)),
                "entities_tested": int(beta_psd_all_entities["entity"].nunique()) if not beta_psd_all_entities.empty else 0,
                "p_value": "p_kruskal",
                "fdr_value": "q_kruskal_bh",
                "significant_q_lt_0_05": int((beta_psd_all_entities.get("q_kruskal_bh", pd.Series(dtype=float)) < 0.05).sum()),
            },
            *[
                {
                    "analysis": f"lap_level_misc_threshold_sensitivity_{scheme}",
                    "unit": "lap_mean_entity_aggregate",
                    "features_tested": len(stat_cols),
                    "entities_tested": int((misc_threshold_sensitivity.get("scheme", pd.Series(dtype=str)) == scheme).sum()),
                    "p_value": "combined_p_fisher",
                    "fdr_value": "combined_fdr_bh",
                    "significant_q_lt_0_05": int(
                        (
                            misc_threshold_sensitivity.loc[
                                misc_threshold_sensitivity.get("scheme", pd.Series(dtype=str)) == scheme,
                                "combined_fdr_bh",
                            ]
                            < 0.05
                        ).sum()
                    ),
                }
                for scheme in misc_threshold_sensitivity.get("scheme", pd.Series(dtype=str)).dropna().unique()
            ],
            {
                "analysis": "lap_level_misc_3class_dunn",
                "unit": "lap_mean_pairwise",
                "features_tested": int(len(misc_dunn_tests)),
                "p_value": "p_dunn",
                "fdr_value": "q_dunn_bh",
                "significant_q_lt_0_05": int((misc_dunn_tests.get("q_dunn_bh", pd.Series(dtype=float)) < 0.05).sum()),
            },
        ]
    )
    summary.to_csv(out_dir / "statistics_summary.csv", index=False, encoding="utf-8-sig")

    return {
        "lap_level": lap_df,
        "subject_condition": subject_condition,
        "subject_level": subject_features,
        "condition_tests": condition_tests,
        "misc_tests": misc_tests,
        "misc_kruskal_tests": misc_kruskal_tests,
        "misc_dunn_tests": misc_dunn_tests,
        "misc_entity_ranking": misc_entity_ranking,
        "misc_threshold_sensitivity": misc_threshold_sensitivity,
        "beta_psd_all_entities": beta_psd_all_entities,
        "beta_representative": beta_representative,
        "susceptibility_tests": susceptibility_tests,
        "quality_reports": quality_reports,
        "summary": summary,
    }
