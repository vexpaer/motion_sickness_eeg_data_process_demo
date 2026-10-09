from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

from .features import metadata_columns


class OrdinalThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator: Any | None = None, class_weight: str | None = "balanced"):
        self.base_estimator = base_estimator
        self.class_weight = class_weight

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalThresholdClassifier":
        self.classes_ = np.sort(np.unique(y))
        base = self.base_estimator or LogisticRegression(max_iter=1000, class_weight=self.class_weight)
        self.models_ = []
        for threshold in self.classes_[:-1]:
            binary = (y > threshold).astype(int)
            model = clone(base)
            model.fit(X, binary)
            self.models_.append((threshold, model))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(self.classes_) == 1:
            return np.ones((X.shape[0], 1))
        cumulative = []
        for _, model in self.models_:
            cumulative.append(model.predict_proba(X)[:, 1])
        cumulative = np.vstack(cumulative).T
        cumulative = np.sort(cumulative, axis=1)[:, ::-1]
        probs = []
        probs.append(1.0 - cumulative[:, 0])
        for idx in range(1, cumulative.shape[1]):
            probs.append(np.maximum(cumulative[:, idx - 1] - cumulative[:, idx], 0.0))
        probs.append(cumulative[:, -1])
        out = np.vstack(probs).T
        row_sum = out.sum(axis=1, keepdims=True)
        return np.divide(out, row_sum, out=np.zeros_like(out), where=row_sum > 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def _models(random_state: int, balance_method: str = "class_weight") -> dict[str, Any]:
    if balance_method not in {"class_weight", "none"}:
        raise ValueError(f"Unsupported balance_method: {balance_method}")
    class_weight = "balanced" if balance_method == "class_weight" else None
    forest_weight = "balanced_subsample" if balance_method == "class_weight" else None
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight=class_weight, solver="lbfgs"),
        "SVM_RBF": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight=class_weight,
            probability=True,
            random_state=random_state,
            decision_function_shape="ovr",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight=forest_weight,
            random_state=random_state,
            n_jobs=-1,
            max_features="sqrt",
        ),
    }


def _make_pipeline(model: Any, select_k: int, n_features: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(select_k, n_features))),
            ("model", model),
        ]
    )


def _feature_columns(df: pd.DataFrame) -> list[str]:
    meta = metadata_columns()
    columns = []
    for col in df.columns:
        if col in meta or col.startswith("qc__") or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        columns.append(col)
    return columns


def _group_values(df: pd.DataFrame, group_level: str) -> pd.Series:
    if group_level == "subject":
        return df["subject_id"].astype(str)
    if group_level == "lap":
        return df["lap_group"].astype(str)
    if group_level == "session":
        return df["session_id"].astype(str)
    raise ValueError(f"Unknown group_level: {group_level}")


def _safe_auc(y_true: np.ndarray, prob: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    if len(classes) < 2:
        return {"auc_macro_ovr": np.nan, "auc_micro_ovr": np.nan}
    try:
        if len(classes) == 2:
            out["auc_macro_ovr"] = float(roc_auc_score(y_true, prob[:, 1]))
            out["auc_micro_ovr"] = out["auc_macro_ovr"]
        else:
            y_bin = label_binarize(y_true, classes=classes)
            out["auc_macro_ovr"] = float(roc_auc_score(y_bin, prob, average="macro", multi_class="ovr"))
            out["auc_micro_ovr"] = float(roc_auc_score(y_bin, prob, average="micro", multi_class="ovr"))
            for idx, cls in enumerate(classes):
                if len(np.unique(y_bin[:, idx])) > 1:
                    out[f"auc_class_{cls}"] = float(roc_auc_score(y_bin[:, idx], prob[:, idx]))
                else:
                    out[f"auc_class_{cls}"] = np.nan
    except Exception:
        out["auc_macro_ovr"] = np.nan
        out["auc_micro_ovr"] = np.nan
    return out


def _lap_level_predictions(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    prob: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    frame = metadata.reset_index(drop=True).copy()
    frame["y_true"] = np.asarray(y_true, dtype=int)
    for idx, cls in enumerate(classes):
        frame[f"prob_class_{cls}"] = prob[:, idx]
    rows: list[dict[str, Any]] = []
    for lap_group, group in frame.groupby("lap_group", sort=False):
        row: dict[str, Any] = {
            "lap_group": str(lap_group),
            "subject_id": str(group["subject_id"].iloc[0]),
            "y_true": int(group["y_true"].iloc[0]),
        }
        for cls in classes:
            row[f"prob_class_{cls}"] = float(group[f"prob_class_{cls}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_bootstrap_auc(
    lap_predictions: pd.DataFrame,
    classes: np.ndarray,
    random_state: int,
    n_bootstrap: int = 2000,
) -> tuple[dict[str, float], pd.DataFrame]:
    subject_values = lap_predictions["subject_id"].astype(str).to_numpy()
    subjects = np.unique(subject_values)
    subject_indices = {subject: np.flatnonzero(subject_values == subject) for subject in subjects}
    y_all = lap_predictions["y_true"].to_numpy(dtype=int)
    p_all = lap_predictions[[f"prob_class_{cls}" for cls in classes]].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        sampled_indices = np.concatenate([subject_indices[subject] for subject in sampled_subjects])
        y = y_all[sampled_indices]
        p = p_all[sampled_indices]
        binary_values: dict[str, float] = {}
        for idx, cls in enumerate(classes):
            target = (y == cls).astype(int)
            if len(np.unique(target)) == 2:
                binary_values[f"auc_class_{cls}"] = float(roc_auc_score(target, p[:, idx]))
        if len(binary_values) == len(classes):
            binary_values["auc_macro_ovr"] = float(np.mean(list(binary_values.values())))
            if len(classes) == 2:
                binary_values["auc_micro_ovr"] = float(roc_auc_score((y == classes[1]).astype(int), p[:, 1]))
            else:
                y_bin = label_binarize(y, classes=classes)
                binary_values["auc_micro_ovr"] = float(
                    roc_auc_score(y_bin, p, average="micro", multi_class="ovr")
                )
            rows.append(binary_values)

    bootstrap = pd.DataFrame(rows)
    summary: dict[str, float] = {"auc_bootstrap_n": float(len(bootstrap)), "auc_bootstrap_replicates": float(n_bootstrap)}
    if bootstrap.empty:
        return summary, bootstrap
    for column in bootstrap.columns:
        summary[f"{column}_ci_lower"] = float(bootstrap[column].quantile(0.025))
        summary[f"{column}_ci_upper"] = float(bootstrap[column].quantile(0.975))
    return summary, bootstrap


def _calibration_metrics(y_true: np.ndarray, prob: np.ndarray, classes: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-12, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)
    one_hot = np.zeros_like(prob)
    for idx, cls in enumerate(classes):
        one_hot[:, idx] = (y_true == cls).astype(float)
    confidence = prob.max(axis=1)
    correct = (classes[np.argmax(prob, axis=1)] == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        mask = (confidence >= edges[idx]) & (confidence <= edges[idx + 1] if idx == n_bins - 1 else confidence < edges[idx + 1])
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return {
        "log_loss": float(log_loss(y_true, prob, labels=classes)),
        "brier_multiclass": float(np.mean(np.sum((prob - one_hot) ** 2, axis=1))),
        "ece_top_label": float(ece),
    }


def _calibration_table(y_true: np.ndarray, prob: np.ndarray, classes: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for idx, cls in enumerate(classes):
        target = (y_true == cls).astype(int)
        score = np.asarray(prob[:, idx], dtype=float)
        for bin_idx in range(n_bins):
            mask = (score >= edges[bin_idx]) & (score <= edges[bin_idx + 1] if bin_idx == n_bins - 1 else score < edges[bin_idx + 1])
            rows.append(
                {
                    "class": int(cls),
                    "bin": bin_idx + 1,
                    "lower": float(edges[bin_idx]),
                    "upper": float(edges[bin_idx + 1]),
                    "n": int(mask.sum()),
                    "mean_predicted_probability": float(score[mask].mean()) if np.any(mask) else np.nan,
                    "observed_frequency": float(target[mask].mean()) if np.any(mask) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray, title: str, output: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=[str(c) for c in classes]).plot(ax=ax, cmap="Blues", values_format=".2f")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_roc(y_true: np.ndarray, prob: np.ndarray, classes: np.ndarray, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, prob[:, 1], pos_label=classes[1])
        ax.plot(fpr, tpr, label=f"class {classes[1]}")
    else:
        y_bin = label_binarize(y_true, classes=classes)
        for idx, cls in enumerate(classes):
            if len(np.unique(y_bin[:, idx])) > 1:
                fpr, tpr, _ = roc_curve(y_bin[:, idx], prob[:, idx])
                ax.plot(fpr, tpr, label=f"class {cls}")
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _parse_selected_feature(feature: str) -> dict[str, str]:
    parts = str(feature).split("__")
    entity = parts[0] if parts else ""
    family = parts[1] if len(parts) > 1 else ""
    metric = parts[2] if len(parts) > 2 else ""
    level = "channel"
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
        measure = metric.removesuffix("_mean")
    else:
        measure = metric or base_family
    return {
        "feature": str(feature),
        "entity": entity,
        "level": level,
        "feature_family": base_family,
        "band": band,
        "measure": measure,
    }


def evaluate_label(
    df: pd.DataFrame,
    label_col: str,
    output_dir: Path,
    group_level: str,
    select_k: int,
    random_state: int,
    balance_method: str = "class_weight",
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    work = df[df[label_col].notna() & (df[label_col] >= 0)].copy()
    features = _feature_columns(work)
    X = work[features].to_numpy(dtype=float)
    y = work[label_col].to_numpy(dtype=int)
    groups = _group_values(work, group_level).to_numpy()
    classes = np.sort(np.unique(y))
    metadata = work[["sample_id", "subject_id", "session_id", "lap_group", label_col]].copy()
    metrics = []
    feature_counts: dict[str, int] = {}

    for model_name, model in _models(random_state, balance_method).items():
        logo = LeaveOneGroupOut()
        y_pred = np.full_like(y, fill_value=-1)
        prob = np.zeros((len(y), len(classes)), dtype=float)
        fold_rows = []

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
            if len(np.unique(y[train_idx])) < 2:
                continue
            pipe = _make_pipeline(model, select_k, X.shape[1])
            pipe.fit(X[train_idx], y[train_idx])
            pred = pipe.predict(X[test_idx])
            fold_prob_small = pipe.predict_proba(X[test_idx])
            model_classes = pipe.named_steps["model"].classes_
            y_pred[test_idx] = pred
            for small_idx, cls in enumerate(model_classes):
                full_idx = int(np.where(classes == cls)[0][0])
                prob[test_idx, full_idx] = fold_prob_small[:, small_idx]

            support = pipe.named_steps["variance"].get_support()
            after_variance = np.array(features)[support]
            selected = after_variance[pipe.named_steps["select"].get_support()]
            for feature in selected:
                feature = str(feature)
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
            fold_rows.append(
                {
                    "fold": fold,
                    "held_out_group": groups[test_idx][0],
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "train_classes": ",".join(map(str, np.unique(y[train_idx]))),
                    "test_classes": ",".join(map(str, np.unique(y[test_idx]))),
                    "balance_method": balance_method,
                }
            )

        valid = y_pred >= 0
        if int(valid.sum()) == 0:
            metrics.append(
                {
                    "label": label_col,
                    "model": model_name,
                    "n_samples": 0,
                    "n_groups": int(len(np.unique(groups))),
                    "group_level": group_level,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "precision_macro": np.nan,
                    "recall_macro": np.nan,
                    "f1_macro": np.nan,
                    "precision_weighted": np.nan,
                    "recall_weighted": np.nan,
                    "f1_weighted": np.nan,
                    "auc_macro_ovr": np.nan,
                    "auc_micro_ovr": np.nan,
                    "balance_method": balance_method,
                    "notes": "no fold had at least two training classes",
                }
            )
            continue
        y_true = y[valid]
        pred_valid = y_pred[valid]
        prob_valid = prob[valid]
        valid_metadata = metadata.loc[valid].reset_index(drop=True)
        metric = {
            "label": label_col,
            "model": model_name,
            "balance_method": balance_method,
            "n_samples": int(valid.sum()),
            "n_groups": int(len(np.unique(groups))),
            "group_level": group_level,
            "accuracy": float(accuracy_score(y_true, pred_valid)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred_valid)),
            "precision_macro": float(precision_score(y_true, pred_valid, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, pred_valid, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, pred_valid, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, pred_valid, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, pred_valid, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, pred_valid, average="weighted", zero_division=0)),
        }
        metric.update(_safe_auc(y_true, prob_valid, classes))
        lap_predictions = _lap_level_predictions(valid_metadata, y_true, prob_valid, classes)
        lap_y_true = lap_predictions["y_true"].to_numpy(dtype=int)
        lap_prob = lap_predictions[[f"prob_class_{cls}" for cls in classes]].to_numpy(dtype=float)
        metric.update({f"{key}_lap": value for key, value in _safe_auc(lap_y_true, lap_prob, classes).items()})
        metric.update(_calibration_metrics(lap_y_true, lap_prob, classes))
        metric["calibration_unit"] = "lap"
        metric["calibration_n_laps"] = int(len(lap_predictions))
        auc_ci, bootstrap_rows = _cluster_bootstrap_auc(
            lap_predictions,
            classes,
            random_state=random_state + sum(ord(ch) for ch in model_name) + len(label_col),
        )
        metric.update(auc_ci)
        metric["auc_ci_unit"] = "participant-cluster bootstrap over lap-averaged probabilities"
        bootstrap_rows.to_csv(
            output_dir / f"auc_bootstrap_{label_col}_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _calibration_table(lap_y_true, lap_prob, classes).to_csv(
            output_dir / f"calibration_{label_col}_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        precision_by_class, recall_by_class, f1_by_class, support_by_class = precision_recall_fscore_support(
            y_true, pred_valid, labels=classes, zero_division=0
        )
        for idx, cls in enumerate(classes):
            metric[f"support_class_{cls}"] = int(support_by_class[idx])
            metric[f"precision_class_{cls}"] = float(precision_by_class[idx])
            metric[f"recall_class_{cls}"] = float(recall_by_class[idx])
            metric[f"f1_class_{cls}"] = float(f1_by_class[idx])
        metrics.append(metric)

        oof = metadata.loc[valid].copy()
        oof["model"] = model_name
        oof["y_true"] = y_true
        oof["y_pred"] = pred_valid
        for idx, cls in enumerate(classes):
            oof[f"prob_class_{cls}"] = prob_valid[:, idx]
        oof.to_csv(output_dir / f"oof_{label_col}_{model_name}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fold_rows).to_csv(output_dir / f"folds_{label_col}_{model_name}.csv", index=False, encoding="utf-8-sig")
        _plot_confusion(y_true, pred_valid, classes, f"{label_col} - {model_name}", output_dir / f"confusion_{label_col}_{model_name}.png")
        _plot_roc(y_true, prob_valid, classes, f"{label_col} - {model_name}", output_dir / f"roc_{label_col}_{model_name}.png")

    if feature_counts:
        rows = []
        for feature, count in sorted(feature_counts.items(), key=lambda x: (-x[1], x[0])):
            item = _parse_selected_feature(feature)
            item["selected_count"] = count
            rows.append(item)
        pd.DataFrame(rows).to_csv(
            output_dir / f"feature_selection_counts_{label_col}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    result = pd.DataFrame(metrics)
    result.to_csv(output_dir / f"metrics_{label_col}.csv", index=False, encoding="utf-8-sig")
    return result


def evaluate_ordinal_misc(
    df: pd.DataFrame,
    output_dir: Path,
    group_level: str,
    select_k: int,
    balance_method: str = "class_weight",
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    work = df[df["misc_score"].notna() & (df["misc_score"] >= 0)].copy()
    features = _feature_columns(work)
    X = work[features].to_numpy(dtype=float)
    y = work["misc_score"].to_numpy(dtype=int)
    groups = _group_values(work, group_level).to_numpy()
    classes = np.sort(np.unique(y))
    logo = LeaveOneGroupOut()
    pred = np.full_like(y, -1)

    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=min(select_k, X.shape[1]))),
                (
                    "model",
                    OrdinalThresholdClassifier(
                        class_weight="balanced" if balance_method == "class_weight" else None
                    ),
                ),
            ]
        )
        pipe.fit(X[train_idx], y[train_idx])
        pred[test_idx] = pipe.predict(X[test_idx])

    valid = pred >= 0
    rho = spearmanr(y[valid], pred[valid]).correlation if valid.any() else np.nan
    mae = float(mean_absolute_error(y[valid], pred[valid])) if valid.any() else np.nan
    result = pd.DataFrame(
        [
            {
                "label": "ordinal_misc_score",
                "model": "OrdinalThresholdClassifier",
                "analysis": "ordinal_misc_score",
                "balance_method": balance_method,
                "n_samples": int(valid.sum()),
                "n_groups": int(len(np.unique(groups))),
                "group_level": group_level,
                "mae": mae,
                "spearman_r": float(rho) if rho == rho else np.nan,
                "classes": ",".join(map(str, classes)),
            }
        ]
    )
    result.to_csv(output_dir / "metrics_ordinal_misc_score.csv", index=False, encoding="utf-8-sig")
    oof = work.loc[valid, ["sample_id", "subject_id", "session_id", "lap_group", "misc_score"]].copy()
    oof["y_pred"] = pred[valid]
    oof.to_csv(output_dir / "oof_ordinal_misc_score.csv", index=False, encoding="utf-8-sig")
    return result


def write_lap_auc_summary(output_dir: Path) -> pd.DataFrame:
    model_names = ("LogisticRegression", "RandomForest", "SVM_RBF")
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("oof_*.csv")):
        name = path.stem.removeprefix("oof_")
        model = next((candidate for candidate in model_names if name.endswith(f"_{candidate}")), None)
        if model is None:
            continue
        label = name[: -(len(model) + 1)]
        frame = pd.read_csv(path)
        probability_columns = sorted(
            [column for column in frame.columns if column.startswith("prob_class_")],
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )
        if "y_true" not in frame or not probability_columns:
            continue
        classes = np.array([int(column.rsplit("_", 1)[1]) for column in probability_columns], dtype=int)
        aggregation: dict[str, str] = {"y_true": "first", "subject_id": "first"}
        aggregation.update({column: "mean" for column in probability_columns})
        lap = frame.groupby("lap_group", as_index=False).agg(aggregation)
        y = lap["y_true"].to_numpy(dtype=int)
        probability = lap[probability_columns].to_numpy(dtype=float)
        row: dict[str, Any] = {"label": label, "model": model, "n_laps": int(len(lap))}
        auc = _safe_auc(y, probability, classes)
        row.update({f"{key}_lap": value for key, value in auc.items()})
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "auc_lap_level_summary.csv", index=False, encoding="utf-8-sig")
    return result


def run_modeling(config: dict[str, Any], features_path: Path, output_root: Path) -> pd.DataFrame:
    df = pd.read_csv(features_path, dtype={"subject_id": str})
    model_cfg = config["model"]
    output_dir = output_root / "models"
    balance_method = str(model_cfg.get("balance_method", "class_weight"))
    if balance_method not in {"class_weight", "none"}:
        raise ValueError("model.balance_method must be 'class_weight' or 'none'")
    all_metrics = []
    for label in model_cfg["labels"]:
        all_metrics.append(
            evaluate_label(
                df=df,
                label_col=label,
                output_dir=output_dir,
                group_level=model_cfg["group_level"],
                select_k=int(model_cfg["select_k"]),
                random_state=int(model_cfg["random_state"]),
                balance_method=balance_method,
            )
        )
    all_metrics.append(
        evaluate_ordinal_misc(
            df,
            output_dir,
            model_cfg["group_level"],
            int(model_cfg["select_k"]),
            balance_method,
        )
    )
    result = pd.concat(all_metrics, ignore_index=True, sort=False)
    result.to_csv(output_dir / "metrics_all.csv", index=False, encoding="utf-8-sig")
    write_lap_auc_summary(output_dir)
    sensitivity = result[result["label"].isin(["misc_3class_id", "misc_binary_2plus", "misc_binary_6plus", "ordinal_misc_score"])].copy()
    sensitivity.to_csv(output_dir / "model_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    distributions = []
    for label in model_cfg["labels"]:
        work = df[df[label].notna() & (df[label] >= 0)].copy()
        counts = work[label].astype(int).value_counts().sort_index()
        total = int(counts.sum())
        for cls, count in counts.items():
            distributions.append(
                {
                    "label": label,
                    "class": int(cls),
                    "count": int(count),
                    "proportion": float(count / total) if total else np.nan,
                    "unit": "window",
                    "balance_method": balance_method,
                }
            )
    pd.DataFrame(distributions).to_csv(output_dir / "model_label_distributions.csv", index=False, encoding="utf-8-sig")
    return result
