import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm import tqdm

from src.core.config import load_config, save_config, to_container
from src.core.log import (
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.core.paths import OutputPaths
from src.core.utils import flush_timing, load_from_json, timed

setup_logger(log_file="resources/logs.txt")
logger = logging.getLogger(__name__)


def _max_safe_splits(n_minority: int, n_splits_cfg: int) -> int:
    """Largest k <= n_splits_cfg such that StratifiedKFold(k) won't degenerate."""
    k = min(n_splits_cfg, n_minority)
    return k if k >= 2 else 0


def _run_outer_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    inner_cv: StratifiedKFold | None,
    param_grid: dict,
    random_state: int,
) -> dict:
    """Run one outer CV fold. When inner_cv is None, skip GridSearchCV and use default RF."""
    if inner_cv is None:
        best = RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced",
        )
        best.fit(X_train, y_train)
    else:
        grid = GridSearchCV(
            estimator=RandomForestClassifier(
                random_state=random_state,
                class_weight="balanced",
            ),
            param_grid=param_grid,
            cv=inner_cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")

    return {
        "f1": f1_score(y_test, y_pred),
        "auc": auc,
        "importances": best.feature_importances_,
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
        "indices": X_test.index.tolist(),
    }


def build_cluster_summary(
    complexity: dict,
    class_complexity: dict,
    predictions: dict,
    failure_threshold: float,
) -> dict:
    """Merge per-cluster complexity with class-level complexity (joined on the
    cluster's class via `cluster_class`) and per-classifier failure rates.

    Output schema per cluster:
        cluster_<measure>  — cluster-level complexity (vs top-K adversarial clusters)
        class_<measure>    — class-level complexity of the cluster's class
        cluster_class, is_noise_cluster, failure_rate, is_failed
    """
    cluster_errors = predictions.get("clusters", {}).get("global", {}) or {}
    summary: dict[str, dict] = {}
    for cid, cluster_measures in complexity.items():
        class_id = cluster_measures.get("cluster_class")
        class_measures = (
            class_complexity.get(str(class_id), {}) if class_id is not None else {}
        )
        cluster_feats = {
            f"cluster_{k}": v
            for k, v in cluster_measures.items()
            if k not in ("cluster_class", "is_noise_cluster")
        }
        class_feats = {
            f"class_{k}": v
            for k, v in class_measures.items()
            if k != "is_noise_cluster"
        }
        error_entry = cluster_errors.get(str(cid), {})
        failure_rate = error_entry.get("error_rate")
        summary[str(cid)] = {
            **cluster_feats,
            **class_feats,
            "cluster_class": class_id,
            "is_noise_cluster": cluster_measures.get("is_noise_cluster", False),
            "failure_rate": failure_rate,
            "is_failed": failure_rate is not None and failure_rate > failure_threshold,
        }
    return summary


@timed
def fit_failure_classifier(
    cluster_stats: dict,
    param_grid: dict,
    *,
    feature_cols: list[str] | None = None,
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    random_state: int = 42,
    failure_threshold: float = 0.0,
    analysis_bus: LogDispatcher | None = None,
) -> dict:
    """Train a Random Forest with nested CV to predict cluster failure from separability features.

    Uses nested cross-validation: outer StratifiedKFold for unbiased evaluation,
    inner GridSearchCV for hyperparameter selection. Metrics are aggregated over
    out-of-fold (OOF) predictions.
    """
    logger.info("Running failure classifier ...")
    df = pd.DataFrame.from_dict(cluster_stats, orient="index")
    if feature_cols is None:
        feature_cols = [
            c
            for c in df.select_dtypes("number").columns
            if c != "is_failed" and c != "failure_rate"
        ]
    X = df[feature_cols].copy()

    y = df["failure_rate"].apply(lambda x: 1 if x > failure_threshold else 0)

    n_positives = int(y.sum())
    n_negatives = int((1 - y).sum())
    n_minority = min(n_positives, n_negatives)
    outer_k = _max_safe_splits(n_minority, n_outer_splits)
    if outer_k == 0:
        message = (
            f"Failure classifier skipped: only {n_minority} minority sample(s) "
            f"(positives={n_positives}, threshold={failure_threshold}). "
            f"Need >=2 for stratified CV."
        )
        logger.warning("[STAGE-SKIP] %s", message)
        results = {
            "skipped": True,
            "reason": "insufficient_minority_samples",
            "message": message,
            "n_minority": n_minority,
            "n_positives": n_positives,
            "n_negatives": n_negatives,
            "threshold": failure_threshold,
            "min_required": 2,
        }
        if analysis_bus is not None:
            analysis_bus.publish(
                LogBundle.from_dict({"json/analysis/classifier_results": results})
            )
        return results

    m_train_worst = n_minority - math.ceil(n_minority / outer_k)
    inner_k = _max_safe_splits(m_train_worst, n_inner_splits)
    if outer_k < n_outer_splits or inner_k < n_inner_splits:
        logger.warning(
            "[CV-ADAPT] Adapting CV (minority=%d): outer %d→%d, inner %d→%d%s",
            n_minority,
            n_outer_splits,
            outer_k,
            n_inner_splits,
            inner_k or 0,
            " (no GridSearchCV — using RF defaults)" if inner_k == 0 else "",
        )

    outer_cv = StratifiedKFold(
        n_splits=outer_k, shuffle=True, random_state=random_state
    )
    inner_cv = (
        StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=random_state)
        if inner_k > 0
        else None
    )

    fold_f1s: list[float] = []
    fold_aucs: list[float] = []
    fold_importances: list[np.ndarray] = []
    oof_y_true: list[int] = []
    oof_y_pred: list[int] = []
    oof_y_proba: list[float] = []
    oof_indices: list = []

    for train_idx, test_idx in tqdm(
        outer_cv.split(X, y), total=outer_k, desc="Outer CV"
    ):
        fold = _run_outer_fold(
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[test_idx],
            inner_cv,
            param_grid,
            random_state,
        )
        fold_f1s.append(fold["f1"])
        fold_aucs.append(fold["auc"])
        fold_importances.append(fold["importances"])
        oof_y_true.extend(y.iloc[test_idx].tolist())
        oof_y_pred.extend(fold["y_pred"])
        oof_y_proba.extend(fold["y_proba"])
        oof_indices.extend(fold["indices"])

    oof_y_true_arr = np.array(oof_y_true)
    oof_y_pred_arr = np.array(oof_y_pred)
    oof_y_proba_arr = np.array(oof_y_proba)

    fpr, tpr, _ = roc_curve(oof_y_true_arr, oof_y_proba_arr)
    mean_importances = np.mean(fold_importances, axis=0)

    try:
        oof_auc = float(roc_auc_score(oof_y_true_arr, oof_y_proba_arr))
    except ValueError:
        oof_auc = float("nan")

    results = {
        "f1_score": float(f1_score(oof_y_true_arr, oof_y_pred_arr)),
        "f1_score_std": float(np.std(fold_f1s)),
        "f1_scores_per_fold": fold_f1s,
        "roc_auc": oof_auc,
        "roc_auc_std": float(np.nanstd(fold_aucs)),
        "roc_auc_per_fold": fold_aucs,
        "roc_curve_data": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "confusion_matrix": confusion_matrix(oof_y_true_arr, oof_y_pred_arr).tolist(),
        "classification_report": classification_report(
            oof_y_true_arr,
            oof_y_pred_arr,
            digits=4,
            output_dict=True,
        ),
        "feature_importances": dict(zip(feature_cols, mean_importances.tolist())),
        "oof_predictions": {
            str(cid): int(pred == true)
            for cid, pred, true in zip(oof_indices, oof_y_pred, oof_y_true)
        },
        "oof_risk_proba": {
            str(cid): float(proba) for cid, proba in zip(oof_indices, oof_y_proba)
        },
    }
    if analysis_bus is not None:
        analysis_bus.publish(
            LogBundle.from_dict({"json/analysis/classifier_results": results})
        )
    logger.info(
        "Classifier results — F1: %.4f, ROC-AUC: %.4f",
        results["f1_score"],
        results["roc_auc"],
    )
    return results


def main():
    """Main entry point for failure-classifier training (per-classifier stage)."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    paths = OutputPaths(
        processed_data=Path(cfg.path.processed_data),
        shared=Path(cfg.path.shared),
        configs=Path(cfg.path.configs),
        outputs=Path(cfg.path.outputs),
        pickle=Path(cfg.path.pickle),
        models=Path(cfg.path.models),
        figures=Path(cfg.path.figures),
    )
    save_config(cfg, paths.configs / "config_composed.json")

    complexity_path = paths.shared / "complexity.json"
    class_complexity_path = paths.shared / "class_complexity.json"
    for p in (complexity_path, class_complexity_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing complexity artifact at {p}. "
                "Run `make complexity` first."
            )
    complexity = load_from_json(complexity_path)
    class_complexity = load_from_json(class_complexity_path)
    predictions = load_from_json(paths.outputs / "analysis/predictions/test.json")

    cluster_summary = build_cluster_summary(
        complexity, class_complexity, predictions, cfg.failure_classifier.threshold
    )

    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.outputs))
    bus.publish(LogBundle.from_dict({"json/analysis/cluster_summary": cluster_summary}))
    logger.info("Cluster summary published.")

    fit_failure_classifier(
        cluster_summary,
        to_container(cfg.failure_classifier.param_grid),
        n_outer_splits=cfg.failure_classifier.n_outer_splits,
        n_inner_splits=cfg.failure_classifier.n_inner_splits,
        failure_threshold=cfg.failure_classifier.threshold,
        random_state=cfg.seed,
        analysis_bus=bus,
    )

    flush_timing(paths.outputs / "timing.json")


if __name__ == "__main__":
    main()
