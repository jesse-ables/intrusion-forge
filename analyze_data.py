import logging
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

from src.common.config import load_config, save_config
from src.common.log import (
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.common.paths import OutputPaths
from src.common.utils import flush_timing, load_from_json, timed

from src.data.io import load_df
from src.data.complexity import compute_all_complexity_measures

setup_logger(log_file="resources/logs.txt")
logger = logging.getLogger(__name__)

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 3, 5, 10, 20],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", 0.5],
}


def _run_outer_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    inner_cv: StratifiedKFold,
    param_grid: dict,
    random_state: int,
) -> dict:
    """Run one outer CV fold: fit GridSearchCV, predict, collect per-fold metrics.

    Output keys: 'f1', 'auc', 'importances' (np.ndarray),
    'y_pred' (list[int]), 'y_proba' (list[float]), 'indices' (list).
    """
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


@timed
def fit_failure_classifier(
    paths: OutputPaths,
    param_grid: dict,
    cluster_stats: dict | None = None,
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
    if cluster_stats is None:
        cluster_stats = load_from_json(
            paths.json_logs / "analysis/cluster_summary.json"
        )
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

    outer_cv = StratifiedKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=n_inner_splits, shuffle=True, random_state=random_state
    )

    fold_f1s: list[float] = []
    fold_aucs: list[float] = []
    fold_importances: list[np.ndarray] = []
    oof_y_true: list[int] = []
    oof_y_pred: list[int] = []
    oof_y_proba: list[float] = []
    oof_indices: list = []

    for train_idx, test_idx in tqdm(
        outer_cv.split(X, y), total=n_outer_splits, desc="Outer CV"
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

    results = {
        "f1_score": float(np.mean(fold_f1s)),
        "f1_score_std": float(np.std(fold_f1s)),
        "f1_scores_per_fold": fold_f1s,
        "roc_auc": float(np.nanmean(fold_aucs)),
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
            str(cid): float(proba)
            for cid, proba in zip(oof_indices, oof_y_proba)
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


def run_complexity(
    paths: OutputPaths,
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    centroids: dict,
    noise_cluster_ids: list[int],
    k: int,
    top_k_clusters: int,
    min_subsample_per_cluster: int,
    max_complexity_samples: int | None,
    metric: str,
    random_state: int,
    analysis_bus: LogDispatcher,
) -> tuple[dict, dict]:
    """Compute complexity measures, build cluster summary, publish to log bus."""
    pred_infos = load_from_json(paths.json_logs / "analysis/predictions/test.json")
    df_meta = load_from_json(paths.data_logs / "data/df_meta.json")

    logger.info("Computing complexity measures ...")
    complexity = compute_all_complexity_measures(
        X_num,
        X_cat,
        y_class,
        y_cluster,
        centroids,
        k=k,
        top_k_clusters=top_k_clusters,
        max_samples=max_complexity_samples,
        min_per_cluster=min_subsample_per_cluster,
        metric=metric,
        noise_cluster_ids=set(noise_cluster_ids),
        random_state=random_state,
    )

    cluster_errors = pred_infos["clusters"]["global"]

    cluster_to_class: dict[str, int] = {}
    for cid in np.unique(y_cluster):
        if cid == -1:
            continue
        mask = y_cluster == cid
        cluster_to_class[str(cid)] = int(y_class[mask][0])

    cluster_summary = {}
    for cid, measures in complexity.items():
        error_entry = (cluster_errors or {}).get(str(cid), {})
        failure_rate = error_entry.get("error_rate")
        cluster_summary[str(cid)] = {
            **measures,
            "cluster_class": cluster_to_class.get(str(cid)),
            "failure_rate": failure_rate,
            "is_failed": failure_rate is not None and failure_rate > 0.0,
        }

    analysis_bus.publish(
        LogBundle.from_dict({"json/analysis/cluster_summary": cluster_summary})
    )
    logger.info("Cluster summary published.")
    return cluster_summary, df_meta


@timed
def analyze(
    paths: OutputPaths,
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    y_class: np.ndarray,
    y_cluster: np.ndarray,
    centroids: dict,
    failure_threshold: float,
    k: int,
    top_k_clusters: int,
    min_subsample_per_cluster: int,
    max_complexity_samples: int | None = None,
    noise_cluster_ids: list[int] | None = None,
    metric: str = "cosine",
    random_state: int = 0,
) -> None:
    """Run data analysis pipeline (compute only — plotting is handled by render_plots.py)."""
    logger.info("Starting analysis pipeline ...")
    noise_ids = noise_cluster_ids or []
    analysis_bus = LogDispatcher()
    analysis_bus.subscribe(JSONSubscriber(paths.json_logs))

    cluster_summary, _ = run_complexity(
        paths,
        X_num,
        X_cat,
        y_class,
        y_cluster,
        centroids,
        noise_ids,
        k,
        top_k_clusters,
        min_subsample_per_cluster,
        max_complexity_samples,
        metric,
        random_state,
        analysis_bus,
    )
    fit_failure_classifier(
        paths,
        RF_PARAM_GRID,
        cluster_stats=cluster_summary,
        failure_threshold=failure_threshold,
        analysis_bus=analysis_bus,
    )


def main():
    """Main entry point for data analysis."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    paths = OutputPaths(
        processed_data=Path(cfg.path.processed_data),
        data_logs=Path(cfg.path.data_logs),
        configs=Path(cfg.path.configs),
        json_logs=Path(cfg.path.json_logs),
        pickle=Path(cfg.path.pickle),
        models=Path(cfg.path.models),
        figures=Path(cfg.path.figures),
    )
    save_config(cfg, paths.configs / "config_composed.json")

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    ext = cfg.data.extension

    train_df = load_df(str(paths.processed_data / f"train.{ext}"))
    val_df = load_df(str(paths.processed_data / f"val.{ext}"))
    test_df = load_df(str(paths.processed_data / f"test.{ext}"))
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    X_num = (
        combined[num_cols].to_numpy(dtype=np.float64)
        if num_cols
        else np.empty((len(combined), 0))
    )
    X_cat = combined[cat_cols].to_numpy() if cat_cols else None
    y_class = combined[f"encoded_{cfg.data.label_col}"].to_numpy(dtype=np.int64)
    y_cluster = combined["cluster"].to_numpy(dtype=np.int64)

    clusters_meta = load_from_json(paths.data_logs / "data/clusters_meta.json")
    centroids = clusters_meta.get("centroids", {})
    noise_cluster_ids = clusters_meta.get("noise_cluster_ids", [])

    analyze(
        paths=paths,
        X_num=X_num,
        X_cat=X_cat,
        y_class=y_class,
        y_cluster=y_cluster,
        centroids=centroids,
        failure_threshold=cfg.analysis.failure_threshold or 0.0,
        k=cfg.complexity.k,
        top_k_clusters=cfg.complexity.top_k_clusters,
        min_subsample_per_cluster=cfg.complexity.min_subsample_per_cluster,
        max_complexity_samples=cfg.complexity.max_complexity_samples,
        noise_cluster_ids=noise_cluster_ids,
        metric=cfg.complexity.distance,
        random_state=cfg.seed,
    )
    flush_timing(paths.json_logs / "timing.json")


if __name__ == "__main__":
    main()
