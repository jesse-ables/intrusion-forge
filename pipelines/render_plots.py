import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from src.core.config import load_config, save_config
from src.core.log import (
    FilesystemFigureSubscriber,
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.core.paths import OutputPaths
from src.core.utils import flush_timing, load_from_json, timed
from src.domain.plot.charts import (
    bar_plot,
    heatmap_plot,
    ridgeline_plot,
    strip_count_panel_plot,
    violin_plot,
)
from src.domain.plot.metrics import confusion_matrix_plot, roc_plot
from src.domain.plot.base import Plot
from src.domain.plot.style import (
    CORRECT_COLOR,
    FAILED_COLOR,
    HIGHLIGHT_COLOR,
    PALETTE,
    apply_plot_style,
)

setup_logger(log_file="resources/logs.txt")
apply_plot_style()
logger = logging.getLogger(__name__)


def _plot_failure_strips(
    summary_df: pd.DataFrame, rf_correct: np.ndarray
) -> dict[str, Plot]:
    """Build failure-rate and RF-prediction strip plots ordered by class severity."""
    class_order = (
        summary_df.groupby("class_name")["failure_rate"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    classes = summary_df["class_name"].values
    failure_rate = summary_df["failure_rate"].values
    counts_by_class = summary_df.groupby("class_name").size().to_dict()
    failed_counts_by_class = (
        summary_df.loc[summary_df["is_failed"]].groupby("class_name").size().to_dict()
    )

    monochrome_fill = np.zeros(len(classes), dtype=float)

    return {
        "summary/failure_rate_strip_box": strip_count_panel_plot(
            categories=classes,
            values=failure_rate,
            category_order=class_order,
            counts_by_class=counts_by_class,
            failed_counts_by_class=failed_counts_by_class,
            fill_values=monochrome_fill,
            fill_categorical_colors=(PALETTE[0],),
            x_label="Failure rate",
        ),
        "summary/rf_prediction_strip_box": strip_count_panel_plot(
            categories=classes,
            values=failure_rate,
            category_order=class_order,
            counts_by_class=counts_by_class,
            failed_counts_by_class=failed_counts_by_class,
            fill_values=rf_correct,
            fill_categorical_colors=(HIGHLIGHT_COLOR, PALETTE[0]),
            marker_values=rf_correct,
            marker_shapes=("X", "o"),
            x_label="Failure rate",
        ),
    }


def _plot_feature_by_outcome(
    summary_df: pd.DataFrame, features: list[str]
) -> dict[str, Plot]:
    """Build per-feature split-violin plots for correct vs failed clusters."""
    categories = np.where(summary_df["is_failed"], "failed", "correct")
    return {
        f"summary/global/{feature}": violin_plot(
            categories=categories,
            values=summary_df[feature].values,
            category_order=["correct", "failed"],
            split=True,
            inner="box",
            colors=(CORRECT_COLOR, FAILED_COLOR),
            y_label=feature,
        )
        for feature in features
    }


def _plot_cluster_risk_heatmap(
    summary_df: pd.DataFrame,
    centroids: dict[str, list[float]],
    label_mapping: dict[str, str],
    rf_risk: dict[str, float],
    max_clusters: int = 30,
    failed_fraction: float = 0.6,
    metric: str = "cosine",
) -> dict[str, Plot]:
    """Risk heatmap focused on the worst class.

    Selection:
      1. Target class = class with the highest mean failure_rate.
      2. Failed quota = `failed_fraction * max_clusters` clusters of the target
         class with failure_rate > 0, sampled via linspace over sorted
         failure_rate to spread values.
      3. Safe quota = remaining slots from clusters with failure_rate == 0:
         half nearest (conflict) and half farthest (no conflict) from the
         centroids of the selected failed clusters.

    Diagonal: RF out-of-fold failure probability per cluster.
    Off-diagonal: pairwise centroid distance under `metric`.
    Display order: failure_rate descending, class secondary.
    """
    valid = summary_df[
        summary_df["cluster_class"].notna()
        & summary_df["failure_rate"].notna()
        & summary_df.index.astype(str).isin(centroids.keys())
    ].copy()
    if len(valid) < 2:
        return {}

    failure_rate_all = valid["failure_rate"].to_numpy(dtype=float)
    cluster_classes_all = valid["cluster_class"].astype(int).to_numpy()
    cluster_ids_all = [str(cid) for cid in valid.index]

    target_class = int(valid.groupby("cluster_class")["failure_rate"].mean().idxmax())

    centroid_matrix = np.stack(
        [np.asarray(centroids[cid], dtype=np.float64) for cid in cluster_ids_all]
    )
    dist_all = pairwise_distances(centroid_matrix, metric=metric)

    n_failed_budget = int(round(failed_fraction * max_clusters))
    n_safe_budget = max_clusters - n_failed_budget

    failed_pool = np.where(
        (cluster_classes_all == target_class) & (failure_rate_all > 0)
    )[0]
    if failed_pool.size == 0:
        return {}
    failed_sorted = failed_pool[np.argsort(failure_rate_all[failed_pool])]
    n_failed = min(n_failed_budget, failed_sorted.size)
    positions = np.round(np.linspace(0, failed_sorted.size - 1, n_failed)).astype(int)
    selected_failed = failed_sorted[positions]

    safe_pool = np.setdiff1d(np.where(failure_rate_all == 0.0)[0], selected_failed)
    if safe_pool.size and n_safe_budget > 0:
        min_dists = dist_all[np.ix_(safe_pool, selected_failed)].min(axis=1)
        safe_sorted = safe_pool[np.argsort(min_dists)]
        n_safe = min(n_safe_budget, safe_sorted.size)
        n_conflict = n_safe // 2
        n_far = n_safe - n_conflict
        conflict = safe_sorted[:n_conflict]
        far = safe_sorted[-n_far:] if n_far else np.array([], dtype=int)
        selected_safe = np.concatenate([conflict, far])
    else:
        selected_safe = np.array([], dtype=int)

    selected = np.concatenate([selected_failed, selected_safe])
    if selected.size < 2:
        return {}

    display_order = selected[
        np.lexsort((cluster_classes_all[selected], -failure_rate_all[selected]))
    ]

    cluster_ids = [cluster_ids_all[i] for i in display_order]
    cluster_classes = cluster_classes_all[display_order]
    failure_rate = failure_rate_all[display_order]

    risk_scores = np.array(
        [rf_risk.get(cid, np.nan) for cid in cluster_ids], dtype=float
    )

    class_names_arr = np.array(
        [label_mapping.get(str(int(c)), str(c)) for c in cluster_classes]
    )
    unique_names = sorted(set(class_names_arr.tolist()))
    name_to_num = {name: i + 1 for i, name in enumerate(unique_names)}
    labels = [
        f"{cid}:{name_to_num[name]}" for cid, name in zip(cluster_ids, class_names_arr)
    ]
    class_legend = {num: name for name, num in name_to_num.items()}

    dist = dist_all[np.ix_(display_order, display_order)]

    return {
        "summary/cluster_risk_heatmap": heatmap_plot(
            matrix=dist,
            labels=labels,
            sidebar_values=failure_rate,
            sidebar_label="failure rate",
            diagonal_values=risk_scores,
            diagonal_label="RF failure risk (OOF proba)",
            matrix_label=f"{metric} centroid distance",
            label_legend=class_legend,
            legend_title="cluster label format — id : class id",
        ),
    }


def _plot_class_separability_ridgeline(
    summary_df: pd.DataFrame,
) -> dict[str, Plot]:
    """Ridgeline of inter/intra neighbourhood ratios per class, split correct vs failed.

    Ratio = 1 − class_n2_min. Low ratio means even the easiest adversary class
    sits inside the cluster's local neighbourhood.
    One row per class with at least one correct and one failed cluster.
    """
    required = {"class_n2_min", "is_failed", "class_name", "failure_rate"}
    if not required.issubset(summary_df.columns):
        return {}

    ratio = 1.0 - summary_df["class_n2_min"].astype(float)

    valid = (
        summary_df["failure_rate"].notna()
        & summary_df["class_name"].notna()
        & ratio.notna()
        & np.isfinite(ratio)
    )
    df = summary_df.loc[valid].assign(_ratio=ratio[valid])
    if df.empty:
        return {}

    distributions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    class_median_failure: dict[str, float] = {}
    for class_name, group in df.groupby("class_name"):
        correct = group.loc[~group["is_failed"], "_ratio"].to_numpy(dtype=float)
        failed = group.loc[group["is_failed"], "_ratio"].to_numpy(dtype=float)
        if len(correct) == 0 or len(failed) == 0:
            continue
        distributions[str(class_name)] = (correct, failed)
        class_median_failure[str(class_name)] = float(group["failure_rate"].median())

    if not distributions:
        return {}

    order = sorted(
        distributions.keys(),
        key=lambda k: class_median_failure.get(k, 0.0),
        reverse=True,
    )

    return {
        "summary/class_separability_ridgeline": ridgeline_plot(
            distributions,
            order=order,
            legend_labels=("correct", "failed"),
            colors=(CORRECT_COLOR, FAILED_COLOR),
            x_label="1 - class_n2_min",
        ),
    }


def _plot_rf_evaluation(classifier_results: dict) -> dict[str, Plot]:
    """Build confusion matrix, ROC curve, and feature-importance bar plots."""
    cm = np.array(classifier_results["confusion_matrix"], dtype=float)
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    roc_data = classifier_results["roc_curve_data"]
    importances = classifier_results["feature_importances"]

    return {
        "summary/correlation/confusion_matrix": confusion_matrix_plot(
            cm=cm,
            class_names=["correct", "failed"],
            normalize="row",
            extra_metrics={
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Specificity": specificity,
            },
        ),
        "summary/correlation/roc_curve": roc_plot(
            fpr=np.array(roc_data["fpr"]),
            tpr=np.array(roc_data["tpr"]),
            auc_score=classifier_results["roc_auc"],
            title="ROC (OOF)",
        ),
        "summary/correlation/feature_importances": bar_plot(
            labels=list(importances.keys()),
            values=list(importances.values()),
            orientation="h",
            sort="asc",
            top_k=20,
            annotate_values=False,
            color_gradient=True,
            x_label="Importance",
        ),
    }


@timed
def assemble_analysis_figures(
    cluster_summary: dict,
    centroids: dict[str, list[float]],
    df_meta: dict,
    classifier_results: dict,
    *,
    metric: str = "cosine",
    analysis_bus: LogDispatcher | None = None,
) -> dict[str, Plot]:
    """Build all analysis figures and publish to log bus."""
    logger.info("Building summary visualizations ...")
    summary_df = pd.DataFrame.from_dict(cluster_summary, orient="index")
    label_mapping = {str(k): v for k, v in df_meta["label_mapping"].items()}
    summary_df["class_name"] = (
        summary_df["cluster_class"].astype(str).map(label_mapping)
    )

    if classifier_results.get("skipped"):
        logger.warning(
            "[STAGE-SKIP] Skipping failure-classifier plots: %s",
            classifier_results.get("message", classifier_results.get("reason")),
        )
        figures: dict[str, Plot] = {}
        figures.update(_plot_class_separability_ridgeline(summary_df))
        if analysis_bus is not None:
            analysis_bus.publish(LogBundle(figures=figures))
        return figures

    oof_preds = classifier_results.get("oof_predictions", {})
    rf_correct = np.array(
        [oof_preds.get(str(cid), np.nan) for cid in summary_df.index], dtype=float
    )
    summary_df["rf_correct"] = rf_correct

    sorted_by_importance = sorted(
        classifier_results["feature_importances"].items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top10 = [name for name, _ in sorted_by_importance[:10]]
    violin_features = [f for f in top10 if f in summary_df.columns]

    figures: dict[str, Plot] = {}
    figures.update(_plot_failure_strips(summary_df, rf_correct))
    figures.update(_plot_feature_by_outcome(summary_df, violin_features))
    figures.update(_plot_rf_evaluation(classifier_results))
    figures.update(_plot_class_separability_ridgeline(summary_df))
    figures.update(
        _plot_cluster_risk_heatmap(
            summary_df,
            centroids,
            label_mapping,
            classifier_results["oof_risk_proba"],
            metric=metric,
        )
    )
    if analysis_bus is not None:
        analysis_bus.publish(LogBundle(figures=figures))
    return figures


def main():
    """Main entry point for plot rendering."""
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
    save_config(cfg, paths.configs / "config_composed_render.json")

    clusters_meta = load_from_json(paths.shared / "metadata/clusters_meta.json")
    centroids = clusters_meta.get("centroids", {})

    cluster_summary = load_from_json(paths.outputs / "analysis/cluster_summary.json")
    df_meta = load_from_json(paths.shared / "metadata/df_meta.json")
    classifier_results = load_from_json(
        paths.outputs / "analysis/classifier_results.json"
    )

    analysis_bus = LogDispatcher()
    analysis_bus.subscribe(JSONSubscriber(paths.outputs))
    analysis_bus.subscribe(FilesystemFigureSubscriber(paths.figures))

    assemble_analysis_figures(
        cluster_summary=cluster_summary,
        centroids=centroids,
        df_meta=df_meta,
        classifier_results=classifier_results,
        metric=cfg.complexity.distance,
        analysis_bus=analysis_bus,
    )
    flush_timing(paths.outputs / "timing.json")


if __name__ == "__main__":
    main()
