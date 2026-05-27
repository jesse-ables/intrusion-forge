import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

import shap
import matplotlib.pyplot as plt

from src.core.config import load_config, save_config
from src.core.log import (
    FilesystemFigureSubscriber,
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    PickleSubscriber,
    setup_logger,
)
from src.core.paths import OutputPaths
from src.core.utils import flush_timing, load_from_json, timed
from src.core.io import load_listed_dfs
from src.domain.data.preprocessing import subsample_df
from src.domain.projection import stratified_subsample, tsne_projection
from src.domain.plot.base import Plot
from src.domain.plot.charts import bar_plot, line_plot, scatter_plot
from src.domain.plot.metrics import confusion_matrix_plot
from src.domain.plot.style import apply_plot_style, extended_palette

setup_logger(log_file="resources/logs.txt")
apply_plot_style()
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Shared data parameters across stages."""

    processed_data_path: Path
    extension: str
    num_cols: list[str]
    cat_cols: list[str]
    label_col: str
    n_samples: int | None


def _load_data(
    data: DataConfig, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits; optionally subsample the training set."""
    train_df, val_df, test_df = load_listed_dfs(
        data.processed_data_path,
        [
            f"train.{data.extension}",
            f"val.{data.extension}",
            f"test.{data.extension}",
        ],
    )
    if data.n_samples is not None:
        train_df = subsample_df(
            train_df,
            data.n_samples,
            random_state=random_state,
            label_col=data.label_col,
        )
    return train_df, val_df, test_df


_METRIC_FNS: list[tuple[str, Callable]] = [
    ("precision", precision_score),
    ("recall", recall_score),
    ("f1", f1_score),
]


def _compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Overall accuracy, macro/weighted precision/recall/F1, plus per-class arrays."""
    full: dict = {"accuracy": float(accuracy_score(y_true, y_pred))}

    for avg in ("macro", "weighted"):
        for name, fn in _METRIC_FNS:
            full[f"{name}_{avg}"] = float(
                fn(y_true, y_pred, average=avg, zero_division=0)
            )

    for name, fn in _METRIC_FNS:
        full[f"{name}_per_class"] = fn(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()

    return full


def _cluster_error_rates(
    clusters: np.ndarray, error_mask: np.ndarray
) -> dict[str, dict]:
    """Return {cluster_id: {n_error, n_total, error_rate}} sorted by error_rate desc."""
    failed = clusters[error_mask]
    stats: dict[str, dict] = {}
    for c in np.unique(clusters):
        n_total = int((clusters == c).sum())
        n_error = int((failed == c).sum())
        stats[str(c)] = {
            "n_error": n_error,
            "n_total": n_total,
            "error_rate": (n_error / n_total) if n_total > 0 else None,
        }
    return dict(
        sorted(stats.items(), key=lambda x: x[1]["error_rate"] or 0.0, reverse=True)
    )


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    clusters: np.ndarray | None = None,
) -> dict:
    """Per-class prediction quality and cluster-level error rates.

    `confidences` may be a 1D max-prob array or a 2D (n_samples, n_classes)
    probability matrix; in the latter case the max per row is used.
    """
    confidences = np.asarray(confidences)
    if confidences.ndim == 2:
        confidences = confidences.max(axis=1)

    has_cluster = clusters is not None
    global_error_mask = y_true != y_pred

    cluster_errors_total = (
        _cluster_error_rates(clusters, global_error_mask) if has_cluster else None
    )
    cluster_errors_by_class: dict[str, dict] | None = {} if has_cluster else None

    classes: dict[str, dict] = {}
    for label in np.unique(y_true):
        mask = y_true == label
        n_total = int(mask.sum())
        n_errors = int((y_true[mask] != y_pred[mask]).sum())
        error_mask = mask & global_error_mask

        if has_cluster:
            wrong_preds = y_pred[error_mask]
            wrong_clusters = clusters[error_mask]
            cluster_in_fn = {
                str(cls): np.unique(wrong_clusters[wrong_preds == cls]).tolist()
                for cls in np.unique(wrong_preds)
            }
            tp_clusters = clusters[mask & ~global_error_mask]
            cluster_in_tp = np.unique(tp_clusters).tolist()

            class_clusters = clusters[mask]
            cluster_errors_by_class[str(label)] = _cluster_error_rates(
                class_clusters, error_mask[mask]
            )
        else:
            cluster_in_fn = cluster_in_tp = None

        classes[str(label)] = {
            "tot_failures": n_errors,
            "tot_samples": n_total,
            "failure_rate": n_errors / n_total if n_total > 0 else None,
            "mean_confidence": (
                float(confidences[mask].mean()) if n_total > 0 else None
            ),
            "cluster_in_fn": cluster_in_fn,
            "cluster_in_tp": cluster_in_tp,
        }

    classes = dict(
        sorted(
            classes.items(),
            key=lambda x: x[1]["failure_rate"] or 0.0,
            reverse=True,
        )
    )

    return {
        "classes": classes,
        "clusters": {
            "global": cluster_errors_total,
            "by_class": cluster_errors_by_class,
        },
    }


def _build_test_figures(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_mapping: dict,
    n_samples: int = 2000,
) -> dict[str, Plot]:
    """Confusion matrix + per-class F1 bar + t-SNE scatter on raw features."""
    figures: dict[str, Plot] = {}

    classes = np.unique(y_true)
    class_names = [label_mapping.get(str(int(c)), str(c)) for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    figures["figure/testing/confusion_matrix"] = confusion_matrix_plot(
        cm, class_names=class_names, normalize=None
    )

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_dict = {
        label_mapping.get(str(int(c)), str(c)): float(v)
        for c, v in zip(classes, f1_per_class)
    }
    figures["figure/testing/f1_per_class"] = bar_plot(
        list(f1_dict.keys()),
        list(f1_dict.values()),
        orientation="v",
        color=extended_palette(len(f1_dict)),
        sort=None,
        ylim=(0, 1),
    )

    names = {int(c): label_mapping.get(str(int(c)), str(c)) for c in classes}
    correct = y_pred == y_true
    vis_idx = stratified_subsample(y_true, n_samples=n_samples, stratify=False)
    figures["figure/testing/raw/classes"] = scatter_plot(
        tsne_projection(X[vis_idx], n_components=2),
        y_true[vis_idx],
        highlight_mask=~correct[vis_idx],
        names=names,
        marker_size=12.0,
    )
    return figures


def _training_history_figures(history: dict[str, list[float]]) -> dict[str, Plot]:
    """One line plot per scalar in the per-step DL training history."""
    return {
        f"figure/training/{name}_curve": line_plot(
            {name: values},
            y_label=name,
            show_legend=False,
        )
        for name, values in history.items()
        if values
    }


_TRAINING_MODULES = {
    "ml": "src.domain.training.ml",
    "dl": "src.domain.training.dl",
}


def _resolve_training_module(kind: str):
    """Return the training module matching the classifier kind."""
    if kind not in _TRAINING_MODULES:
        raise ValueError(
            f"Unknown classifier kind: {kind!r}. "
            f"Expected one of {sorted(_TRAINING_MODULES)}."
        )
    return importlib.import_module(_TRAINING_MODULES[kind])


def _build_dl_context(
    cfg,
    paths: OutputPaths,
    df_meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
    label_col: str,
) -> dict:
    """Bundle the per-call args the DL training module expects."""
    return {
        "device": torch.device(cfg.device),
        "df_meta": df_meta,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "label_col": label_col,
        "loss_cfg": cfg.loss,
        "optimizer_cfg": cfg.optimizer,
        "scheduler_cfg": cfg.scheduler,
        "loops_cfg": cfg.loops,
        "models_path": paths.models,
    }


def _prepare_train_payload(
    kind: str,
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
) -> tuple[object, object]:
    """Shape (X, y) for the training module.

    ML: ``X`` is a DataFrame slice (named columns are needed by ColumnTransformer);
    ``y`` is a numpy array.
    DL: ``X`` is the full DataFrame (label_col still inside it); ``y`` unused.
    """
    if kind == "ml":
        return df[feat_cols], df[label_col].to_numpy()
    return df, None


def _build_ml_context(num_cols: list[str], cat_cols: list[str]) -> dict:
    """Pack the column groups needed by ML preprocessing pipeline."""
    return {"num_cols": num_cols, "cat_cols": cat_cols}


def _resolve_dl_params(
    name: str,
    params: dict,
    num_cols: list[str],
    cat_cols: list[str],
    num_classes: int,
) -> dict:
    """Inject fit-time params into a DL classifier's `params` from data shape.

    Keeps `configs/classifier/*.yaml` free of `${data.num_*}` interpolation; the
    values are derived from `num_cols` / `cat_cols` / `num_classes` at fit time
    and persisted in the checkpoint by `save_model`, so `load_model` works
    unchanged.
    """
    out = dict(params)
    out["num_classes"] = num_classes
    if name == "numerical":
        out["in_features"] = len(num_cols)
    elif name == "categorical":
        out["num_features"] = len(cat_cols)
    elif name == "tabular":
        out["num_numerical_features"] = len(num_cols)
        out["num_categorical_features"] = len(cat_cols)
    return out


@timed
def _train_stage(
    cfg,
    paths: OutputPaths,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    df_meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
    bus: LogDispatcher,
) -> None:
    """Run training (optionally grid search), save the model, publish summaries."""
    kind = cfg.classifier.kind
    training_mod = _resolve_training_module(kind)
    context = (
        _build_dl_context(cfg, paths, df_meta, num_cols, cat_cols, label_col)
        if kind == "dl"
        else _build_ml_context(num_cols, cat_cols)
    )

    X, y = _prepare_train_payload(kind, train_df, feat_cols, label_col)
    X_val, y_val = _prepare_train_payload(kind, val_df, feat_cols, label_col)

    params = OmegaConf.to_container(cfg.classifier.params, resolve=True)
    if kind == "dl":
        params = _resolve_dl_params(
            cfg.classifier.name,
            params,
            num_cols,
            cat_cols,
            df_meta["num_classes"],
        )

    has_grid = "grid" in cfg.classifier and len(cfg.classifier.grid) > 0
    if cfg.grid_search.enabled and has_grid:
        logger.info(
            "Grid search for %s — scoring=%s, cv=%d",
            cfg.classifier.name,
            cfg.grid_search.scoring,
            cfg.grid_search.cv,
        )
        model, summary = training_mod.grid_search_classifier(
            name=cfg.classifier.name,
            params=params,
            grid=dict(cfg.classifier.grid),
            X=X,
            y=y,
            scoring=cfg.grid_search.scoring,
            cv=cfg.grid_search.cv,
            context=context,
            max_samples=cfg.grid_search.max_samples,
        )
        logger.info(
            "Best params: %s | Best score (%s): %.4f",
            summary["best_params"],
            summary["scoring"],
            summary["best_score"],
        )
        bus.publish(LogBundle.from_dict({"json/training/grid_search": summary}))
    else:
        logger.info("Training %s ...", cfg.classifier.name)
        model, fit_summary = training_mod.fit_classifier(
            name=cfg.classifier.name,
            params=params,
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            context=context,
        )
        history = fit_summary.get("history", {})
        if history:
            bus.publish(LogBundle.from_dict(_training_history_figures(history)))

    training_mod.save_model(
        model,
        paths.models,
        name=cfg.classifier.name,
        params=params,
    )
    logger.info("Model saved under %s", paths.models)


@timed
def _evaluate_stage(
    cfg,
    paths: OutputPaths,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str,
    df_meta: dict,
    num_cols: list[str],
    cat_cols: list[str],
    bus: LogDispatcher,
) -> None:
    """Load the trained model, run predictions, publish metrics + figures + dumps."""
    kind = cfg.classifier.kind
    training_mod = _resolve_training_module(kind)
    context = (
        _build_dl_context(cfg, paths, df_meta, num_cols, cat_cols, label_col)
        if kind == "dl"
        else _build_ml_context(num_cols, cat_cols)
    )

    logger.info("Loading model from %s ...", paths.models)
    model = training_mod.load_model(paths.models, context=context)

    X = test_df[feat_cols] if kind == "ml" else test_df
    y_pred, y_proba = training_mod.predict_with_proba(model, X, context=context)

    y_true = test_df[label_col].to_numpy()
    clusters = test_df["cluster"].to_numpy() if "cluster" in test_df.columns else None
    X_np = test_df[feat_cols].to_numpy()

    full_metrics = _compute_classification_metrics(y_true, y_pred)
    pred_infos = _evaluate_predictions(y_true, y_pred, y_proba, clusters)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true), normalize="true")
    figures = _build_test_figures(X_np, y_true, y_pred, df_meta["label_mapping"])

    bus.publish(
        LogBundle.from_dict(
            {
                **figures,
                "json/testing/summary": full_metrics,
                "json/analysis/predictions/test": pred_infos,
                "pickle/analysis/confusion_matrices/test": cm,
            }
        )
    )


def _explain(
    cfg,
            paths,
            train_df,
            test_df,
            val_df,
            feat_cols,
            label_col,
            df_meta,
            num_cols,
            cat_cols,
            bus,
        ) -> None:
    "Run some XAI on the model"
    logger.info("XAI Stage")

    # build the model
    kind = cfg.classifier.kind
    training_mod = _resolve_training_module(kind)
    context = (
        _build_dl_context(cfg, paths, df_meta, num_cols, cat_cols, label_col)
        if kind == "dl"
        else _build_ml_context(num_cols, cat_cols)
    )

    logger.info("Loading model from %s ...", paths.models)
    model = training_mod.load_model(paths.models, context=context)
    model.eval()

    # build a feature-only dataframe for SHAP and preserve feature dtypes
    X, y = _prepare_train_payload(kind, train_df, feat_cols, label_col)
    X_ref = X if kind == "ml" else X[feat_cols]

    y_true = test_df[label_col].to_numpy()
    classes = np.unique(y_true)
    class_names = [df_meta["label_mapping"].get(str(int(c)), str(c)) for c in classes]
    
    FEATURE_NAMES = X_ref.columns
    FEATURE_DTYPES = X_ref.dtypes.to_dict()

    def shap_predict(x):
        # x is a numpy array from SHAP
        X_batch = pd.DataFrame(x, columns=FEATURE_NAMES)
        X_batch = X_batch.astype(FEATURE_DTYPES)

        _, y_proba = training_mod.predict_with_proba(model, X_batch, context=context)
        if hasattr(y_proba, "detach"):
            y_proba = y_proba.detach().cpu().numpy()
        return y_proba

    # make the sample set for SHAP using the same feature columns as training
    X_background = X_ref.sample(n=min(100, len(X_ref)), random_state=0)
    background_np = X_background.to_numpy()

    N = cfg.explain.num_samples
    X_test_small = test_df[feat_cols].sample(n=min(N, len(test_df)), random_state=0)

    explainer = shap.KernelExplainer(shap_predict, X_background)
    shap_values = explainer(X_test_small)
   
    
    if isinstance(shap_values, np.ndarray):
        shap_values = [shap_values]

    
    for k in range(train_df[label_col].nunique()):
        logger.info(f"SHAP beeswarm for class {class_names[k]}")
        shap.plots.beeswarm(shap_values[:, :, k], show=False)
        plt.title(f"SHAP Beeswarm for class '{class_names[k]}'", fontsize=14)
        plt.savefig(f"{paths.figures}/shap_beeswarm_{class_names[k]}.png", dpi=300, bbox_inches="tight")
        plt.clf()

@timed
def classify(cfg) -> None:
    """Run the supervised classification pipeline for a single classifier."""
    paths = OutputPaths(
        processed_data=Path(cfg.path.processed_data),
        shared=Path(cfg.path.shared),
        configs=Path(cfg.path.configs),
        outputs=Path(cfg.path.outputs),
        pickle=Path(cfg.path.pickle),
        models=Path(cfg.path.models),
        figures=Path(cfg.path.figures),
    )    
    
    df_meta = load_from_json(paths.shared / "metadata/df_meta.json")
    save_config(cfg, paths.configs / "config_composed.json")

    num_cols = list(cfg.data.num_cols) + list(cfg.data.complexity_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    #complexity_cols = list(cfg.data.complexity_cols) if cfg.data.complexity_cols else []
    label_col = "encoded_" + cfg.data.label_col
    feat_cols = num_cols + cat_cols

    data = DataConfig(
        processed_data_path=paths.processed_data,
        extension=cfg.data.extension,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
        n_samples=cfg.n_samples,
    )

    stage = cfg.stage
    if stage not in ("all", "training", "testing"):
        logger.error(
            "Unknown stage: %r. Valid: 'all', 'training', 'testing'.",
            stage,
        )
        sys.exit(1)

    train_df, val_df, test_df = _load_data(data, cfg.seed)
    logger.info(
        "Data loaded — train: %d, val: %d, test: %d samples",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info("Classifier: %s (kind=%s)", cfg.classifier.name, cfg.classifier.kind)

    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.outputs))
    bus.subscribe(PickleSubscriber(paths.pickle))
    bus.subscribe(FilesystemFigureSubscriber(paths.figures))

    if stage in ("training", "all"):
        _train_stage(
            cfg,
            paths,
            train_df,
            val_df,
            feat_cols,
            label_col,
            df_meta,
            num_cols,
            cat_cols,
            bus,
        )

    if stage in ("testing", "all"):
        _evaluate_stage(
            cfg,
            paths,
            test_df,
            feat_cols,
            label_col,
            df_meta,
            num_cols,
            cat_cols,
            bus,
        )

    if cfg.explain.generate:
        _explain(
            cfg,
            paths,
            train_df,
            test_df,
            val_df,
            feat_cols,
            label_col,
            df_meta,
            num_cols,
            cat_cols,
            bus,
        )

    logger.info("All stages completed.")


def main():
    """Main entry point for supervised classification."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    classify(cfg)
    flush_timing(Path(cfg.path.outputs) / "timing.json")


if __name__ == "__main__":
    main()
