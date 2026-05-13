import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ignite.engine import Events
from ignite.handlers.tensorboard_logger import TensorboardLogger
from ignite.metrics import Average
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.common.config import load_config, save_config
from src.common.log import (
    LogBundle,
    LogDispatcher,
    JSONSubscriber,
    PickleSubscriber,
    TensorBoardSubscriber,
    setup_logger,
)
from src.common.utils import flush_timing, load_from_json, timed

from src.data.io import load_listed_dfs
from src.data.preprocessing import subsample_df

from src.ml.projection import stratified_subsample, tsne_projection

from src.plot.array import confusion_matrix_to_plot, scatter_2d
from src.plot.base import Plot
from src.plot.dict import dict_to_bar_plot

from torch.utils.data import DataLoader

from src.torch.builders import (
    create_dataloader,
    create_dataset,
    create_loss,
    create_model,
    create_optimizer,
    create_scheduler,
)
from src.torch.engine import eval_step, train_step
from src.torch.infer import df_to_tensors, get_predictions
from src.torch.module.checkpoint import load_best_checkpoint

from src.ignite.builders import EngineBuilder

setup_logger(log_file="resources/logs.txt")
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Shared data parameters for train() and evaluate()."""

    processed_data_path: Path
    extension: str
    num_cols: list[str]
    cat_cols: list[str]
    label_col: str
    n_samples: int | None


@dataclass
class OutputPaths:
    """Output and checkpoint paths for the classification pipeline."""

    models: Path
    tb_logs: Path
    json_logs: Path
    pickle: Path


METRIC_CLASSES: list[tuple[str, Callable]] = [
    ("precision", precision_score),
    ("recall", recall_score),
    ("f1", f1_score),
]


def load_data(
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
        train_df = subsample_df(train_df, data.n_samples, random_state, data.label_col)
    return train_df, val_df, test_df


def _make_loader(df, num_cols, cat_cols, label_cols, dataloader_cfg):
    return create_dataloader(
        create_dataset(df, num_cols, cat_cols, label_cols), dataloader_cfg
    )


def _build_trainer(
    model, loss_fn, optimizer, scheduler, device, max_grad_norm, tb_logger
):
    """Configure the training engine with TensorBoard logging."""
    return (
        EngineBuilder(train_step)
        .with_state(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        .with_tensorboard(
            tb_logger=tb_logger,
            tag="train",
            output_transform=lambda x: {"loss": x["loss"], "grad_norm": x["grad_norm"]},
        )
        .with_optimizer_logging(tb_logger=tb_logger, optimizer=optimizer)
        .with_weights_logging(tb_logger=tb_logger, model=model)
        .with_gradients_logging(tb_logger=tb_logger, model=model)
        .build()
    )


def _build_validator(
    model,
    loss_fn,
    device,
    tb_logger,
    trainer,
    early_stopping_patience,
    early_stopping_min_delta,
    models_path,
):
    """Configure the validation engine with early stopping and checkpointing."""
    return (
        EngineBuilder(eval_step)
        .with_state(model=model, loss_fn=loss_fn, device=device)
        .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        .with_early_stopping(
            trainer=trainer,
            metric="loss",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        )
        .with_checkpointing(
            trainer=trainer,
            checkpoint_dir=models_path,
            objects_to_save={"model": model},
            metric="loss",
        )
        .with_tensorboard(
            tb_logger=tb_logger,
            event=Events.COMPLETED,
            tag="validation",
            metric_names=["loss"],
            trainer=trainer,
        )
        .build()
    )


def _cluster_error_rates(clusters: np.ndarray, error_mask: np.ndarray) -> dict:
    """Return {cluster_id: {n_error, n_total, error_rate}} sorted by error_rate desc."""
    failed = clusters[error_mask]
    stats = {}
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


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    clusters: np.ndarray | None = None,
) -> dict:
    """
    Evaluate per-class prediction quality and cluster-level error rates.

    Returns:
        {
            "classes": {
                "<label>": {
                    "tot_failures", "tot_samples", "failure_rate",
                    "mean_confidence", "cluster_in_fn", "cluster_in_tp"
                }, ...
            },
            "clusters": {
                "global":   {<cluster_id>: {n_error, n_total, error_rate}, ...} | None,
                "by_class": {<label>: {<cluster_id>: {...}}, ...}              | None,
            },
        }
    """
    has_cluster = clusters is not None
    global_error_mask = y_true != y_pred

    cluster_errors_total = (
        _cluster_error_rates(clusters, global_error_mask) if has_cluster else None
    )
    cluster_errors_by_class = {} if has_cluster else None

    classes = {}
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
            "mean_confidence": float(confidences[mask].mean()) if n_total > 0 else None,
            "cluster_in_fn": cluster_in_fn,
            "cluster_in_tp": cluster_in_tp,
        }

    classes = dict(
        sorted(classes.items(), key=lambda x: x[1]["failure_rate"] or 0.0, reverse=True)
    )

    return {
        "classes": classes,
        "clusters": {
            "global": cluster_errors_total,
            "by_class": cluster_errors_by_class,
        },
    }


def _build_figures(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    z: np.ndarray | None,
    label_mapping: dict,
) -> dict[str, Plot]:
    """Build confusion matrix and projection figures. Returns prefixed dict."""
    figures: dict[str, Plot] = {}

    classes = np.unique(y_true)
    class_names = [label_mapping.get(str(int(c)), str(c)) for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    figures["figure/testing/confusion_matrix"] = confusion_matrix_to_plot(
        cm, class_names=class_names
    )

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_dict = {
        label_mapping.get(str(int(c)), str(c)): float(v)
        for c, v in zip(classes, f1_per_class)
    }
    figures["figure/testing/f1_per_class"] = dict_to_bar_plot(f1_dict)

    names = {int(c): label_mapping.get(str(int(c)), str(c)) for c in classes}
    correct = y_pred == y_true
    vis_idx = stratified_subsample(y_true, n_samples=2000, stratify=False)
    for tag, data in (("raw/classes", X), ("latent/classes", z)):
        if data is None:
            continue
        figures[f"figure/testing/{tag}"] = scatter_2d(
            tsne_projection(data[vis_idx], n_components=2),
            y_true[vis_idx],
            highlight_mask=~correct[vis_idx],
            names=names,
            x_label="t-SNE 1",
            y_label="t-SNE 2",
        )

    return figures


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, dict]:
    """Compute scalar and full classification metrics from predictions."""
    acc = accuracy_score(y_true, y_pred)
    scalars: dict[str, float] = {"scalar/testing/accuracy": acc}
    full_metrics: dict = {"accuracy": acc}

    for avg in ("macro", "weighted"):
        for name, fn in METRIC_CLASSES:
            val = float(fn(y_true, y_pred, average=avg, zero_division=0))
            scalars[f"scalar/testing/{name}_{avg}"] = val
            full_metrics[f"{name}_{avg}"] = val

    for name, fn in METRIC_CLASSES:
        full_metrics[f"{name}_per_class"] = fn(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()

    return scalars, full_metrics


@timed
def evaluate(
    model: nn.Module,
    inputs: list[torch.Tensor],
    y: torch.Tensor,
    X: np.ndarray,
    clusters: np.ndarray | None,
    label_mapping: dict,
    device: torch.device,
) -> dict:
    """Evaluate the model on the test set.

    Accepts pre-extracted tensors and arrays — conversion from df happens at the call site.
    Computes sklearn metrics via _compute_metrics.
    Calls evaluate_predictions for failure rates and cluster error rates.
    Calls _build_test_figures for confusion matrix and projection figures.
    """
    y_true_t, y_pred_t, z_t, confidences_t = get_predictions(model, inputs, y, device)

    y_true = y_true_t.numpy()
    y_pred = y_pred_t.numpy()
    z = z_t.numpy() if z_t is not None else None
    confidences = confidences_t.numpy()

    scalars, full_metrics = _compute_metrics(y_true, y_pred)
    pred_infos = evaluate_predictions(y_true, y_pred, confidences, clusters)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true), normalize="true")
    figures = _build_figures(X, y_true, y_pred, z, label_mapping)

    return {
        "pred_infos": pred_infos,
        "scalars": scalars,
        "figures": figures,
        "full_metrics": full_metrics,
        "confusion_matrix": cm,
    }


@timed
def train(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    paths: OutputPaths,
    max_epochs: int,
    max_grad_norm: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    device: torch.device,
) -> None:
    """Train the supervised classifier in-place.

    TensorBoard per-step logging (loss/step, grad norms, weight histograms) is
    handled internally by EngineBuilder.
    """
    logger.info("Starting training phase...")

    log_dir = paths.tb_logs / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorboardLogger(log_dir=log_dir)

    trainer = _build_trainer(
        model, loss_fn, optimizer, scheduler, device, max_grad_norm, tb_logger
    )
    validator = _build_validator(
        model,
        loss_fn,
        device,
        tb_logger,
        trainer,
        early_stopping_patience,
        early_stopping_min_delta,
        paths.models,
    )

    epoch_start: list[float] = []

    @trainer.on(Events.EPOCH_STARTED)
    def record_epoch_start(_engine):
        epoch_start.clear()
        epoch_start.append(time.perf_counter())

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        epoch_duration = time.perf_counter() - epoch_start[0]
        tb_logger.writer.add_scalar(
            "train/epoch_duration_s", epoch_duration, engine.state.epoch
        )
        logger.info(
            "Epoch [%d] Train Loss: %.6f | Duration: %.2fs",
            engine.state.epoch,
            engine.state.metrics["loss"],
            epoch_duration,
        )
        validator.run(val_loader)
        logger.info(
            "Epoch [%d] Val Loss: %.6f",
            engine.state.epoch,
            validator.state.metrics["loss"],
        )

    try:
        trainer.run(train_loader, max_epochs=max_epochs)
    finally:
        tb_logger.close()

    logger.info("Training completed.")


@timed
def classify(cfg) -> None:
    """Run supervised classification pipeline (training and/or evaluation)."""
    paths = OutputPaths(
        models=Path(cfg.path.models),
        tb_logs=Path(cfg.path.tb_logs),
        json_logs=Path(cfg.path.json_logs),
        pickle=Path(cfg.path.pickle),
    )
    df_meta = load_from_json(paths.json_logs / "data/df_meta.json")
    cfg.model.params.num_classes = df_meta["num_classes"]
    cfg.loss.params.class_weight = df_meta["class_weights"]

    save_config(cfg, Path(cfg.path.configs) / "config_composed.json")

    device = torch.device(cfg.device)
    logger.info("Using device: %s", device)

    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = "encoded_" + cfg.data.label_col
    feat_cols = num_cols + cat_cols

    data = DataConfig(
        processed_data_path=Path(cfg.path.processed_data),
        extension=cfg.data.extension,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
        n_samples=cfg.n_samples,
    )

    stage = cfg.stage

    if stage not in ("all", "training", "testing", "inference"):
        logger.error(
            "Unknown stage: %r. Valid: 'all', 'training', 'testing', 'inference'.",
            stage,
        )
        sys.exit(1)

    train_df, val_df, test_df = load_data(data, cfg.seed)
    logger.info(
        "Data loaded — train: %d, val: %d, test: %d samples",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    model = create_model(cfg.model.name, cfg.model.params, device)
    loss_fn = create_loss(cfg.loss.name, cfg.loss.params, device)
    train_loader = _make_loader(
        train_df,
        data.num_cols,
        data.cat_cols,
        [data.label_col],
        cfg.loops.training.dataloader,
    )
    val_loader = _make_loader(
        val_df,
        data.num_cols,
        data.cat_cols,
        [data.label_col],
        cfg.loops.validation.dataloader,
    )

    if stage in ("testing", "inference"):
        logger.info("Loading best checkpoint from %s ...", paths.models)
        load_best_checkpoint(paths.models, model, device)

    if stage in ("training", "all"):
        logger.info("Starting training stage ...")
        optimizer = create_optimizer(
            cfg.optimizer.name, cfg.optimizer.params, model, loss_fn
        )
        scheduler = create_scheduler(
            cfg.scheduler.name, cfg.scheduler.params, optimizer, train_loader
        )
        train(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            paths=paths,
            max_epochs=cfg.loops.training.epochs,
            max_grad_norm=cfg.loops.training.max_grad_norm,
            early_stopping_patience=cfg.loops.training.early_stopping.patience,
            early_stopping_min_delta=cfg.loops.training.early_stopping.min_delta,
            device=device,
        )
        load_best_checkpoint(paths.models, model, device)
        logger.info("Best checkpoint reloaded after training.")

    if stage in ("testing", "inference", "all"):
        logger.info("Starting evaluation stage ...")
        *inputs, y = df_to_tensors(
            test_df,
            [data.num_cols, data.cat_cols, [data.label_col]],
            [torch.float32, torch.long, torch.long],
        )
        X = test_df[feat_cols].to_numpy()
        clusters = (
            test_df["cluster"].to_numpy() if "cluster" in test_df.columns else None
        )

        eval_bus = LogDispatcher()
        tb_eval_logger = TensorboardLogger(log_dir=paths.tb_logs / "testing")
        eval_bus.subscribe(TensorBoardSubscriber(tb_eval_logger.writer))
        eval_bus.subscribe(JSONSubscriber(paths.json_logs))
        eval_bus.subscribe(PickleSubscriber(paths.pickle))
        try:
            result = evaluate(
                model,
                inputs,
                y,
                X,
                clusters,
                df_meta["label_mapping"],
                device,
            )
            eval_bus.publish(
                LogBundle.from_dict(
                    {
                        **result["scalars"],
                        **result["figures"],
                        "json/testing/summary": result["full_metrics"],
                        "json/analysis/predictions/test": result["pred_infos"],
                        "pickle/analysis/confusion_matrices/test": result[
                            "confusion_matrix"
                        ],
                    }
                )
            )
        finally:
            tb_eval_logger.close()

    logger.info("All stages completed.")


def main():
    """Main entry point for supervised classification."""
    cfg = load_config(
        config_path=Path(__file__).parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    classify(cfg)
    flush_timing(Path(cfg.path.json_logs) / "timing.json")


if __name__ == "__main__":
    main()
