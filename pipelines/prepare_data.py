import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from src.core.config import load_config, save_config
from src.core.log import (
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.core.utils import flush_timing, skip_if_exists, timed

from src.domain.analysis.metadata import (
    compute_clusters_metadata,
    compute_df_metadata,
    get_df_info,
)
from src.core.io import load_df, save_df
from sklearn.preprocessing import RobustScaler

from src.domain.data.preprocessing import (
    LogTransformer,
    TopNHashEncoder,
    build_preprocessor,
    drop_nans,
    encode_labels,
    ml_split,
    query_filter,
    rare_category_filter,
    random_undersample_df,
)
from src.domain.analysis.complexity.shared import _l2_normalize
from src.domain.clustering import ClusterFn, build_cluster_fn

setup_logger(log_file="resources/logs.txt")
logger = logging.getLogger(__name__)


def _cluster_per_class(
    X_num: np.ndarray,
    y_class: np.ndarray,
    classes: list,
    cluster_fn: ClusterFn,
    metric: str = "euclidean",
) -> tuple[np.ndarray, dict[int, np.ndarray], set[int]]:
    """Per-class clustering. Returns (labels, centroids, noise_cluster_ids).

    Cluster IDs are globally unique via offset. Residual -1 noise points (single
    HDBSCAN runs only — ensemble output never has -1) are reassigned to per-class
    pseudo-clusters; their IDs are collected in noise_cluster_ids.
    """
    n = X_num.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    centroids: dict[int, np.ndarray] = {}
    offset = 0

    for cls in tqdm(classes, desc="Clustering classes"):
        mask = y_class == cls
        if not mask.any():
            continue
        X_num_cls = X_num[mask]
        X_fit_cls = _l2_normalize(X_num_cls) if metric == "cosine" else X_num_cls

        raw_labels = cluster_fn(X_fit_cls, None)

        cluster_ids = np.unique(raw_labels[raw_labels != -1])
        labels[mask] = np.where(raw_labels == -1, -1, raw_labels + offset)
        for cid in cluster_ids:
            centroids[int(cid + offset)] = X_num_cls[raw_labels == cid].mean(axis=0)
        if len(cluster_ids) > 0:
            offset += int(cluster_ids.max()) + 1

    # reassign noise points (-1) to per-class pseudo-clusters
    noise_cluster_ids: set[int] = set()
    noise_count = int((labels == -1).sum())
    if noise_count > 0:
        next_id = max(centroids.keys(), default=-1) + 1
        for noise_cls in sorted(np.unique(y_class)):
            noise_mask = (y_class == noise_cls) & (labels == -1)
            if noise_mask.any():
                labels[noise_mask] = next_id
                centroids[next_id] = X_num[noise_mask].mean(axis=0)
                noise_cluster_ids.add(next_id)
                next_id += 1

    return labels, centroids, noise_cluster_ids


@timed
def preprocess_df(
    df,
    num_cols,
    cat_cols,
    label_col,
    filter_query,
    min_cat_count,
    train_frac,
    val_frac,
    test_frac,
    random_state,
    top_n,
    hash_buckets,
):
    """Preprocess dataframe: filter, encode, scale, and split."""
    logger.info(
        "Preprocessing: %d rows, %d num_cols, %d cat_cols",
        len(df),
        len(num_cols),
        len(cat_cols),
    )
    df = drop_nans(df, num_cols + cat_cols + [label_col])
    df = query_filter(df, query=filter_query)
    df = rare_category_filter(df, [label_col], min_count=min_cat_count)

    train_df, val_df, test_df = ml_split(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
        label_col=label_col,
    )
    train_df = random_undersample_df(train_df, label_col, random_state=random_state)
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    preprocessor = build_preprocessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_steps=[
            ("log_transformer", LogTransformer()),
            ("scaler", RobustScaler()),
        ],
        cat_steps=[
            ("top_n_encoder", TopNHashEncoder(top_n=top_n, hash_buckets=hash_buckets)),
        ],
    )
    logger.info("Preprocessor: %s", preprocessor)
    preprocessor.fit(train_df)
    train_df, val_df, test_df = (
        preprocessor.transform(split) for split in [train_df, val_df, test_df]
    )

    return train_df, val_df, test_df


@timed
def prepare(cfg):
    """Prepare data given a configuration object."""
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = cfg.data.label_col

    raw_data_path = Path(cfg.path.raw_data)
    processed_data_path = Path(cfg.path.processed_data)
    data_logs_path = Path(cfg.path.shared)

    dispatcher = LogDispatcher()
    dispatcher.subscribe(JSONSubscriber(data_logs_path / "metadata"))

    logger.info("Loading and preprocessing data...")
    df = load_df(str(raw_data_path))
    logger.info("Raw data loaded: %d rows, %d columns", *df.shape)

    df_info = get_df_info(df, label_col=label_col)
    dispatcher.publish(LogBundle.from_dict({"json/df_info": df_info}))

    train_df, val_df, test_df = preprocess_df(
        df,
        num_cols,
        cat_cols,
        label_col,
        cfg.data.filter_query,
        cfg.data.min_cat_count,
        cfg.data.train_frac,
        cfg.data.val_frac,
        cfg.data.test_frac,
        cfg.seed,
        cfg.data.top_n,
        cfg.data.hash_buckets,
    )

    train_df, val_df, test_df = (
        df.reset_index(drop=True) for df in [train_df, val_df, test_df]
    )

    n_train, n_val = len(train_df), len(val_df)

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    X_num = combined[num_cols].to_numpy(dtype=np.float64)
    y_class = combined[label_col].to_numpy()
    all_classes = sorted(combined[label_col].unique().tolist())

    logger.info("Running per-class clustering...")
    algorithms = OmegaConf.to_container(cfg.clustering.algorithms, resolve=True)
    cluster_fn = build_cluster_fn(
        algorithms=algorithms,
        consensus_threshold=cfg.clustering.consensus_threshold,
        max_fit_samples=cfg.clustering.max_fit_samples,
        random_state=cfg.seed,
    )
    labels, centroids, noise_cluster_ids = _cluster_per_class(
        X_num,
        y_class,
        all_classes,
        cluster_fn=cluster_fn,
        metric=cfg.clustering.distance,
    )
    noise_count = (
        int(np.isin(labels, sorted(noise_cluster_ids)).sum())
        if noise_cluster_ids
        else 0
    )

    combined["cluster"] = labels
    train_df = combined.iloc[:n_train].reset_index(drop=True)
    val_df = combined.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = combined.iloc[n_train + n_val :].reset_index(drop=True)

    logger.info(
        "Clustering complete — %d clusters (noise reassigned: %d points into pseudo-clusters)",
        len(centroids),
        noise_count,
    )

    train_df, val_df, test_df, label_mapping = encode_labels(
        train_df, val_df, test_df, label_col, dst_label_col=f"encoded_{label_col}"
    )

    logger.info("Saving processed data...")
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        save_df(split_df, processed_data_path / f"{split_name}.{cfg.data.extension}")

    logger.info("Computing and saving metadata...")
    metadata = compute_df_metadata(
        {"train": train_df, "val": val_df, "test": test_df},
        label_col,
        num_cols,
        cat_cols,
        cfg.data.benign_tag,
        label_mapping=label_mapping,
    )
    dispatcher.publish(LogBundle.from_dict({"json/df_meta": metadata}))

    clusters_metadata = compute_clusters_metadata(
        train_df,
        val_df,
        test_df,
        label_col,
        cluster_col="cluster",
        centroids={str(k): v.tolist() for k, v in centroids.items()},
        noise_cluster_ids=sorted(noise_cluster_ids),
    )
    dispatcher.publish(LogBundle.from_dict({"json/clusters_meta": clusters_metadata}))
    logger.info("Cluster metadata saved.")

    return train_df, val_df, test_df, metadata


def main():
    """Main entry point for data preparation."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    ext = cfg.data.extension
    processed = Path(cfg.path.processed_data)
    shared = Path(cfg.path.shared)
    markers = [processed / f"{s}.{ext}" for s in ("train", "val", "test")]
    markers.append(shared / "metadata/clusters_meta.json")
    if skip_if_exists(markers, cfg.prepare.force, "prepare"):
        return

    save_config(cfg, shared / "config_composed.json")
    prepare(cfg)
    flush_timing(shared / "timing.json")


if __name__ == "__main__":
    main()
