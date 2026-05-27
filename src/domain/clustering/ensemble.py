import numpy as np

from src.domain.clustering.base import ClusterFn


def _disambiguate_noise(labels: np.ndarray) -> np.ndarray:
    """Replace each -1 with a unique negative ID so noise points never compare equal."""
    if not np.any(labels == -1):
        return labels
    out = labels.astype(np.int64, copy=True)
    noise_idx = np.where(out == -1)[0]
    out[noise_idx] = -(np.arange(len(noise_idx)) + 2)
    return out


def compute_ensemble_labels(
    labels_list: list[np.ndarray], threshold: float = 0.5
) -> np.ndarray:
    """Greedy consensus clustering by majority vote across algorithms."""
    if not labels_list:
        raise ValueError("compute_ensemble_labels: empty labels_list")

    n = labels_list[0].shape[0]
    if any(lab.shape[0] != n for lab in labels_list):
        raise ValueError(
            f"compute_ensemble_labels: inconsistent lengths {[l.shape[0] for l in labels_list]}"
        )

    labels_arr = np.stack([_disambiguate_noise(l) for l in labels_list])

    result = np.full(n, -1, dtype=np.int64)
    next_id = 0
    for i in range(n):
        if result[i] != -1:
            continue
        agreement = (labels_arr[:, i : i + 1] == labels_arr).mean(axis=0)
        mask = (agreement >= threshold) & (result == -1)
        result[mask] = next_id
        next_id += 1
    return result


def make_ensemble_cluster_fn(
    cluster_fns: list[ClusterFn], threshold: float = 0.5
) -> ClusterFn:
    """Compose multiple ClusterFns via consensus into a single ClusterFn."""

    def _fn(X_num: np.ndarray, X_cat: np.ndarray | None = None) -> np.ndarray:
        labels_list = [fn(X_num, X_cat) for fn in cluster_fns]
        return compute_ensemble_labels(labels_list, threshold=threshold)

    return _fn
