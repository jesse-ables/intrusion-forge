import itertools
from collections.abc import Callable

import numpy as np
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from src.core.utils import timed

FitFn = Callable[..., np.ndarray]  # (X_num, X_cat=None, **params) -> labels
ClusterFn = Callable[[np.ndarray, np.ndarray | None], np.ndarray]  # (X_num, X_cat) -> labels


def _subsample(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    max_samples: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Random subsample of X_num (and X_cat) to at most max_samples rows."""
    n = X_num.shape[0]
    if n <= max_samples:
        return X_num, X_cat
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    return X_num[idx], (X_cat[idx] if X_cat is not None else None)


def _score_silhouette(
    X_num: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Silhouette on non-noise points only. Returns -inf on failure or < 2 clusters."""
    mask = labels != -1
    if mask.sum() < 2:
        return float("-inf")
    unique = np.unique(labels[mask])
    if len(unique) < 2:
        return float("-inf")
    try:
        return float(silhouette_score(X_num[mask], labels[mask], metric=metric))
    except Exception:
        return float("-inf")


@timed
def grid_search(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    fit_fn: FitFn,
    param_grid: dict[str, list],
    *,
    max_fit_samples: int = 50_000,
    random_state: int = 0,
    **fixed_params,
) -> dict:
    """Generic grid search over param_grid, scored by Euclidean silhouette.

    Steps:
      1. _subsample(X_num, X_cat, max_fit_samples, random_state)
      2. itertools.product over param_grid values
      3. fit_fn(sub_num, sub_cat, **combo, **fixed_params)
      4. score via silhouette on non-noise points
      5. return best combo as flat dict; RuntimeError if no valid clustering found.

    fixed_params are forwarded unchanged to fit_fn on every call.
    """
    sub_num, sub_cat = _subsample(X_num, X_cat, max_fit_samples, random_state)

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_score = float("-inf")
    best_combo: dict | None = None
    fallback_combo: dict | None = None

    for combo_values in tqdm(
        itertools.product(*values),
        total=np.prod([len(v) for v in values]),
        desc="Grid search",
    ):
        combo = dict(zip(keys, combo_values))
        try:
            labels = fit_fn(sub_num, X_cat=sub_cat, **combo, **fixed_params)
        except Exception:
            continue

        s = _score_silhouette(sub_num, labels)
        if fallback_combo is None:
            fallback_combo = combo
        if s > best_score:
            best_score = s
            best_combo = combo

    if best_combo is None:
        best_combo = fallback_combo

    if best_combo is None:
        raise RuntimeError(
            "grid_search: no valid clustering found across all parameter combinations."
        )

    return best_combo
