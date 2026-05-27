import numpy as np

from src.domain.clustering import ClusteringFactory
from src.domain.clustering.base import ClusterFn, grid_search
from src.domain.clustering.ensemble import make_ensemble_cluster_fn


def _split_grid_fixed(params: dict) -> tuple[dict, dict]:
    """Split params into ({list-valued → grid}, {scalar → fixed})."""
    grid = {k: v for k, v in params.items() if isinstance(v, list)}
    fixed = {k: v for k, v in params.items() if not isinstance(v, list)}
    return grid, fixed


def _make_single_cluster_fn(
    name: str, params: dict, max_fit_samples: int, random_state: int
) -> ClusterFn:
    """Build a ClusterFn for a single registered algorithm."""
    fit_fn = ClusteringFactory.get(name)
    grid, fixed = _split_grid_fixed(params)

    def _fn(X_num: np.ndarray, X_cat: np.ndarray | None = None) -> np.ndarray:
        common = {
            "max_fit_samples": max_fit_samples,
            "random_state": random_state,
            **fixed,
        }
        if grid:
            best = grid_search(X_num, X_cat, fit_fn, grid, **common)
            return fit_fn(X_num, X_cat=X_cat, **best, **common)
        return fit_fn(X_num, X_cat=X_cat, **common)

    return _fn


def build_cluster_fn(
    algorithms: dict[str, dict],
    consensus_threshold: float,
    max_fit_samples: int,
    random_state: int,
) -> ClusterFn:
    """Build a ClusterFn from {algorithm_name: params}; ensembles when >1 key."""
    if not algorithms:
        raise ValueError("build_cluster_fn: algorithms is empty")
    fns = [
        _make_single_cluster_fn(name, params, max_fit_samples, random_state)
        for name, params in algorithms.items()
    ]
    if len(fns) == 1:
        return fns[0]
    return make_ensemble_cluster_fn(fns, threshold=consensus_threshold)
