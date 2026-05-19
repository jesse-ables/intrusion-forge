from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from ..common.utils import load_from_joblib, save_to_joblib
from .classifier import MLClassifierFactory


def fit_classifier(
    name: str,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    context: dict | None = None,
) -> tuple[BaseEstimator, dict]:
    """Instantiate via factory and fit on (X, y).

    `X_val`/`y_val`/`context` are accepted for interface parity with the DL
    training module but are ignored — sklearn estimators do not consume a
    validation set during fit.

    Returns ``(model, {})``.
    """
    model = MLClassifierFactory.create(name, params)
    model.fit(X, y)
    return model, {}


def grid_search_classifier(
    name: str,
    params: dict,
    grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    *,
    scoring: str = "f1_macro",
    cv: int = 5,
    n_jobs: int = -1,
    context: dict | None = None,
) -> tuple[BaseEstimator, dict]:
    """Cross-validated grid search starting from a base classifier built with `params`.

    Returns ``(best_estimator, summary)``. `summary` keys: ``best_params``,
    ``best_score``, ``scoring``, ``cv``, ``cv_results`` (slim: param combos +
    mean/std test scores). `context` is ignored.
    """
    base = MLClassifierFactory.create(name, params)
    search = GridSearchCV(
        base,
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)

    cv_results = [
        {
            "params": dict(p),
            "mean_test_score": float(s),
            "std_test_score": float(std),
        }
        for p, s, std in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        )
    ]

    summary = {
        "best_params": dict(search.best_params_),
        "best_score": float(search.best_score_),
        "scoring": scoring,
        "cv": cv,
        "cv_results": cv_results,
    }
    return search.best_estimator_, summary


def predict_with_proba(
    model: BaseEstimator,
    X: np.ndarray,
    *,
    context: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(predicted labels, class probability matrix)``."""
    return model.predict(X), model.predict_proba(X)


def save_model(
    model: BaseEstimator,
    path: Path,
    *,
    name: str = "",
    params: dict | None = None,
) -> None:
    """Save the estimator to ``path / 'model.joblib'``."""
    save_to_joblib(model, Path(path) / "model.joblib")


def load_model(path: Path, *, context: dict | None = None) -> BaseEstimator:
    """Load the estimator from ``path / 'model.joblib'``."""
    return load_from_joblib(Path(path) / "model.joblib")
