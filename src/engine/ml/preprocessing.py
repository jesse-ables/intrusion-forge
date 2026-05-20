import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .model import MLClassifierFactory

"""Each classifier name maps to one preprocessing strategy that determines how
categorical columns are fed to the estimator:

  - "onehot":         num passthrough + OneHotEncoder on cat (linear/distance models).
  - "passthrough":    integer-encoded cat passed as-is (tree models).
  - "native_sklearn": cat dtype + classifier `categorical_features` flag (HistGB).
  - "native_xgb":     cat dtype + classifier `enable_categorical=True` (XGBoost).
  - "drop_cat":       cat columns dropped, only num reach the estimator (GaussianNB).

`build_pipeline(name, params, num_cols, cat_cols)` returns a fitted-ready
sklearn `Pipeline` wrapping the appropriate preprocessing step plus the
classifier built via `MLClassifierFactory`.
"""

CLASSIFIER_PREPROCESS: dict[str, str] = {
    "logistic_regression": "onehot",
    "lda": "onehot",
    "svm_rbf": "onehot",
    "linear_svc": "onehot",
    "knn": "onehot",
    "decision_tree": "passthrough",
    "random_forest": "passthrough",
    "hist_gradient_boosting": "native_sklearn",
    "xgboost": "native_xgb",
    "naive_bayes": "drop_cat",
}


class CappedCategoryEncoder(BaseEstimator, TransformerMixin):
    """Cast columns to pandas Categorical, capping cardinality at `max_cardinality`.

    Learns the top-N most frequent values from the training fold; values absent
    from that set become NaN (treated as missing by HistGB). Prevents the HistGB
    hard limit of 255 unique categories from being exceeded on high-cardinality
    features like ICMP_TYPE.
    """

    def __init__(self, max_cardinality: int | None = 255):
        self.max_cardinality = max_cardinality

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_: dict[str, pd.Index] = {}
        for col in X.columns:
            counts = X[col].value_counts()
            keep = len(counts) if self.max_cardinality is None else min(self.max_cardinality, len(counts))
            self.categories_[col] = counts.nlargest(keep).index
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        for col in result.columns:
            cats = self.categories_[col]
            result[col] = pd.Categorical(
                result[col].where(result[col].isin(cats)), categories=cats
            )
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_.copy()



def _build_preprocess(
    strategy: str,
    num_cols: list[str],
    cat_cols: list[str],
):
    """Return the sklearn step that prepares features for the given strategy.

    For "passthrough" returns the string ``"passthrough"`` (sklearn convention).
    For "native_*" strategies returns a column-wise cast to ``pd.Categorical``,
    which makes both HistGB (`categorical_features="from_dtype"`) and XGBoost
    (`enable_categorical=True`) detect categorical features automatically.
    """
    if strategy == "passthrough":
        return "passthrough"
    if strategy == "drop_cat":
        return ColumnTransformer([("num", "passthrough", num_cols)], remainder="drop")
    if strategy == "onehot":
        return ColumnTransformer(
            [
                ("num", "passthrough", num_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )
    if strategy == "native_sklearn":
        # CappedCategoryEncoder learns top-255 categories per feature from the
        # training fold, preventing HistGB's hard cardinality limit from firing
        # on high-cardinality features (e.g. ICMP_TYPE with 340+ unique values).
        return ColumnTransformer(
            [
                ("num", "passthrough", num_cols),
                ("cat", CappedCategoryEncoder(max_cardinality=255), cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    if strategy == "native_xgb":
        # max_cardinality=None: keep all training-fold categories, map unseen
        # test values to NaN. XGBoost raises on out-of-set categories otherwise.
        return ColumnTransformer(
            [
                ("num", "passthrough", num_cols),
                ("cat", CappedCategoryEncoder(max_cardinality=None), cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    raise ValueError(f"Unknown preprocessing strategy: {strategy!r}")


def _augment_params_for_strategy(
    strategy: str,
    params: dict,
    num_cols: list[str],
    cat_cols: list[str],
) -> dict:
    """Add classifier kwargs required by the native strategies."""
    params = dict(params)
    if strategy == "native_sklearn":
        params.setdefault("categorical_features", "from_dtype")
    elif strategy == "native_xgb":
        params.setdefault("enable_categorical", True)
        params.setdefault("tree_method", "hist")
    return params


def build_pipeline(
    name: str,
    params: dict,
    num_cols: list[str],
    cat_cols: list[str],
) -> Pipeline:
    """Build a sklearn `Pipeline([("pre", <preprocessor>), ("clf", <classifier>)])`.

    The preprocessing step is chosen via the `CLASSIFIER_PREPROCESS` table.
    """
    strategy = CLASSIFIER_PREPROCESS.get(name, "passthrough")
    pre = _build_preprocess(strategy, num_cols, cat_cols)
    full_params = _augment_params_for_strategy(strategy, params, num_cols, cat_cols)
    clf = MLClassifierFactory.create(name, full_params)
    return Pipeline([("pre", pre), ("clf", clf)])
