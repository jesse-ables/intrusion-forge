"""Interactive dashboard for intrusion-forge experiment results.

Usage:
    streamlit run dashboard.py
    make dashboard
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ══════════════════════════════ Constants ════════════════════════════════════

EXPERIMENTS_ROOT_DEFAULT = str(Path(__file__).parent / "resources" / "experiments")

RESERVED_DIRS = frozenset({"shared", "processed_data", "tb"})

HEATMAP_METRICS: dict[str, str] = {
    "f1_macro": "Test F1 macro",
    "f1_weighted": "Test F1 weighted",
    "accuracy": "Test accuracy",
    "precision_macro": "Test precision macro",
    "recall_macro": "Test recall macro",
    "fc_f1": "Failure-classifier F1",
    "fc_auc": "Failure-classifier ROC AUC",
}

GALLERY_CATEGORIES: list[str] = [
    "testing",
    "training",
    "summary",
    "summary/correlation",
    "summary/global",
]

CLUSTER_NON_FEATURE_COLS = frozenset(
    {
        "cluster_id",
        "cluster_class",
        "is_noise_cluster",
        "is_failed",
        "failure_rate",
    }
)


# ══════════════════════════════ Schema ═══════════════════════════════════════


@dataclass(frozen=True)
class ExperimentRecord:
    """Lightweight handle to one (variant, dataset, seed, classifier) experiment.

    Headline metrics are pre-parsed so the heatmap can be rendered without
    loading the heavy detail JSON/pickle.
    """

    variant: str
    dataset_dir: str
    file_name: str
    seed: int
    classifier: str
    family: Literal["ml", "dl"]
    root: Path
    shared: Path
    accuracy: float | None
    f1_macro: float | None
    f1_weighted: float | None
    precision_macro: float | None
    recall_macro: float | None
    fc_f1: float | None
    fc_auc: float | None
    fc_skipped: bool = False

    @property
    def key(self) -> str:
        return f"{self.variant}|{self.dataset_dir}|{self.classifier}"

    @property
    def label(self) -> str:
        return f"{self.variant} · {self.file_name} · {self.classifier}"


@dataclass
class ExperimentDetail:
    """All heavy artifacts for one experiment, loaded lazily."""

    testing: dict | None = None
    classifier_results: dict | None = None
    cluster_summary: pd.DataFrame | None = None
    predictions: dict | None = None
    confusion_matrix: np.ndarray | None = None
    grid_search: dict | None = None
    df_meta: dict = field(default_factory=dict)
    df_info: dict = field(default_factory=dict)
    complexity: dict = field(default_factory=dict)
    clusters_meta: dict = field(default_factory=dict)


# ══════════════════════════════ IO helpers ═══════════════════════════════════


def _read_json(path: Path) -> dict | list | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ══════════════════════════════ Discovery ════════════════════════════════════


def _parse_dataset_dir(name: str) -> tuple[str, int] | None:
    """Split `{file_name}_{seed}` → (file_name, seed). Returns None on failure."""
    head, _, tail = name.rpartition("_")
    if not head or not tail.isdigit():
        return None
    return head, int(tail)


def _classifier_family(classifier_dir: Path) -> Literal["ml", "dl"]:
    """Infer ML vs DL from config_composed.json::classifier.kind (fallback: model file)."""
    cfg = _read_json(classifier_dir / "configs" / "config_composed.json")
    if isinstance(cfg, dict):
        kind = cfg.get("classifier", {}).get("kind")
        if kind in {"ml", "dl"}:
            return kind  # type: ignore[return-value]
    if (classifier_dir / "models" / "model.joblib").exists():
        return "ml"
    if any((classifier_dir / "models").glob("*.pt")):
        return "dl"
    return "ml"


def _extract_headline_metrics(classifier_dir: Path) -> dict[str, float | bool | None]:
    """Read summary.json + classifier_results.json and return the headline metrics."""
    summary = _read_json(classifier_dir / "outputs" / "testing" / "summary.json") or {}
    fc = _read_json(classifier_dir / "outputs" / "analysis" / "classifier_results.json") or {}
    return {
        "accuracy": _safe_float(summary.get("accuracy")),
        "f1_macro": _safe_float(summary.get("f1_macro")),
        "f1_weighted": _safe_float(summary.get("f1_weighted")),
        "precision_macro": _safe_float(summary.get("precision_macro")),
        "recall_macro": _safe_float(summary.get("recall_macro")),
        "fc_f1": _safe_float(fc.get("f1_score")),
        "fc_auc": _safe_float(fc.get("roc_auc")),
        "fc_skipped": bool(fc.get("skipped", False)),
    }


def _is_classifier_dir(path: Path) -> bool:
    """Heuristic: a classifier dir has either an outputs/ or a configs/config_composed.json."""
    if not path.is_dir():
        return False
    if path.name in RESERVED_DIRS or path.name.startswith("."):
        return False
    return (path / "outputs").exists() or (path / "configs" / "config_composed.json").exists()


@st.cache_data(show_spinner="Scanning experiments…")
def discover_experiments(root: str) -> tuple[list[ExperimentRecord], int]:
    """Walk `root/<variant>/<dataset_dir>/` and emit one record per classifier.

    A variant/dataset is considered valid only when `shared/metadata/df_meta.json` exists,
    which excludes the legacy `{run_id}/logs/...` layout automatically.

    Returns `(records, n_skipped_legacy)`.
    """
    root_path = Path(root)
    records: list[ExperimentRecord] = []
    skipped = 0

    if not root_path.is_dir():
        return [], 0

    for variant_dir in sorted(root_path.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name.startswith("."):
            continue
        for dataset_dir in sorted(variant_dir.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
                continue
            parsed = _parse_dataset_dir(dataset_dir.name)
            if parsed is None:
                continue
            file_name, seed = parsed
            shared = dataset_dir / "shared"
            if not (shared / "metadata/df_meta.json").exists():
                skipped += 1
                continue

            for classifier_dir in sorted(dataset_dir.iterdir()):
                if not _is_classifier_dir(classifier_dir):
                    continue
                family = _classifier_family(classifier_dir)
                metrics = _extract_headline_metrics(classifier_dir)
                records.append(
                    ExperimentRecord(
                        variant=variant_dir.name,
                        dataset_dir=dataset_dir.name,
                        file_name=file_name,
                        seed=seed,
                        classifier=classifier_dir.name,
                        family=family,
                        root=classifier_dir,
                        shared=shared,
                        **metrics,
                    )
                )

    return records, skipped


# ══════════════════════════════ Detail loaders ═══════════════════════════════


def _cluster_summary_df(data: dict | None) -> pd.DataFrame | None:
    if not data:
        return None
    rows = []
    for cluster_id, row in data.items():
        rows.append({"cluster_id": cluster_id, **row})
    df = pd.DataFrame(rows)
    if "failure_rate" in df.columns:
        df = df.sort_values("failure_rate", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_experiment_detail(record_root: str, record_shared: str) -> ExperimentDetail:
    """Load every heavy artifact for one experiment. Returns empty fields when missing."""
    root = Path(record_root)
    shared = Path(record_shared)
    detail = ExperimentDetail(
        testing=_read_json(root / "outputs" / "testing" / "summary.json"),
        classifier_results=_read_json(root / "outputs" / "analysis" / "classifier_results.json"),
        predictions=_read_json(root / "outputs" / "analysis" / "predictions" / "test.json"),
        grid_search=_read_json(root / "outputs" / "training" / "grid_search.json"),
        df_meta=_read_json(shared / "metadata/df_meta.json") or {},
        df_info=_read_json(shared / "metadata/df_info.json") or {},
        complexity=_read_json(shared / "complexity.json") or {},
        clusters_meta=_read_json(shared / "metadata/clusters_meta.json") or {},
    )
    cs = _read_json(root / "outputs" / "analysis" / "cluster_summary.json")
    detail.cluster_summary = _cluster_summary_df(cs if isinstance(cs, dict) else None)
    cm = _read_pickle(root / "pickle" / "analysis" / "confusion_matrices" / "test.pkl")
    if isinstance(cm, np.ndarray):
        detail.confusion_matrix = cm
    elif isinstance(cm, list):
        detail.confusion_matrix = np.asarray(cm)
    return detail


@st.cache_data(show_spinner=False)
def load_figure_index(record_root: str) -> dict[str, str]:
    """Return `{relative_posix_path: absolute_path}` for every PNG under `figures/`."""
    figures_dir = Path(record_root) / "figures"
    if not figures_dir.is_dir():
        return {}
    out: dict[str, str] = {}
    for png in sorted(figures_dir.rglob("*.png")):
        rel = png.relative_to(figures_dir).as_posix()
        out[rel] = str(png)
    return out


# ══════════════════════════════ Selectors ════════════════════════════════════


def records_to_df(records: list[ExperimentRecord]) -> pd.DataFrame:
    """Flat DataFrame view of records (one row per (variant, dataset, classifier))."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "variant": r.variant,
                "dataset": r.file_name,
                "seed": r.seed,
                "classifier": r.classifier,
                "family": r.family,
                "accuracy": r.accuracy,
                "f1_macro": r.f1_macro,
                "f1_weighted": r.f1_weighted,
                "precision_macro": r.precision_macro,
                "recall_macro": r.recall_macro,
                "fc_f1": r.fc_f1,
                "fc_auc": r.fc_auc,
                "key": r.key,
            }
            for r in records
        ]
    )


def filter_records(
    records: list[ExperimentRecord],
    *,
    variants: list[str] | None = None,
    seed: int | None = None,
    datasets: list[str] | None = None,
    classifiers: list[str] | None = None,
) -> list[ExperimentRecord]:
    out = records
    if variants is not None:
        out = [r for r in out if r.variant in variants]
    if seed is not None:
        out = [r for r in out if r.seed == seed]
    if datasets is not None:
        out = [r for r in out if r.file_name in datasets]
    if classifiers is not None:
        out = [r for r in out if r.classifier in classifiers]
    return out


def find_record(records: list[ExperimentRecord], key: str) -> ExperimentRecord | None:
    for r in records:
        if r.key == key:
            return r
    return None


# ══════════════════════════════ Figure utilities ═════════════════════════════


def find_figure(record: ExperimentRecord, relative: str) -> Path | None:
    candidate = record.root / "figures" / relative
    return candidate if candidate.is_file() else None


def render_figure_if_present(record: ExperimentRecord, relative: str, caption: str) -> bool:
    path = find_figure(record, relative)
    if path is None:
        return False
    st.image(str(path), caption=caption, width="stretch")
    return True


# ══════════════════════════════ Plot builders ════════════════════════════════


def heatmap_fig(
    pivot: pd.DataFrame,
    *,
    title: str,
    metric_label: str,
) -> go.Figure:
    z = pivot.values.astype(float)
    text = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.3f}")(z))
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            text=text,
            texttemplate="%{text}",
            hovertemplate="dataset=%{y}<br>classifier=%{x}<br>"
            + metric_label
            + "=%{z:.4f}<extra></extra>",
            colorbar=dict(title=metric_label),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Classifier",
        yaxis_title="Dataset",
        height=max(280, 40 + 28 * len(pivot.index)),
        margin=dict(l=120, r=40, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def confusion_matrix_fig(cm: np.ndarray, labels: list[str], *, title: str = "") -> go.Figure:
    is_normalized = cm.dtype.kind == "f" and cm.max() <= 1.0 + 1e-6
    text_fmt = "%{z:.2f}" if is_normalized else "%{z:d}"
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate=text_fmt,
            hovertemplate="true=%{y}<br>pred=%{x}<br>value=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis=dict(autorange="reversed"),
        height=360,
    )
    return fig


def per_class_bar_fig(
    *,
    classes: list[str],
    f1: list[float],
    precision: list[float],
    recall: list[float],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="F1", x=classes, y=f1, marker_color="#1f77b4"))
    fig.add_trace(go.Bar(name="Precision", x=classes, y=precision, marker_color="#9ecae1"))
    fig.add_trace(go.Bar(name="Recall", x=classes, y=recall, marker_color="#6baed6"))
    fig.update_layout(
        barmode="group",
        height=360,
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(range=[0, 1.05])
    fig.update_xaxes(tickangle=-30)
    return fig


def feature_importance_bar(importances: dict, *, top_k: int = 20) -> go.Figure:
    pairs = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    names = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color="#2ca02c"))
    fig.update_layout(
        height=max(280, 22 * len(pairs)),
        xaxis_title="Importance",
        yaxis_title="",
        margin=dict(l=160, r=40, t=20, b=40),
    )
    return fig


def roc_curve_fig(roc_data: dict) -> go.Figure:
    fpr = roc_data.get("fpr", [])
    tpr = roc_data.get("tpr", [])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color="#d62728")))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="grey"))
    )
    fig.update_layout(
        height=360,
        xaxis_title="FPR",
        yaxis_title="TPR",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def complexity_vs_failure_scatter(cluster_df: pd.DataFrame, feature: str) -> go.Figure:
    df = cluster_df.dropna(subset=[feature, "failure_rate"])
    fig = px.scatter(
        df,
        x=feature,
        y="failure_rate",
        color="is_failed" if "is_failed" in df.columns else None,
        hover_data=["cluster_id", "cluster_class"],
        color_discrete_map={True: "#d62728", False: "#1f77b4"},
    )
    fig.update_layout(
        height=400,
        xaxis_title=feature,
        yaxis_title="Failure rate",
    )
    return fig


def failure_rate_strip(cluster_df: pd.DataFrame, label_map: dict) -> go.Figure:
    plot_df = cluster_df[["cluster_id", "cluster_class", "failure_rate"]].copy()
    plot_df["class_name"] = plot_df["cluster_class"].map(
        lambda c: label_map.get(str(int(c)), str(int(c)))
    )
    fig = px.strip(
        plot_df,
        x="class_name",
        y="failure_rate",
        color="class_name",
        hover_data=["cluster_id"],
        stripmode="overlay",
    )
    fig.update_layout(height=380, showlegend=False, yaxis_title="Failure rate")
    fig.update_xaxes(tickangle=-30, title_text="Class")
    return fig


# ══════════════════════════════ Panels ═══════════════════════════════════════


def _wkey(prefix: str, name: str, record: ExperimentRecord) -> str:
    """Stable element/widget key. Including `prefix` (view) + record.key prevents
    DuplicateElementId/DuplicateWidgetID when the same panel is rendered in
    multiple tabs or columns (e.g. side-by-side)."""
    return f"{prefix}_{name}_{record.key}"


def panel_test_performance(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Test performance**")
    if detail.testing is None:
        st.caption("No `outputs/testing/summary.json` for this run.")
        return
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{detail.testing.get('accuracy', float('nan')):.4f}")
    cols[1].metric("F1 macro", f"{detail.testing.get('f1_macro', float('nan')):.4f}")
    cols[2].metric("F1 weighted", f"{detail.testing.get('f1_weighted', float('nan')):.4f}")
    cols[3].metric("Recall macro", f"{detail.testing.get('recall_macro', float('nan')):.4f}")
    render_figure_if_present(record, "testing/f1_per_class.png", "f1_per_class.png")


def panel_confusion_matrix(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Confusion matrix**")
    labels = list((detail.df_meta.get("label_mapping") or {}).values())
    if detail.confusion_matrix is not None:
        if not labels or len(labels) != detail.confusion_matrix.shape[0]:
            labels = [str(i) for i in range(detail.confusion_matrix.shape[0])]
        st.plotly_chart(
            confusion_matrix_fig(detail.confusion_matrix, labels),
            width="stretch",
            key=_wkey(key_prefix, "cm", record),
        )
        return
    if not render_figure_if_present(record, "testing/confusion_matrix.png", "confusion_matrix.png"):
        st.caption("No confusion matrix available.")


def panel_failure_classifier(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Failure classifier (RF on cluster complexity)**")
    fc = detail.classifier_results
    if fc is None:
        st.caption("⚠️ Not computed — run `make failure-classify` to generate.")
        return
    if fc.get("skipped"):
        st.warning(
            f"🚫 **Stage skipped** — {fc.get('message', fc.get('reason'))}\n\n"
            f"Positives: {fc.get('n_positives', '?')} | "
            f"Threshold: {fc.get('threshold', '?')} | "
            f"Required: ≥{fc.get('min_required', 2)}"
        )
        return
    cols = st.columns(3)
    cols[0].metric(
        "F1",
        f"{fc.get('f1_score', float('nan')):.4f}",
        delta=f"± {fc.get('f1_score_std', 0):.3f}",
    )
    cols[1].metric(
        "ROC AUC",
        f"{fc.get('roc_auc', float('nan')):.4f}",
        delta=f"± {fc.get('roc_auc_std', 0):.3f}",
    )
    cols[2].metric("CV folds", f"{len(fc.get('f1_scores_per_fold', []))}")
    if fc.get("roc_curve_data"):
        st.plotly_chart(
            roc_curve_fig(fc["roc_curve_data"]),
            width="stretch",
            key=_wkey(key_prefix, "roc", record),
        )


def panel_feature_importances(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Feature importances (failure classifier)**")
    fc = detail.classifier_results
    if fc is None:
        st.caption("⚠️ Not computed — run `make failure-classify` to generate.")
        return
    if fc.get("skipped"):
        st.warning(f"🚫 Skipped — {fc.get('message', fc.get('reason'))}")
        return
    if "feature_importances" not in fc:
        st.caption("Malformed `classifier_results.json` (no feature_importances).")
        return
    top_k = st.slider(
        "Top-K features", 5, 50, 20, key=_wkey(key_prefix, "fi_topk", record)
    )
    st.plotly_chart(
        feature_importance_bar(fc["feature_importances"], top_k=top_k),
        width="stretch",
        key=_wkey(key_prefix, "fi_chart", record),
    )


def panel_feature_distribution(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Feature distribution across clusters**")
    cdf = detail.cluster_summary
    if cdf is None or cdf.empty:
        st.caption("No cluster_summary.json for this run.")
        return
    candidates = [c for c in cdf.columns if c not in CLUSTER_NON_FEATURE_COLS]
    if not candidates:
        st.caption("No complexity features in cluster_summary.")
        return
    default = "cluster_p5_silhouette" if "cluster_p5_silhouette" in candidates else candidates[0]
    feature = st.selectbox(
        "Feature",
        candidates,
        index=candidates.index(default),
        key=_wkey(key_prefix, "fd_feat", record),
    )
    fig = px.violin(
        cdf.dropna(subset=[feature]),
        y=feature,
        box=True,
        points="all",
        color="is_failed" if "is_failed" in cdf.columns else None,
        color_discrete_map={True: "#d62728", False: "#1f77b4"},
    )
    fig.update_layout(height=380, yaxis_title=feature)
    st.plotly_chart(fig, width="stretch", key=_wkey(key_prefix, "fd_chart", record))
    png_name = f"summary/global/{feature}.png"
    render_figure_if_present(record, png_name, png_name)


def panel_complexity_vs_failure(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Complexity feature → failure rate**")
    cdf = detail.cluster_summary
    if cdf is None or cdf.empty or "failure_rate" not in cdf.columns:
        st.caption("Need cluster_summary with `failure_rate`.")
        return
    candidates = [c for c in cdf.columns if c not in CLUSTER_NON_FEATURE_COLS]
    default = "cluster_p5_silhouette" if "cluster_p5_silhouette" in candidates else candidates[0]
    feature = st.selectbox(
        "X axis",
        candidates,
        index=candidates.index(default),
        key=_wkey(key_prefix, "sc_feat", record),
    )
    st.plotly_chart(
        complexity_vs_failure_scatter(cdf, feature),
        width="stretch",
        key=_wkey(key_prefix, "sc_chart", record),
    )


def panel_failure_rate_distribution(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Failure rate distribution**")
    cdf = detail.cluster_summary
    if cdf is None or cdf.empty or "failure_rate" not in cdf.columns:
        st.caption("No cluster_summary with `failure_rate`.")
        return
    label_map = detail.df_meta.get("label_mapping") or {}
    st.plotly_chart(
        failure_rate_strip(cdf, label_map),
        width="stretch",
        key=_wkey(key_prefix, "fr_chart", record),
    )
    render_figure_if_present(
        record, "summary/rf_prediction_strip_box.png", "summary/rf_prediction_strip_box.png"
    )


def panel_per_class_breakdown(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Per-class breakdown**")
    if detail.testing is None:
        st.caption("No `summary.json` for per-class metrics.")
        return
    f1 = detail.testing.get("f1_per_class") or []
    prec = detail.testing.get("precision_per_class") or []
    rec = detail.testing.get("recall_per_class") or []
    label_map = detail.df_meta.get("label_mapping") or {}
    classes = [label_map.get(str(i), str(i)) for i in range(len(f1))]
    st.plotly_chart(
        per_class_bar_fig(classes=classes, f1=f1, precision=prec, recall=rec),
        width="stretch",
        key=_wkey(key_prefix, "pc_chart", record),
    )


def panel_cluster_table(
    record: ExperimentRecord, detail: ExperimentDetail, key_prefix: str = "drill"
) -> None:
    st.markdown("**Cluster table** (sorted by failure rate desc)")
    cdf = detail.cluster_summary
    if cdf is None or cdf.empty:
        st.caption("No cluster_summary.json.")
        return
    label_map = detail.df_meta.get("label_mapping") or {}
    show_cols = ["cluster_id", "cluster_class", "failure_rate", "is_failed", "is_noise_cluster"]
    show_cols += [c for c in ["cluster_p5_silhouette", "cluster_frac_at_risk", "class_f1_min", "class_n1_max"] if c in cdf.columns]
    view = cdf[[c for c in show_cols if c in cdf.columns]].copy()
    if "cluster_class" in view.columns:
        view["class_name"] = view["cluster_class"].map(lambda c: label_map.get(str(int(c)), str(int(c))))
    st.dataframe(
        view, width="stretch", hide_index=True, key=_wkey(key_prefix, "cluster_tbl", record)
    )
    render_figure_if_present(
        record, "summary/cluster_risk_heatmap.png", "summary/cluster_risk_heatmap.png"
    )


def panel_sibling_classifiers(
    record: ExperimentRecord,
    all_records: list[ExperimentRecord],
    key_prefix: str = "drill",
) -> None:
    st.markdown(f"**Other classifiers on {record.file_name} (seed {record.seed})**")
    siblings = [
        r
        for r in all_records
        if r.variant == record.variant and r.file_name == record.file_name and r.seed == record.seed
    ]
    df = records_to_df(siblings)
    if df.empty:
        st.caption("No sibling classifiers found.")
        return
    view_cols = [
        "classifier",
        "family",
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "fc_f1",
        "fc_auc",
    ]
    view = df[view_cols].sort_values("f1_macro", ascending=False, na_position="last")
    styled = view.style.highlight_max(
        subset=[c for c in ["accuracy", "f1_macro", "f1_weighted", "fc_f1", "fc_auc"] if c in view.columns],
        color="rgba(46, 160, 67, 0.25)",
    ).format({c: "{:.4f}" for c in view.columns if view[c].dtype.kind == "f"}, na_rep="—")
    st.dataframe(
        styled, width="stretch", hide_index=True, key=_wkey(key_prefix, "siblings_tbl", record)
    )


def panel_training_curve(record: ExperimentRecord) -> None:
    st.markdown("**Training curve (DL)**")
    if not render_figure_if_present(record, "training/loss_curve.png", "training/loss_curve.png"):
        st.caption("Training figure not produced for this run.")


def panel_grid_search(record: ExperimentRecord, detail: ExperimentDetail) -> None:
    st.markdown("**Grid search (ML)**")
    if not detail.grid_search:
        st.caption("Grid search not run for this classifier.")
        return
    st.json(detail.grid_search, expanded=False)


# ══════════════════════════════ Tabs ═════════════════════════════════════════


def render_overview(
    records: list[ExperimentRecord],
    selected_variants: list[str],
    seed: int,
    metric: str,
) -> None:
    if not selected_variants:
        st.info("Pick at least one variant in the sidebar.")
        return

    metric_label = HEATMAP_METRICS[metric]
    for variant in selected_variants:
        rs = filter_records(records, variants=[variant], seed=seed)
        if not rs:
            st.warning(f"No records for variant `{variant}` at seed {seed}.")
            continue

        df = records_to_df(rs)
        all_datasets = sorted(df["dataset"].unique())
        all_classifiers = sorted(df["classifier"].unique())
        pivot = df.pivot_table(index="dataset", columns="classifier", values=metric, aggfunc="first")
        pivot = pivot.reindex(index=all_datasets, columns=all_classifiers)

        st.subheader(f"Variant: {variant}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Datasets", len(all_datasets))
        c2.metric("Classifiers", len(all_classifiers))
        missing = int(pivot.isna().sum().sum())
        c3.metric("Empty cells", missing)

        event = st.plotly_chart(
            heatmap_fig(pivot, title=metric_label, metric_label=metric),
            width="stretch",
            key=f"heatmap_{variant}",
            on_select="rerun",
            selection_mode="points",
        )

        clicked = _heatmap_click_target(event, pivot, variant, rs)
        if clicked is not None:
            st.session_state["drill_target"] = clicked
            st.toast(f"Drill-down armed → {clicked['variant']} · {clicked['dataset']} · {clicked['classifier']}")

        with st.expander("Show data", expanded=False):
            st.dataframe(
                pivot.style.background_gradient(cmap="viridis", vmin=0, vmax=1).format("{:.4f}", na_rep="—"),
                width="stretch",
            )


def _heatmap_click_target(
    event: Any,
    pivot: pd.DataFrame,
    variant: str,
    records: list[ExperimentRecord],
) -> dict | None:
    """Extract (variant, dataset, classifier) from a plotly_chart selection event."""
    try:
        selection = event.selection  # type: ignore[attr-defined]
        points = selection.get("points") if isinstance(selection, dict) else None
        if not points:
            return None
        p = points[0]
        dataset = pivot.index[int(p["y"])] if isinstance(p.get("y"), (int, np.integer)) else p.get("y")
        classifier = (
            pivot.columns[int(p["x"])] if isinstance(p.get("x"), (int, np.integer)) else p.get("x")
        )
        if dataset is None or classifier is None:
            return None
        match = next(
            (r for r in records if r.file_name == dataset and r.classifier == classifier),
            None,
        )
        if match is None:
            return None
        return {"variant": variant, "dataset": match.file_name, "classifier": classifier, "key": match.key}
    except (AttributeError, KeyError, IndexError, ValueError, TypeError):
        return None


def render_drilldown(records: list[ExperimentRecord], seed: int) -> None:
    if not records:
        st.info("No records to drill into.")
        return

    target = st.session_state.get("drill_target")
    variants = sorted({r.variant for r in records})
    pre_variant = target["variant"] if target and target["variant"] in variants else variants[0]

    col_v, col_d, col_c = st.columns(3)
    variant = col_v.selectbox("Variant", variants, index=variants.index(pre_variant), key="dd_variant")

    rs_v = filter_records(records, variants=[variant], seed=seed)
    datasets = sorted({r.file_name for r in rs_v})
    if not datasets:
        st.warning("No datasets for this variant at the selected seed.")
        return
    pre_dataset = target["dataset"] if target and target.get("dataset") in datasets else datasets[0]
    dataset = col_d.selectbox("Dataset", datasets, index=datasets.index(pre_dataset), key="dd_dataset")

    rs_vd = filter_records(rs_v, datasets=[dataset])
    classifiers = sorted({r.classifier for r in rs_vd})
    if not classifiers:
        st.warning("No classifier for this (variant, dataset, seed).")
        return
    pre_clf = target["classifier"] if target and target.get("classifier") in classifiers else classifiers[0]
    classifier = col_c.selectbox("Classifier", classifiers, index=classifiers.index(pre_clf), key="dd_classifier")

    record = next(r for r in rs_vd if r.classifier == classifier)
    detail = load_experiment_detail(str(record.root), str(record.shared))

    st.caption(f"`{record.root}` · family `{record.family}` · seed `{record.seed}`")

    # Pair the panels into rows. One st.columns(2) per row keeps left/right
    # aligned at the top of each row, even if the panels above had unequal
    # heights. Pairs are picked so the two panels in each row carry roughly the
    # same content (metrics+image, plotly+plotly, scatter+strip, plotly+plotly).
    row_pairs = [
        (panel_test_performance, panel_confusion_matrix),
        (panel_failure_classifier, panel_feature_importances),
        (panel_complexity_vs_failure, panel_failure_rate_distribution),
        (panel_per_class_breakdown, panel_feature_distribution),
    ]
    for left_panel, right_panel in row_pairs:
        left, right = st.columns(2, gap="medium")
        with left:
            with st.container(border=True):
                left_panel(record, detail)
        with right:
            with st.container(border=True):
                right_panel(record, detail)

    st.divider()
    with st.container(border=True):
        panel_sibling_classifiers(record, records)

    st.divider()
    with st.container(border=True):
        if record.family == "dl":
            panel_training_curve(record)
        else:
            panel_grid_search(record, detail)

    st.divider()
    with st.container(border=True):
        panel_cluster_table(record, detail)


def render_side_by_side(records: list[ExperimentRecord], seed: int) -> None:
    rs = [r for r in records if r.seed == seed]
    if not rs:
        st.info("No records at the selected seed.")
        return
    options = {r.key: r.label for r in rs}
    selected_keys = st.multiselect(
        "Pick 2–4 experiments to compare",
        options=list(options.keys()),
        format_func=lambda k: options[k],
        max_selections=4,
        key="sbs_keys",
    )
    if len(selected_keys) < 2:
        st.caption("Select at least two experiments to start comparing.")
        return

    selected = [find_record(records, k) for k in selected_keys]
    selected = [r for r in selected if r is not None]
    details = [load_experiment_detail(str(r.root), str(r.shared)) for r in selected]

    cols = st.columns(len(selected))
    for i, (col, record, detail) in enumerate(zip(cols, selected, details)):
        prefix = f"sbs{i}"
        with col:
            st.markdown(f"### {record.label}")
            st.caption(f"family `{record.family}` · seed `{record.seed}`")
            panel_test_performance(record, detail, key_prefix=prefix)
            panel_confusion_matrix(record, detail, key_prefix=prefix)
            panel_failure_classifier(record, detail, key_prefix=prefix)
            panel_per_class_breakdown(record, detail, key_prefix=prefix)
            st.divider()
            panel_cluster_table(record, detail, key_prefix=prefix)


def render_gallery(records: list[ExperimentRecord], seed: int) -> None:
    rs = [r for r in records if r.seed == seed]
    if not rs:
        st.info("No records at the selected seed.")
        return

    col_v, col_mode = st.columns([1, 2])
    variants = sorted({r.variant for r in rs})
    variant = col_v.selectbox("Variant", variants, key="gal_variant")
    rs_v = [r for r in rs if r.variant == variant]

    mode = col_mode.radio(
        "Mode",
        ["Single experiment", "Cross-experiment"],
        horizontal=True,
        key="gal_mode",
    )

    if mode == "Single experiment":
        _render_gallery_single(rs_v)
    else:
        _render_gallery_cross(rs_v)


def _render_gallery_single(rs_v: list[ExperimentRecord]) -> None:
    col_d, col_c, col_cat = st.columns([1, 1, 1])
    datasets = sorted({r.file_name for r in rs_v})
    dataset = col_d.selectbox("Dataset", datasets, key="gal_dataset")
    rs_vd = [r for r in rs_v if r.file_name == dataset]
    classifiers = sorted({r.classifier for r in rs_vd})
    if not classifiers:
        st.warning("No classifier outputs for this selection.")
        return
    classifier = col_c.selectbox("Classifier", classifiers, key="gal_classifier")
    category = col_cat.selectbox("Category", ["all", *GALLERY_CATEGORIES], key="gal_category")

    record = next(r for r in rs_vd if r.classifier == classifier)
    figures = load_figure_index(str(record.root))
    if not figures:
        st.info(f"No PNGs under `{record.root / 'figures'}`.")
        return

    if category != "all":
        figures = {rel: abs_ for rel, abs_ in figures.items() if rel.startswith(category + "/") or rel == category + ".png"}

    if not figures:
        st.info(f"No PNGs matching category `{category}`.")
        return

    st.caption(f"{len(figures)} figure(s) in `{record.root.name}/figures/`")
    rels = sorted(figures.keys())
    for i in range(0, len(rels), 3):
        cols = st.columns(3)
        for col, rel in zip(cols, rels[i : i + 3]):
            with col:
                st.image(figures[rel], caption=rel, width="stretch")


def _render_gallery_cross(rs_v: list[ExperimentRecord]) -> None:
    all_datasets = sorted({r.file_name for r in rs_v})
    all_classifiers = sorted({r.classifier for r in rs_v})

    col_d, col_c = st.columns([1, 1])
    sel_datasets = col_d.multiselect("Datasets", all_datasets, default=all_datasets, key="gal_x_datasets")
    sel_classifiers = col_c.multiselect("Classifiers", all_classifiers, default=all_classifiers, key="gal_x_classifiers")

    if not sel_datasets or not sel_classifiers:
        st.warning("Select at least one dataset and one classifier.")
        return

    rs_filtered = [
        r for r in rs_v
        if r.file_name in sel_datasets and r.classifier in sel_classifiers
    ]
    if not rs_filtered:
        st.warning("No experiments match the current selection.")
        return

    # Union of figure paths across all filtered records, indexed by (dataset|classifier)
    all_fig_paths: set[str] = set()
    fig_index: dict[str, dict[str, str]] = {}
    for r in rs_filtered:
        for rel, abs_ in load_figure_index(str(r.root)).items():
            all_fig_paths.add(rel)
            fig_index.setdefault(rel, {})[f"{r.file_name}|{r.classifier}"] = abs_

    if not all_fig_paths:
        st.info("No figures found in the selected experiments.")
        return

    col_fig, col_cat = st.columns([2, 1])
    category = col_cat.selectbox("Category", ["all", *GALLERY_CATEGORIES], key="gal_x_category")
    rels_all = sorted(all_fig_paths)
    if category != "all":
        rels_all = [p for p in rels_all if p.startswith(category + "/") or p == category + ".png"]
    if not rels_all:
        st.info(f"No PNGs matching category `{category}`.")
        return

    figure_path = col_fig.selectbox("Figure", rels_all, key="gal_x_figure")

    st.markdown(f"**`{figure_path}`**")
    cell_map = fig_index.get(figure_path, {})

    # Header row
    header_cols = st.columns([1] + [2] * len(sel_classifiers))
    header_cols[0].markdown("**Dataset \\ Classifier**")
    for i, clf in enumerate(sel_classifiers):
        header_cols[i + 1].markdown(f"**{clf}**")

    # One row per dataset
    for ds in sel_datasets:
        row_cols = st.columns([1] + [2] * len(sel_classifiers))
        row_cols[0].markdown(f"`{ds}`")
        for i, clf in enumerate(sel_classifiers):
            abs_path = cell_map.get(f"{ds}|{clf}")
            with row_cols[i + 1]:
                if abs_path:
                    st.image(abs_path, width="stretch")
                else:
                    st.caption("—")


# ══════════════════════════════ Sidebar ══════════════════════════════════════


def render_sidebar(records: list[ExperimentRecord], n_skipped: int) -> tuple[str, list[str], int, str]:
    st.sidebar.title("Filters")

    root = st.sidebar.text_input("Experiments root", value=EXPERIMENTS_ROOT_DEFAULT)

    variants = sorted({r.variant for r in records})
    selected_variants = st.sidebar.multiselect("Variants", variants, default=variants, key="sb_variants")

    seeds = sorted({r.seed for r in records if r.variant in selected_variants})
    default_seed = 42 if 42 in seeds else (seeds[0] if seeds else 42)
    if seeds:
        seed = st.sidebar.selectbox("Seed", seeds, index=seeds.index(default_seed), key="sb_seed")
    else:
        seed = default_seed
        st.sidebar.warning("No seeds found.")

    metric = st.sidebar.selectbox(
        "Heatmap metric",
        list(HEATMAP_METRICS.keys()),
        format_func=lambda k: HEATMAP_METRICS[k],
        index=0,
        key="sb_metric",
    )

    st.sidebar.divider()
    if st.sidebar.button("Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    with st.sidebar.expander("Debug", expanded=False):
        st.write(f"Valid records: **{len(records)}**")
        st.write(f"Legacy folders skipped: **{n_skipped}**")
        st.write(f"Variants: {len(variants)} · Datasets: {len({r.file_name for r in records})}")

    return root, selected_variants, seed, metric


# ══════════════════════════════ Entrypoint ═══════════════════════════════════


def main() -> None:
    st.set_page_config(page_title="Intrusion Forge — Experiments", layout="wide")
    st.title("Experiment Dashboard")

    initial_root = EXPERIMENTS_ROOT_DEFAULT
    records, n_skipped = discover_experiments(initial_root)

    root, selected_variants, seed, metric = render_sidebar(records, n_skipped)

    if root != initial_root:
        records, n_skipped = discover_experiments(root)

    if not records:
        st.error(
            "No valid experiments found. The dashboard probes for "
            "`shared/metadata/df_meta.json` under each `<variant>/<dataset>_<seed>/`. "
            f"Scanned root: `{root}`"
        )
        st.stop()

    tab_overview, tab_drill, tab_sbs, tab_gallery = st.tabs(
        ["Overview", "Drill-Down", "Side-by-Side", "Gallery"]
    )

    with tab_overview:
        render_overview(records, selected_variants, seed, metric)
    with tab_drill:
        render_drilldown(filter_records(records, variants=selected_variants), seed)
    with tab_sbs:
        render_side_by_side(filter_records(records, variants=selected_variants), seed)
    with tab_gallery:
        render_gallery(filter_records(records, variants=selected_variants), seed)


main()
