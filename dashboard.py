"""Interactive dashboard for intrusion-forge experiment results.

Usage:
    streamlit run dashboard.py
    make dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────── constants ────────────────────────────────────

EXPERIMENTS_ROOT = Path(__file__).parent / "resources" / "experiments"
_NON_FEATURE_COLS = frozenset(
    {"cluster_id", "cluster_class", "is_noise_cluster", "failure_rate", "is_failed"}
)

# ─────────────────────────────── data loading ─────────────────────────────────


def _read_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _load_experiment(variant_path: Path, dataset_dir: str, run: int = 0) -> dict | None:
    """Load all JSON logs for one dataset / run. Returns None if no data found."""
    logs = variant_path / dataset_dir / str(run) / "logs"

    testing = _read_json(logs / "testing" / "summary.json")
    classifier = _read_json(logs / "analysis" / "classifier_results.json")
    cluster_raw = _read_json(logs / "analysis" / "cluster_summary.json")
    predictions = _read_json(logs / "analysis" / "predictions" / "test.json")
    meta = _read_json(logs / "data" / "df_meta.json")

    if testing is None and classifier is None:
        return None

    clusters_df: pd.DataFrame | None = None
    if cluster_raw:
        rows = [
            {"cluster_id": cid, **features} for cid, features in cluster_raw.items()
        ]
        clusters_df = pd.DataFrame(rows)

    return {
        "testing": testing,
        "classifier": classifier,
        "clusters_df": clusters_df,
        "predictions": predictions,
        "meta": meta,
    }


@st.cache_data
def load_all_experiments(experiments_root: str) -> dict[str, dict[str, dict]]:
    """Discover all variant folders and datasets, return nested dict.

    Structure: variant_name -> dataset_dir -> experiment_dict.
    """
    root = Path(experiments_root)
    variants = sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )

    result: dict[str, dict[str, dict]] = {}
    for variant in variants:
        variant_path = root / variant
        datasets = sorted(
            d.name
            for d in variant_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        result[variant] = {}
        for ds_dir in datasets:
            exp = _load_experiment(variant_path, ds_dir)
            if exp is not None:
                result[variant][ds_dir] = exp

    return result


def _dataset_label(ds_dir: str) -> str:
    """Strip trailing _<seed> suffix: 'bank_marketing_42' -> 'bank_marketing'."""
    parts = ds_dir.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else ds_dir


def _class_name(label_map: dict, class_int: int) -> str:
    return label_map.get(str(class_int), f"class {class_int}")


# ─────────────────────────────── summary builder ──────────────────────────────


def _build_summary_df(variant_data: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for ds_dir, exp in variant_data.items():
        meta = exp.get("meta") or {}
        testing = exp.get("testing") or {}
        clf = exp.get("classifier") or {}
        cdf: pd.DataFrame | None = exp.get("clusters_df")

        sizes = meta.get("dataset_sizes", {})
        total_samples = sum(sizes.values()) if sizes else None
        n_clusters = len(cdf) if cdf is not None else None
        n_failed = (
            int(cdf["is_failed"].sum())
            if cdf is not None and "is_failed" in cdf.columns
            else None
        )

        rows.append(
            {
                "dataset": _dataset_label(ds_dir),
                "ds_dir": ds_dir,
                "num_classes": meta.get("num_classes"),
                "total_samples": total_samples,
                "num_clusters": n_clusters,
                "num_failed_clusters": n_failed,
                "nn_accuracy": testing.get("accuracy"),
                "nn_f1_macro": testing.get("f1_macro"),
                "nn_f1_weighted": testing.get("f1_weighted"),
                "fc_f1": clf.get("f1_score"),
                "fc_f1_std": clf.get("f1_score_std"),
                "clf_auc": clf.get("roc_auc"),
                "clf_auc_std": clf.get("roc_auc_std"),
            }
        )
    return pd.DataFrame(rows)


def _build_nn_metrics_df(variant_data: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for ds_dir, exp in variant_data.items():
        t = exp.get("testing") or {}
        rows.append(
            {
                "dataset": _dataset_label(ds_dir),
                "precision_macro": t.get("precision_macro"),
                "recall_macro": t.get("recall_macro"),
                "f1_macro": t.get("f1_macro"),
                "precision_weighted": t.get("precision_weighted"),
                "recall_weighted": t.get("recall_weighted"),
                "f1_weighted": t.get("f1_weighted"),
            }
        )
    return pd.DataFrame(rows)


def _build_fc_metrics_df(variant_data: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for ds_dir, exp in variant_data.items():
        clf = exp.get("classifier") or {}
        report = clf.get("classification_report") or {}
        cm = clf.get("confusion_matrix")

        macro = report.get("macro avg") or {}
        weighted = report.get("weighted avg") or {}

        tn = fp = fn = tp = None
        if cm and len(cm) == 2:
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]

        rows.append(
            {
                "dataset": _dataset_label(ds_dir),
                "precision_macro": macro.get("precision"),
                "recall_macro": macro.get("recall"),
                "f1_macro": macro.get("f1-score"),
                "precision_weighted": weighted.get("precision"),
                "recall_weighted": weighted.get("recall"),
                "f1_weighted": weighted.get("f1-score"),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────── plot helpers ─────────────────────────────────


def _bar_chart(
    df: pd.DataFrame, x: str, y: str, err: str | None, title: str, color: str
) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=df[x],
            y=df[y],
            error_y=(
                dict(type="data", array=df[err].fillna(0).tolist())
                if err and err in df
                else {}
            ),
            marker_color=color,
        )
    )
    fig.update_layout(title=title, xaxis_title="Dataset", yaxis_title=y, height=360)
    return fig


def _stripe_rows(styler: "pd.io.formats.style.Styler") -> "pd.io.formats.style.Styler":
    """Alternating row background shading for readability."""
    return styler.apply(
        lambda row: pd.Series(
            [
                (
                    "background-color: rgba(128, 128, 128, 0.08)"
                    if row.name % 2 == 0
                    else ""
                )
                for _ in row
            ],
            index=row.index,
        ),
        axis=1,
    )


def _confusion_matrix_fig(cm: list[list[int]], labels: list[str]) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
            text=cm,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis=dict(autorange="reversed"),
        height=350,
    )
    return fig


def _strip_box_fig(
    cdf: pd.DataFrame,
    label_map: dict,
    oof_preds: dict,
) -> go.Figure:
    """Strip-box plot: failure_rate per class, points colored by failure_rate,
    marker border colored by RF prediction correctness."""
    plot_df = cdf[["cluster_id", "cluster_class", "failure_rate"]].copy()
    plot_df["class_name"] = plot_df["cluster_class"].map(
        lambda c: _class_name(label_map, c)
    )
    plot_df["rf_correct"] = plot_df["cluster_id"].apply(
        lambda cid: oof_preds.get(str(cid), float("nan"))
    )

    class_order = sorted(plot_df["class_name"].unique())
    class_to_x = {c: i for i, c in enumerate(class_order)}
    rng = np.random.default_rng(42)
    plot_df["x_pos"] = [
        class_to_x[c] + rng.uniform(-0.25, 0.25) for c in plot_df["class_name"]
    ]

    fig = go.Figure()

    # box per class — distribution only, no individual points
    for cls in class_order:
        mask = plot_df["class_name"] == cls
        fig.add_trace(
            go.Box(
                x=[class_to_x[cls]] * int(mask.sum()),
                y=plot_df.loc[mask, "failure_rate"].tolist(),
                name=cls,
                showlegend=False,
                boxpoints=False,
                line_color="rgba(100,100,100,0.55)",
                fillcolor="rgba(200,200,200,0.15)",
                width=0.35,
            )
        )

    # scatter points — colored by RF prediction (grey=correct, red=wrong)
    fill_color = [
        "#d62728" if v == 0.0 else ("#aaaaaa" if v == 1.0 else "#cccccc")
        for v in plot_df["rf_correct"]
    ]
    fig.add_trace(
        go.Scatter(
            x=plot_df["x_pos"].tolist(),
            y=plot_df["failure_rate"].tolist(),
            mode="markers",
            text=plot_df["cluster_id"].astype(str).tolist(),
            customdata=plot_df[["class_name", "rf_correct"]].values.tolist(),
            hovertemplate=(
                "<b>Cluster %{text}</b><br>"
                "Class: %{customdata[0]}<br>"
                "Failure rate: %{y:.3f}<br>"
                "RF prediction: %{customdata[1]}<extra></extra>"
            ),
            marker=dict(
                color=fill_color,
                size=9,
                line=dict(color="white", width=1),
            ),
            showlegend=False,
            name="",
        )
    )

    # legend entries
    for label, color in [
        ("RF correct", "#aaaaaa"),
        ("RF wrong", "#d62728"),
        ("RF: n/a", "#cccccc"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=9, color=color, line=dict(color="white", width=1)),
                name=label,
                showlegend=True,
            )
        )

    fig.update_layout(
        title="Failure Rate per Class",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(class_order))),
            ticktext=class_order,
            tickangle=-30,
        ),
        yaxis_title="Failure Rate",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=430,
    )
    return fig


# ══════════════════════════════════ APP ═══════════════════════════════════════

st.set_page_config(
    page_title="Intrusion Forge — Experiments",
    layout="wide",
)
st.title("Experiment Dashboard")

with st.spinner("Loading experiments…"):
    all_data = load_all_experiments(str(EXPERIMENTS_ROOT))

if not all_data:
    st.error(f"No experiment variants found in `{EXPERIMENTS_ROOT}`")
    st.stop()

variants = list(all_data.keys())

tab_overview, tab_drilldown, tab_comparison = st.tabs(
    ["Overview", "Drill-Down", "Variant Comparison"]
)

# ══════════════════════════ TAB 1 — OVERVIEW ══════════════════════════════════

with tab_overview:
    st.subheader("Cross-dataset summary")

    variant_ov = st.selectbox("Variant", variants, key="ov_variant")
    vdata_ov = all_data[variant_ov]
    summary_df = _build_summary_df(vdata_ov)

    # ── Table 1 — high-level summary ──────────────────────────────────────────
    display_df = summary_df[
        [
            "dataset",
            "num_classes",
            "total_samples",
            "num_clusters",
            "num_failed_clusters",
            "nn_f1_macro",
            "fc_f1",
        ]
    ].rename(columns={"nn_f1_macro": "nn_f1", "fc_f1": "fc_f1"})
    st.dataframe(
        _stripe_rows(
            display_df.style.format({"nn_f1": "{:.3f}", "fc_f1": "{:.3f}"}, na_rep="—")
        ),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            _bar_chart(
                summary_df,
                "dataset",
                "clf_auc",
                "clf_auc_std",
                "Failure Classifier AUC",
                "#636EFA",
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            _bar_chart(
                summary_df,
                "dataset",
                "nn_f1_macro",
                None,
                "NN Classifier Macro-F1",
                "#EF553B",
            ),
            use_container_width=True,
        )

    # ── Table 2 — NN aggregated metrics ───────────────────────────────────────
    st.subheader("Neural Network — aggregated metrics")
    nn_metrics_df = _build_nn_metrics_df(vdata_ov)
    nn_fmt = {c: "{:.3f}" for c in nn_metrics_df.columns if c != "dataset"}
    st.dataframe(
        _stripe_rows(nn_metrics_df.style.format(nn_fmt, na_rep="—")),
        use_container_width=True,
        hide_index=True,
    )

    # ── Table 3 — Failure Classifier aggregated metrics ───────────────────────
    st.subheader("Failure Classifier — aggregated metrics")
    fc_metrics_df = _build_fc_metrics_df(vdata_ov)
    fc_float_cols = [
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    fc_fmt = {c: "{:.3f}" for c in fc_float_cols}
    fc_fmt.update({"TP": "{:.0f}", "FP": "{:.0f}", "TN": "{:.0f}", "FN": "{:.0f}"})
    st.dataframe(
        _stripe_rows(fc_metrics_df.style.format(fc_fmt, na_rep="—")),
        use_container_width=True,
        hide_index=True,
    )

# ══════════════════════════ TAB 2 — DRILL-DOWN ════════════════════════════════

with tab_drilldown:
    col_v, col_d = st.columns([1, 3])
    with col_v:
        variant_dd = st.selectbox("Variant", variants, key="dd_variant")
    with col_d:
        datasets_dd = list(all_data[variant_dd].keys())
        ds_sel = st.selectbox(
            "Dataset", datasets_dd, format_func=_dataset_label, key="dd_dataset"
        )

    exp = all_data[variant_dd][ds_sel]
    testing = exp.get("testing") or {}
    clf = exp.get("classifier") or {}
    cdf: pd.DataFrame | None = exp.get("clusters_df")
    meta = exp.get("meta") or {}
    predictions = exp.get("predictions") or {}
    label_map: dict = meta.get("label_mapping", {})

    # ── Panel A — NN Test Performance ─────────────────────────────────────────
    st.divider()
    st.subheader("Neural Network Test Performance")

    if testing:
        n_classes = len(testing.get("f1_per_class", []))
        class_labels = [label_map.get(str(i), str(i)) for i in range(n_classes)]

        per_class_df = pd.DataFrame(
            {
                "class": class_labels,
                "precision": testing.get("precision_per_class", []),
                "recall": testing.get("recall_per_class", []),
                "f1": testing.get("f1_per_class", []),
            }
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = px.bar(
                per_class_df.melt(
                    id_vars="class", value_vars=["precision", "recall", "f1"]
                ),
                x="class",
                y="value",
                color="variable",
                barmode="group",
                title="Per-class Precision / Recall / F1",
                labels={"value": "Score", "variable": "Metric", "class": "Class"},
                color_discrete_map={
                    "precision": "#636EFA",
                    "recall": "#EF553B",
                    "f1": "#00CC96",
                },
            )
            fig.update_layout(height=380)
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Accuracy", f"{testing.get('accuracy') or 0:.3f}")
            st.metric("Macro F1", f"{testing.get('f1_macro') or 0:.3f}")
            st.metric("Weighted F1", f"{testing.get('f1_weighted') or 0:.3f}")

        with st.expander("Compare with other datasets"):
            _odss_all = [d for d in datasets_dd if d != ds_sel]
            _odss = st.multiselect(
                "Datasets", _odss_all, default=_odss_all, key="cmp_ds_a"
            )
            if _odss:
                _ccols = st.columns(min(3, len(_odss)))
                for _i, _ods in enumerate(_odss):
                    _ot = all_data[variant_dd][_ods].get("testing") or {}
                    _om = all_data[variant_dd][_ods].get("meta") or {}
                    _olm = _om.get("label_mapping", {})
                    _n = len(_ot.get("f1_per_class", []))
                    with _ccols[_i % len(_ccols)]:
                        if _n:
                            _cpd = pd.DataFrame(
                                {
                                    "class": [
                                        _olm.get(str(j), str(j)) for j in range(_n)
                                    ],
                                    "precision": _ot.get("precision_per_class", []),
                                    "recall": _ot.get("recall_per_class", []),
                                    "f1": _ot.get("f1_per_class", []),
                                }
                            )
                            _fig = px.bar(
                                _cpd.melt(
                                    id_vars="class",
                                    value_vars=["precision", "recall", "f1"],
                                ),
                                x="class",
                                y="value",
                                color="variable",
                                barmode="group",
                                title=_dataset_label(_ods),
                                color_discrete_map={
                                    "precision": "#636EFA",
                                    "recall": "#EF553B",
                                    "f1": "#00CC96",
                                },
                            )
                            _fig.update_layout(height=260, showlegend=(_i == 0))
                            _fig.update_xaxes(tickangle=-30)
                            st.plotly_chart(_fig, use_container_width=True)
                        else:
                            st.caption(f"{_dataset_label(_ods)}: no data")
    else:
        st.info("No `testing/summary.json` found for this experiment.")

    # ── Panel B — Failure Classifier ──────────────────────────────────────────
    st.divider()
    st.subheader("Failure Classifier")

    if clf:
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            roc = clf.get("roc_curve_data", {})
            if roc:
                fig = go.Figure()
                fig.add_scatter(
                    x=roc["fpr"],
                    y=roc["tpr"],
                    mode="lines",
                    name=f"ROC (AUC={clf.get('roc_auc') or 0:.3f})",
                    line=dict(color="#636EFA", width=2),
                )
                fig.add_scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    name="Random",
                )
                fig.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            cm = clf.get("confusion_matrix")
            if cm:
                fig = _confusion_matrix_fig(cm, labels=["Correct", "Failed"])
                fig.update_layout(title="Confusion Matrix (OOF predictions)")
                st.plotly_chart(fig, use_container_width=True)

        with col3:
            auc = clf.get("roc_auc") or 0
            auc_std = clf.get("roc_auc_std") or 0
            f1 = clf.get("f1_score") or 0
            f1_std = clf.get("f1_score_std") or 0
            st.metric("AUC", f"{auc:.3f}", delta=f"±{auc_std:.3f}")
            st.metric("F1", f"{f1:.3f}", delta=f"±{f1_std:.3f}")
            folds = clf.get("f1_scores_per_fold", [])
            if folds:
                st.caption("F1 per fold")
                for i, v in enumerate(folds):
                    st.caption(f"  fold {i+1}: {v:.3f}")

        with st.expander("Compare with other datasets"):
            _odss_all = [d for d in datasets_dd if d != ds_sel]
            _odss = st.multiselect(
                "Datasets", _odss_all, default=_odss_all, key="cmp_ds_b"
            )
            if _odss:
                _ccols = st.columns(min(3, len(_odss)))
                for _i, _ods in enumerate(_odss):
                    _oc = all_data[variant_dd][_ods].get("classifier") or {}
                    _or = _oc.get("roc_curve_data", {})
                    _cm = _oc.get("confusion_matrix")
                    with _ccols[_i % len(_ccols)]:
                        if _or or _cm:
                            st.caption(_dataset_label(_ods))
                            if _or:
                                _fig = go.Figure()
                                _fig.add_scatter(
                                    x=_or["fpr"],
                                    y=_or["tpr"],
                                    mode="lines",
                                    name=f"AUC={_oc.get('roc_auc') or 0:.3f}",
                                    line=dict(color="#636EFA", width=2),
                                )
                                _fig.add_scatter(
                                    x=[0, 1],
                                    y=[0, 1],
                                    mode="lines",
                                    line=dict(color="gray", dash="dash"),
                                    showlegend=False,
                                )
                                _fig.update_layout(
                                    title="ROC",
                                    xaxis_title="FPR",
                                    yaxis_title="TPR",
                                    height=240,
                                    showlegend=True,
                                )
                                st.plotly_chart(_fig, use_container_width=True)
                            if _cm:
                                _fig2 = _confusion_matrix_fig(
                                    _cm, labels=["Correct", "Failed"]
                                )
                                _fig2.update_layout(
                                    title="Confusion Matrix", height=220
                                )
                                st.plotly_chart(_fig2, use_container_width=True)
                        else:
                            st.caption(f"{_dataset_label(_ods)}: no data")
    else:
        st.info("No `classifier_results.json` found for this experiment.")

    # ── Panel C — Feature Importances ─────────────────────────────────────────
    st.divider()
    st.subheader("Complexity Feature Importances")

    fi = (clf or {}).get("feature_importances")
    if fi:
        fi_df = pd.DataFrame(
            {"feature": list(fi.keys()), "importance": list(fi.values())}
        ).sort_values("importance", ascending=True)
        eps = max(fi_df["importance"].max() * 0.02, 1e-4)

        fig_c = go.Figure()
        fig_c.add_trace(
            go.Bar(
                x=fi_df["importance"].tolist(),
                y=fi_df["feature"].tolist(),
                orientation="h",
                base=-eps,
                marker=dict(
                    color=fi_df["importance"].tolist(),
                    colorscale="Blues",
                    colorbar=dict(title="Importance", thickness=12, len=0.7),
                ),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
                name="",
            )
        )
        fig_c.update_layout(
            title="Feature Importances",
            xaxis_range=[-eps * 2, fi_df["importance"].max() * 1.1],
            height=max(450, len(fi_df) * 22),
            showlegend=False,
        )
        st.plotly_chart(fig_c, use_container_width=True)

        with st.expander("Compare with other datasets"):
            _odss_all = [d for d in datasets_dd if d != ds_sel]
            _odss = st.multiselect(
                "Datasets", _odss_all, default=_odss_all, key="cmp_ds_c"
            )
            if _odss:
                _ccols = st.columns(min(3, len(_odss)))
                for _i, _ods in enumerate(_odss):
                    _ofi = (all_data[variant_dd][_ods].get("classifier") or {}).get(
                        "feature_importances"
                    ) or {}
                    with _ccols[_i % len(_ccols)]:
                        if _ofi:
                            _cfi = pd.DataFrame(
                                {
                                    "feature": list(_ofi.keys()),
                                    "importance": list(_ofi.values()),
                                }
                            ).sort_values("importance", ascending=True)
                            _ceps = max(_cfi["importance"].max() * 0.02, 1e-4)
                            _fig = go.Figure(
                                go.Bar(
                                    x=_cfi["importance"].tolist(),
                                    y=_cfi["feature"].tolist(),
                                    orientation="h",
                                    base=-_ceps,
                                    marker=dict(
                                        color=_cfi["importance"].tolist(),
                                        colorscale="Blues",
                                    ),
                                )
                            )
                            _fig.update_layout(
                                title=_dataset_label(_ods),
                                xaxis_range=[
                                    -_ceps * 2,
                                    _cfi["importance"].max() * 1.1,
                                ],
                                height=max(250, len(_cfi) * 15),
                                showlegend=False,
                            )
                            st.plotly_chart(_fig, use_container_width=True)
                        else:
                            st.caption(f"{_dataset_label(_ods)}: no data")
    else:
        st.info("Feature importances not available.")

    # ── Feature Distribution by Outcome ───────────────────────────────────────
    st.divider()
    st.subheader("Feature Distribution by Outcome")

    if fi and cdf is not None:
        _feat_opts = [
            f for f in sorted(fi, key=fi.get, reverse=True) if f in cdf.columns
        ]
        if _feat_opts:
            sel_violin_feat = st.selectbox("Feature", _feat_opts, key="violin_feat_sel")
            _vdf = cdf.copy()
            _vdf["status"] = _vdf["is_failed"].map({True: "Failed", False: "Correct"})
            fig_violin = px.violin(
                _vdf,
                x="status",
                y=sel_violin_feat,
                color="status",
                box=True,
                points="all",
                color_discrete_map={"Failed": "#EF553B", "Correct": "#636EFA"},
                title=f"{sel_violin_feat} \u2014 distribution by cluster status",
                labels={"status": "Status", sel_violin_feat: sel_violin_feat},
            )
            fig_violin.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_violin, use_container_width=True)

            with st.expander("Compare with other datasets"):
                _odss_all = [d for d in datasets_dd if d != ds_sel]
                _odss = st.multiselect(
                    "Datasets", _odss_all, default=_odss_all, key="cmp_ds_v"
                )
                if _odss:
                    _ccols = st.columns(min(3, len(_odss)))
                    for _i, _ods in enumerate(_odss):
                        _ocdf = all_data[variant_dd][_ods].get("clusters_df")
                        with _ccols[_i % len(_ccols)]:
                            if _ocdf is not None and sel_violin_feat in _ocdf.columns:
                                _cvdf = _ocdf.copy()
                                _cvdf["status"] = _cvdf["is_failed"].map(
                                    {True: "Failed", False: "Correct"}
                                )
                                _fig = px.violin(
                                    _cvdf,
                                    x="status",
                                    y=sel_violin_feat,
                                    color="status",
                                    box=True,
                                    points="all",
                                    color_discrete_map={
                                        "Failed": "#EF553B",
                                        "Correct": "#636EFA",
                                    },
                                    title=_dataset_label(_ods),
                                )
                                _fig.update_layout(height=280, showlegend=False)
                                st.plotly_chart(_fig, use_container_width=True)
                            else:
                                st.caption(f"{_dataset_label(_ods)}: no data")
    else:
        st.info("No classifier results available for feature distribution.")

    # ── Panel D — Per-cluster scatter ─────────────────────────────────────────
    st.divider()
    st.subheader("Complexity \u2192 Failure Rate")

    if cdf is not None and not cdf.empty and "failure_rate" in cdf.columns:
        feature_cols = sorted(
            c
            for c in cdf.columns
            if c not in _NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(cdf[c])
        )

        # default x: top feature importance if available, else frac_at_risk
        default_x = "frac_at_risk"
        if fi:
            top_fi = [
                k for k in sorted(fi, key=fi.get, reverse=True) if k in feature_cols
            ]
            if top_fi:
                default_x = top_fi[0]

        x_feat = st.selectbox(
            "X axis — complexity feature",
            feature_cols,
            index=feature_cols.index(default_x) if default_x in feature_cols else 0,
            key="scatter_x",
        )

        scatter_df = cdf.copy()
        scatter_df["class_name"] = scatter_df["cluster_class"].map(
            lambda c: _class_name(label_map, c)
        )

        oof_preds_d = (clf or {}).get("oof_predictions") or {}
        scatter_df["rf_correct"] = scatter_df["cluster_id"].apply(
            lambda cid: oof_preds_d.get(str(cid), float("nan"))
        )
        scatter_df["rf_label"] = scatter_df["rf_correct"].apply(
            lambda v: (
                "RF wrong" if v == 0.0 else ("RF correct" if v == 1.0 else "Not in OOF")
            )
        )

        fig = px.scatter(
            scatter_df,
            x=x_feat,
            y="failure_rate",
            color="rf_label",
            hover_data={"cluster_id": True, "class_name": True, "failure_rate": ":.3f"},
            color_discrete_map={
                "RF wrong": "#d62728",
                "RF correct": "#aaaaaa",
                "Not in OOF": "#cccccc",
            },
            title=f"{x_feat}  vs  Failure Rate per Cluster",
            labels={"failure_rate": "Failure Rate", "rf_label": "RF prediction"},
        )
        fig.update_layout(height=420)

        event = st.plotly_chart(
            fig, use_container_width=True, on_select="rerun", key="scatter_d"
        )

        # ── Panel D2 — Cluster detail on click ────────────────────────────────
        selected_points = event.selection.points if event and event.selection else []
        if selected_points:
            pt_idx = selected_points[0].get("point_index")
            if pt_idx is not None and pt_idx < len(scatter_df):
                row = scatter_df.iloc[pt_idx]
                cid = str(row["cluster_id"])

                with st.container(border=True):
                    st.markdown(
                        f"**Cluster {cid}** — class `{row['class_name']}` — "
                        f"failure rate `{(row.get('failure_rate') or 0):.3f}` — "
                        + ("❌ Failed" if row.get("is_failed") else "✅ Correct")
                        + (" · noise cluster" if row.get("is_noise_cluster") else "")
                    )

                    # TP / FN membership
                    cls_key = str(int(row["cluster_class"]))
                    cls_preds = (predictions.get("classes") or {}).get(cls_key, {})
                    tp_ids = {str(c) for c in cls_preds.get("cluster_in_tp", [])}
                    fn_ids = {
                        str(c)
                        for pred_list in cls_preds.get("cluster_in_fn", {}).values()
                        for c in pred_list
                    }
                    membership = []
                    if cid in tp_ids:
                        membership.append("✅ present in **TP** samples")
                    if cid in fn_ids:
                        membership.append("❌ present in **FN** samples")
                    if membership:
                        st.caption("Test set: " + "  |  ".join(membership))

                    # complexity profile
                    profile_vals = {
                        c: row[c]
                        for c in feature_cols
                        if c in row.index and pd.notna(row[c])
                    }
                    profile_df = pd.DataFrame(
                        {
                            "feature": list(profile_vals.keys()),
                            "value": list(profile_vals.values()),
                        }
                    ).sort_values("value")
                    fig2 = px.bar(
                        profile_df,
                        x="value",
                        y="feature",
                        orientation="h",
                        title=f"Cluster {cid} — Full Complexity Profile",
                        color="value",
                        color_continuous_scale="RdYlGn_r",
                    )
                    fig2.update_layout(
                        height=max(300, len(profile_vals) * 14), showlegend=False
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Compare with other datasets"):
            _odss_all = [d for d in datasets_dd if d != ds_sel]
            _odss = st.multiselect(
                "Datasets", _odss_all, default=_odss_all, key="cmp_ds_d"
            )
            if _odss:
                _ccols = st.columns(min(3, len(_odss)))
                for _i, _ods in enumerate(_odss):
                    _ocdf = all_data[variant_dd][_ods].get("clusters_df")
                    with _ccols[_i % len(_ccols)]:
                        if (
                            _ocdf is not None
                            and x_feat in _ocdf.columns
                            and "failure_rate" in _ocdf.columns
                        ):
                            _cscatter = _ocdf.copy()
                            _ooof = (
                                all_data[variant_dd][_ods].get("classifier") or {}
                            ).get("oof_predictions") or {}
                            _cscatter["rf_correct"] = _cscatter["cluster_id"].apply(
                                lambda cid: _ooof.get(str(cid), float("nan"))
                            )
                            _cscatter["rf_label"] = _cscatter["rf_correct"].apply(
                                lambda v: (
                                    "RF wrong"
                                    if v == 0.0
                                    else ("RF correct" if v == 1.0 else "Not in OOF")
                                )
                            )
                            _fig = px.scatter(
                                _cscatter,
                                x=x_feat,
                                y="failure_rate",
                                color="rf_label",
                                color_discrete_map={
                                    "RF wrong": "#d62728",
                                    "RF correct": "#aaaaaa",
                                    "Not in OOF": "#cccccc",
                                },
                                title=_dataset_label(_ods),
                                labels={"failure_rate": "Failure Rate"},
                            )
                            _fig.update_layout(height=260, showlegend=False)
                            st.plotly_chart(_fig, use_container_width=True)
                        else:
                            st.caption(f"{_dataset_label(_ods)}: no data")
    else:
        st.info("No `cluster_summary.json` found for this experiment.")

    # ── Panel E — Failure rate distribution ───────────────────────────────────
    st.divider()
    st.subheader("Failure Rate Distribution")

    if cdf is not None and "failure_rate" in cdf.columns and "is_failed" in cdf.columns:
        hist_df = cdf.copy()
        hist_df["status"] = hist_df["is_failed"].map({True: "Failed", False: "Correct"})
        fig = px.histogram(
            hist_df,
            x="failure_rate",
            color="status",
            nbins=30,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map={"Failed": "#EF553B", "Correct": "#636EFA"},
            title="Distribution of Cluster Failure Rates",
            labels={"failure_rate": "Failure Rate", "status": "Status"},
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Compare with other datasets"):
            _odss_all = [d for d in datasets_dd if d != ds_sel]
            _odss = st.multiselect(
                "Datasets", _odss_all, default=_odss_all, key="cmp_ds_e"
            )
            if _odss:
                _ccols = st.columns(min(3, len(_odss)))
                for _i, _ods in enumerate(_odss):
                    _ocdf = all_data[variant_dd][_ods].get("clusters_df")
                    with _ccols[_i % len(_ccols)]:
                        if _ocdf is not None and "failure_rate" in _ocdf.columns:
                            _chist = _ocdf.copy()
                            _chist["status"] = _chist["is_failed"].map(
                                {True: "Failed", False: "Correct"}
                            )
                            _fig = px.histogram(
                                _chist,
                                x="failure_rate",
                                color="status",
                                nbins=20,
                                barmode="overlay",
                                opacity=0.75,
                                color_discrete_map={
                                    "Failed": "#EF553B",
                                    "Correct": "#636EFA",
                                },
                                title=_dataset_label(_ods),
                            )
                            _fig.update_layout(height=240, showlegend=False)
                            st.plotly_chart(_fig, use_container_width=True)
                        else:
                            st.caption(f"{_dataset_label(_ods)}: no data")

    # ── Panel F — Per-class cluster breakdown ─────────────────────────────────
    st.divider()
    st.subheader("Failure Rate per Class")

    if (
        cdf is not None
        and "cluster_class" in cdf.columns
        and "failure_rate" in cdf.columns
    ):
        cls_df = cdf.copy()
        cls_df["class_name"] = cls_df["cluster_class"].map(
            lambda c: _class_name(label_map, c)
        )

        oof_preds_f = (clf or {}).get("oof_predictions") or {}
        st.plotly_chart(
            _strip_box_fig(cls_df, label_map, oof_preds_f),
            use_container_width=True,
        )

        with st.expander("Compare with other datasets"):
            _odss_all = [d for d in datasets_dd if d != ds_sel]
            _odss = st.multiselect(
                "Datasets", _odss_all, default=_odss_all, key="cmp_ds_f"
            )
            if _odss:
                _ccols = st.columns(min(3, len(_odss)))
                for _i, _ods in enumerate(_odss):
                    _oexp = all_data[variant_dd][_ods]
                    _ocdf = _oexp.get("clusters_df")
                    _olm = (_oexp.get("meta") or {}).get("label_mapping", {})
                    _ooof = (_oexp.get("classifier") or {}).get("oof_predictions") or {}
                    with _ccols[_i % len(_ccols)]:
                        if _ocdf is not None and "cluster_class" in _ocdf.columns:
                            _ccls = _ocdf.copy()
                            _ccls["class_name"] = _ccls["cluster_class"].map(
                                lambda c: _class_name(_olm, c)
                            )
                            _fig = _strip_box_fig(_ccls, _olm, _ooof)
                            _fig.update_layout(title=_dataset_label(_ods), height=280)
                            st.plotly_chart(_fig, use_container_width=True)
                        else:
                            st.caption(f"{_dataset_label(_ods)}: no data")

    # ── Panel G — Cluster table ────────────────────────────────────────────────
    st.divider()
    st.subheader("Cluster Table")

    if cdf is not None and not cdf.empty:
        # order features by importance if available
        if fi:
            ordered_feats = [
                f for f in sorted(fi, key=fi.get, reverse=True) if f in cdf.columns
            ]
        else:
            ordered_feats = [c for c in cdf.columns if c not in _NON_FEATURE_COLS]

        top_n = st.slider(
            "Top-N complexity features", 5, min(30, len(ordered_feats)), 10, key="topn"
        )

        table_df = cdf[["cluster_id", "cluster_class"]].copy()
        table_df.insert(
            1,
            "class_name",
            cdf["cluster_class"].map(lambda c: _class_name(label_map, c)),
        )
        for col in ordered_feats[:top_n]:
            table_df[col] = cdf[col]
        table_df["failure_rate"] = cdf["failure_rate"]
        table_df["is_failed"] = cdf["is_failed"]

        num_fmt = {c: "{:.4f}" for c in ordered_feats[:top_n]}
        num_fmt["failure_rate"] = "{:.3f}"

        search_cid = st.text_input(
            "Search cluster ID", value="", key="g_search_cid"
        ).strip()
        if search_cid:
            _match = table_df[table_df["cluster_id"].astype(str) == search_cid]
            if not _match.empty:
                with st.container(border=True):
                    st.markdown(f"**Cluster {search_cid}**")
                    st.dataframe(
                        _stripe_rows(_match.style.format(num_fmt, na_rep="—")),
                        use_container_width=True,
                        hide_index=True,
                    )
                _rest = table_df[table_df["cluster_id"].astype(str) != search_cid]
                table_df = pd.concat([_match, _rest], ignore_index=True)
            else:
                st.warning(f"Cluster ID `{search_cid}` not found.")

        st.dataframe(
            _stripe_rows(table_df.style.format(num_fmt, na_rep="—")),
            use_container_width=True,
            hide_index=True,
        )

        oof_preds_g = (clf or {}).get("oof_predictions") or {}
        rf_failed_ids = {cid for cid, val in oof_preds_g.items() if val == 0}
        if rf_failed_ids:
            rf_failed_df = table_df[
                table_df["cluster_id"].astype(str).isin(rf_failed_ids)
            ]
            if not rf_failed_df.empty:
                st.markdown("**RF failed clusters**")
                st.dataframe(
                    _stripe_rows(
                        rf_failed_df.reset_index(drop=True).style.format(
                            num_fmt, na_rep="—"
                        )
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

# ══════════════════════════ TAB 3 — VARIANT COMPARISON ════════════════════════

with tab_comparison:
    st.subheader("Variant Comparison")

    if len(variants) < 2:
        st.info("Only one variant found — nothing to compare.")
    else:
        selected_variants = st.multiselect(
            "Variants to compare", variants, default=variants, key="cmp_variants"
        )

        if not selected_variants:
            st.warning("Select at least one variant.")
        else:
            combined = pd.concat(
                [
                    _build_summary_df(all_data[v]).assign(variant=v)
                    for v in selected_variants
                ],
                ignore_index=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    combined,
                    x="dataset",
                    y="nn_f1_macro",
                    color="variant",
                    barmode="group",
                    title="NN Macro-F1 by Variant",
                    labels={"nn_f1_macro": "Macro F1", "dataset": "Dataset"},
                )
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(
                    combined,
                    x="dataset",
                    y="fc_f1",
                    color="variant",
                    barmode="group",
                    error_y="fc_f1_std",
                    title="Failure Classifier F1 by Variant",
                    labels={"fc_f1": "F1", "dataset": "Dataset"},
                )
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

            # delta table — only for exactly 2 variants
            if len(selected_variants) == 2:
                st.subheader(
                    f"Delta  ({selected_variants[1]} − {selected_variants[0]})"
                )
                metric_cols = [
                    "nn_accuracy",
                    "nn_f1_macro",
                    "nn_f1_weighted",
                    "fc_f1",
                    "clf_auc",
                ]
                df_a = _build_summary_df(all_data[selected_variants[0]]).set_index(
                    "dataset"
                )
                df_b = _build_summary_df(all_data[selected_variants[1]]).set_index(
                    "dataset"
                )
                common = df_a.index.intersection(df_b.index)
                delta = (
                    df_b.loc[common, metric_cols] - df_a.loc[common, metric_cols]
                ).reset_index()
                delta.columns = ["dataset"] + [f"Δ {c}" for c in metric_cols]
                delta_fmt = {c: "{:+.4f}" for c in delta.columns if c != "dataset"}
                st.dataframe(
                    _stripe_rows(delta.style.format(delta_fmt, na_rep="—")),
                    use_container_width=True,
                    hide_index=True,
                )
