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
    fig.update_layout(xaxis_title="Predicted", yaxis_title="True", height=350)
    return fig


def _feature_importance_fig(fi: dict[str, float], top_n: int = 20) -> go.Figure:
    fi_df = (
        pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance")
    )
    fig = px.bar(
        fi_df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top-{top_n} Complexity Features by Importance",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=max(400, top_n * 22), showlegend=False)
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

    border_color = [
        "#2ca02c" if v == 1.0 else ("#d62728" if v == 0.0 else "#aaaaaa")
        for v in plot_df["rf_correct"]
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

    # scatter points — colored by failure_rate, border by RF prediction
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
                color=plot_df["failure_rate"].tolist(),
                colorscale="RdYlGn_r",
                cmin=0.0,
                cmax=1.0,
                size=9,
                colorbar=dict(title="Failure rate", thickness=12, len=0.7),
                line=dict(color=border_color, width=2),
            ),
            showlegend=False,
            name="",
        )
    )

    # invisible legend entries for border color meaning
    for label, color in [
        ("RF: correct", "#2ca02c"),
        ("RF: wrong", "#d62728"),
        ("RF: n/a", "#aaaaaa"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=9, color="white", line=dict(color=color, width=2)),
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
st.title("Intrusion Forge — Experiment Dashboard")

with st.spinner("Loading experiments…"):
    all_data = load_all_experiments(str(EXPERIMENTS_ROOT))

if not all_data:
    st.error(f"No experiment variants found in `{EXPERIMENTS_ROOT}`")
    st.stop()

variants = list(all_data.keys())

tab_overview, tab_drilldown, tab_comparison = st.tabs(
    ["📊 Overview", "🔍 Drill-Down", "⚖️ Variant Comparison"]
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
        display_df.style.format({"nn_f1": "{:.3f}", "fc_f1": "{:.3f}"}, na_rep="—"),
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
        nn_metrics_df.style.format(nn_fmt, na_rep="—"),
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
        fc_metrics_df.style.format(fc_fmt, na_rep="—"),
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
    st.subheader("A — Neural Network Test Performance")

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
            st.metric("Accuracy", f"{testing.get('accuracy', 0):.3f}")
            st.metric("Macro F1", f"{testing.get('f1_macro', 0):.3f}")
            st.metric("Weighted F1", f"{testing.get('f1_weighted', 0):.3f}")
    else:
        st.info("No `testing/summary.json` found for this experiment.")

    # ── Panel B — Failure Classifier ──────────────────────────────────────────
    st.divider()
    st.subheader("B — Failure Classifier")

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
                    name=f"ROC (AUC={clf.get('roc_auc', 0):.3f})",
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
            auc = clf.get("roc_auc", 0)
            auc_std = clf.get("roc_auc_std", 0)
            f1 = clf.get("f1_score", 0)
            f1_std = clf.get("f1_score_std", 0)
            st.metric("AUC", f"{auc:.3f}", delta=f"±{auc_std:.3f}")
            st.metric("F1", f"{f1:.3f}", delta=f"±{f1_std:.3f}")
            folds = clf.get("f1_scores_per_fold", [])
            if folds:
                st.caption("F1 per fold")
                for i, v in enumerate(folds):
                    st.caption(f"  fold {i+1}: {v:.3f}")
    else:
        st.info("No `classifier_results.json` found for this experiment.")

    # ── Panel C — Feature Importances ─────────────────────────────────────────
    st.divider()
    st.subheader("C — Complexity Feature Importances")

    fi = (clf or {}).get("feature_importances")
    if fi:
        fig_c = _feature_importance_fig(fi, top_n=20)
        fi_event = st.plotly_chart(
            fig_c, use_container_width=True, on_select="rerun", key="fi_chart"
        )
        fi_points = fi_event.selection.points if fi_event and fi_event.selection else []
        if fi_points and cdf is not None:
            selected_feat = fi_points[0].get("y")
            if selected_feat and selected_feat in cdf.columns:
                violin_df = cdf.copy()
                violin_df["status"] = violin_df["is_failed"].map(
                    {True: "Failed", False: "Correct"}
                )
                fig_violin = px.violin(
                    violin_df,
                    x="status",
                    y=selected_feat,
                    color="status",
                    box=True,
                    points="all",
                    color_discrete_map={"Failed": "#EF553B", "Correct": "#636EFA"},
                    title=f"{selected_feat} — distribution by cluster status",
                    labels={"status": "Status", selected_feat: selected_feat},
                )
                fig_violin.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_violin, use_container_width=True)
    else:
        st.info("Feature importances not available.")

    # ── Panel D — Per-cluster scatter ─────────────────────────────────────────
    st.divider()
    st.subheader("D — Complexity → Failure Rate  (per cluster)")

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
        scatter_df["status"] = scatter_df["is_failed"].map(
            {True: "Failed", False: "Correct"}
        )
        scatter_df["class_name"] = scatter_df["cluster_class"].map(
            lambda c: _class_name(label_map, c)
        )

        fig = px.scatter(
            scatter_df,
            x=x_feat,
            y="failure_rate",
            color="status",
            hover_data={"cluster_id": True, "class_name": True, "failure_rate": ":.3f"},
            color_discrete_map={"Failed": "#EF553B", "Correct": "#636EFA"},
            title=f"{x_feat}  vs  Failure Rate per Cluster",
            labels={"failure_rate": "Failure Rate", "status": "Status"},
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
                        f"failure rate `{row.get('failure_rate', 0):.3f}` — "
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
    else:
        st.info("No `cluster_summary.json` found for this experiment.")

    # ── Panel E — Failure rate distribution ───────────────────────────────────
    st.divider()
    st.subheader("E — Failure Rate Distribution")

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

    # ── Panel F — Per-class cluster breakdown ─────────────────────────────────
    st.divider()
    st.subheader("F — Per-class Cluster Breakdown")

    if (
        cdf is not None
        and "cluster_class" in cdf.columns
        and "failure_rate" in cdf.columns
    ):
        cls_df = cdf.copy()
        cls_df["class_name"] = cls_df["cluster_class"].map(
            lambda c: _class_name(label_map, c)
        )

        class_summary = (
            cls_df.groupby("class_name", sort=False)
            .agg(
                num_clusters=("cluster_id", "count"),
                num_failed=("is_failed", "sum"),
                mean_failure_rate=("failure_rate", "mean"),
            )
            .reset_index()
        )
        class_summary["num_failed"] = class_summary["num_failed"].astype(int)

        st.dataframe(
            class_summary.style.format({"mean_failure_rate": "{:.3f}"}, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )

        oof_preds_f = (clf or {}).get("oof_predictions") or {}
        st.plotly_chart(
            _strip_box_fig(cls_df, label_map, oof_preds_f),
            use_container_width=True,
        )

    # ── Panel G — Cluster table ────────────────────────────────────────────────
    st.divider()
    st.subheader("G — Cluster Table")

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

        st.dataframe(
            table_df.style.format(num_fmt, na_rep="—"),
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
                    y="clf_auc",
                    color="variant",
                    barmode="group",
                    error_y="clf_auc_std",
                    title="Failure Classifier AUC by Variant",
                    labels={"clf_auc": "AUC", "dataset": "Dataset"},
                )
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
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
                    delta.style.format(delta_fmt, na_rep="—"),
                    use_container_width=True,
                    hide_index=True,
                )
