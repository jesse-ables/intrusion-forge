import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from .base import Plot, _fig_to_plot, _get_fill_cmap
from .style import (
    OUTLINE_COLORS,
    TITLE_FONTSIZE,
    TITLE_PAD,
    LABEL_FONTSIZE,
    LABEL_PAD,
    TICK_LABELSIZE,
    LEGEND_FONTSIZE,
    LEGEND_FRAMEALPHA,
    GRID_ALPHA,
)


def confusion_matrix_to_plot(
    cm: np.ndarray,
    title: str = "",
    class_names: list[str] | None = None,
    cmap: str = "Blues",
    figsize: tuple[float, float] = (8, 6),
    show_colorbar: bool = True,
    values_decimals: int | None = None,
) -> Plot:
    """Plot a confusion matrix, handling both raw counts and normalized values."""
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("`cm` must be a square 2D array (n_classes x n_classes).")

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    elif len(class_names) != n_classes:
        raise ValueError("`class_names` length must match cm.shape[0].")

    is_normalized = (cm.max() <= 1.0) or not np.all(cm == cm.astype(int))

    if values_decimals is not None:
        values_format = f".{values_decimals}f"
    elif is_normalized:
        values_format = ".2f"
    else:
        values_format = "d"  # integer format for raw counts

    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        ax=ax,
        cmap=cmap,
        values_format=values_format,
        colorbar=show_colorbar,
        im_kw={"interpolation": "nearest"},
        xticks_rotation=90,
        include_values=True,
    )
    plt.setp(ax.get_xticklabels(), ha="center")

    if show_colorbar and disp.im_ is not None:
        colorbar = disp.im_.colorbar
        if colorbar is not None:
            colorbar.set_label(
                "Proportion" if is_normalized else "Count",
                fontsize=10,
            )

    if title and is_normalized and "normaliz" not in title.lower():
        title = f"{title} (Normalized)"

    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    ax.set_xlabel("Predicted label", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel("True label", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)

    ax.tick_params(axis="x", labelsize=TICK_LABELSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)
    ax.set_aspect("equal")

    fig.tight_layout()
    return _fig_to_plot(fig)


def scatter_2d(
    X: np.ndarray,
    labels: np.ndarray,
    noise_mask: np.ndarray | None = None,
    highlight_mask: np.ndarray | None = None,
    names: dict | None = None,
    x_label: str = "Dim 1",
    y_label: str = "Dim 2",
    title: str = "",
    figsize: tuple[float, float] = (12, 10),
    marker_size: float = 45.0,
) -> Plot:
    """2D scatter with fill-color per label, optional noise background and highlight edge.

    Args:
        X:              (n, 2) array — any 2D embedding.
        labels:         (n,) integer labels mapped to fill color.
        noise_mask:     Boolean mask; those samples are drawn as small grey dots (background).
        highlight_mask: Boolean mask; those samples get a red border (e.g. misclassified).
        names:          Optional {label_int: str} for a legend. No legend if None.
        x_label:        X-axis label.
        y_label:        Y-axis label.
        title:          Plot title.
        figsize:        Figure size.
        marker_size:    Scatter marker area (``s`` parameter).
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("`X` must have shape (n, 2).")

    noise = (
        np.asarray(noise_mask, dtype=bool)
        if noise_mask is not None
        else np.zeros(len(X), dtype=bool)
    )
    highlight = (
        np.asarray(highlight_mask, dtype=bool)
        if highlight_mask is not None
        else np.zeros(len(X), dtype=bool)
    )

    unique_labels = sorted(int(l) for l in np.unique(labels[~noise]))
    cmap = _get_fill_cmap(max(len(unique_labels), 1))
    color_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=figsize)

    if noise.any():
        ax.scatter(
            X[noise, 0],
            X[noise, 1],
            c="#aaaaaa",
            marker="x",
            s=marker_size * 0.6,
            alpha=0.4,
            linewidths=0.8,
            zorder=1,
        )

    for lbl in unique_labels:
        base = (labels == lbl) & ~noise
        color = color_map[lbl]
        normal = base & ~highlight
        hot = base & highlight
        if normal.any():
            ax.scatter(
                X[normal, 0],
                X[normal, 1],
                c=[color],
                s=marker_size,
                alpha=0.8,
                edgecolors="#333333",
                linewidths=0.4,
                zorder=2,
            )
        if hot.any():
            ax.scatter(
                X[hot, 0],
                X[hot, 1],
                c=[color],
                s=marker_size,
                alpha=0.9,
                edgecolors="#cc3333",
                linewidths=1.5,
                zorder=3,
            )

    if names is not None:
        _name = lambda v: names.get(int(v), str(v))
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=9,
                markerfacecolor=color_map[lbl],
                markeredgecolor="#333333",
                markeredgewidth=0.8,
                label=_name(lbl),
            )
            for lbl in unique_labels
        ]
        if highlight_mask is not None:
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=9,
                    markerfacecolor="#aaaaaa",
                    markeredgecolor="#cc3333",
                    markeredgewidth=1.5,
                    label="misclassified",
                )
            )
        ncol = max(1, len(handles) // 12)
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=LEGEND_FONTSIZE,
            framealpha=LEGEND_FRAMEALPHA,
            ncol=ncol,
        )

    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.tick_params(labelsize=TICK_LABELSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    return _fig_to_plot(fig)


def strip_box_plot(
    categories: np.ndarray,
    values: np.ndarray,
    color_values: np.ndarray | None = None,
    edge_values: np.ndarray | None = None,
    x_label: str = "",
    y_label: str = "",
    c_label: str = "",
    edge_label: str = "",
    edge_value_labels: dict | None = None,
    title: str = "",
    figsize: tuple[float, float] = (12, 7),
    marker_size: float = 36,
    cmap: str = "RdYlGn_r",
) -> Plot:
    """Strip plot with per-point color encoding and optional discrete edge encoding.

    Points are jittered horizontally per category; y-axis shows *values*.
    Fill color maps *color_values* (or *values* if omitted) through *cmap*.
    NaN fill values render as neutral gray.
    If *edge_values* is given, discrete unique values map to high-contrast edge
    colors; NaN edge values receive a muted gray outline.
    A horizontal bar marks the per-category median.
    """
    categories = np.asarray(categories)
    values = np.asarray(values, dtype=float)
    c = (
        np.asarray(color_values, dtype=float)
        if color_values is not None
        else values.copy()
    )

    # --- fill colors: normalize over finite values, NaN → gray ---
    finite = np.isfinite(c)
    c_min = float(c[finite].min()) if finite.any() else 0.0
    c_max = float(c[finite].max()) if finite.any() else 1.0
    norm = mcolors.Normalize(vmin=c_min, vmax=c_max)
    colormap = plt.get_cmap(cmap)
    c_safe = np.where(finite, c, (c_min + c_max) / 2)
    point_colors = colormap(norm(c_safe))
    point_colors[~finite] = [0.75, 0.75, 0.75, 0.85]

    # --- edge colors: discrete mapping, NaN → gray ---
    mapped_edge_colors: list | str = "white"
    unique_edges: list = []
    edge_color_map: dict = {}
    scatter_lw = 0.4
    if edge_values is not None:
        edge_arr = np.asarray(edge_values)
        try:
            nan_edge = ~np.isfinite(edge_arr.astype(float))
        except (ValueError, TypeError):
            nan_edge = np.zeros(len(edge_arr), dtype=bool)
        unique_edges = sorted(
            dict.fromkeys(v for v, m in zip(edge_arr.tolist(), nan_edge) if not m)
        )
        edge_color_map = {
            v: OUTLINE_COLORS[i % len(OUTLINE_COLORS)]
            for i, v in enumerate(unique_edges)
        }
        mapped_edge_colors = [
            "#bbbbbb" if m else edge_color_map[v]
            for v, m in zip(edge_arr.tolist(), nan_edge)
        ]
        scatter_lw = 1.3

    # --- layout ---
    category_order = list(dict.fromkeys(categories))
    cat_to_x = {cat: i for i, cat in enumerate(category_order)}
    rng = np.random.default_rng(seed=42)
    x_positions = np.array(
        [cat_to_x[cat] + rng.uniform(-0.25, 0.25) for cat in categories]
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        x_positions,
        values,
        c=point_colors,
        s=marker_size,
        edgecolors=mapped_edge_colors,
        linewidths=scatter_lw,
        zorder=3,
        alpha=0.85,
    )

    for cat, x in cat_to_x.items():
        median_val = np.median(values[categories == cat])
        ax.plot(
            [x - 0.3, x + 0.3],
            [median_val, median_val],
            color="#444444",
            linewidth=1.5,
            zorder=4,
        )

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03).set_label(c_label, fontsize=10)

    if unique_edges:
        _fmt = lambda v: (
            edge_value_labels[v]
            if edge_value_labels and v in edge_value_labels
            else (str(int(v)) if isinstance(v, float) and v == int(v) else str(v))
        )
        edge_handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor="#dddddd",
                markeredgecolor=edge_color_map[v],
                markeredgewidth=1.5,
                label=_fmt(v),
            )
            for v in unique_edges
        ]
        ax.add_artist(
            ax.legend(
                handles=edge_handles,
                title=edge_label or "edge",
                loc="upper left",
                fontsize=8,
                title_fontsize=9,
                framealpha=LEGEND_FRAMEALPHA,
            )
        )

    ax.set_xticks(range(len(category_order)))
    ax.set_xticklabels(category_order)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha="center")
    fig.subplots_adjust(bottom=0.25)
    fig.tight_layout()
    return _fig_to_plot(fig)


def violin_box_plot(
    categories: np.ndarray,
    values: np.ndarray,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    figsize: tuple[float, float] = (6, 5),
    violin_alpha: float = 0.4,
    palette: str = "Set2",
    category_order: list | None = None,
) -> Plot:
    """Split violin plot: each category occupies one half, with a color legend.

    The first category in order of appearance is drawn on the left half,
    the second on the right. Inner quartile lines mark Q1, median, and Q3.
    """
    categories = np.asarray(categories)
    values = np.asarray(values)

    unique_cats = (
        category_order
        if category_order is not None
        else list(dict.fromkeys(categories))
    )
    palette_colors = sns.color_palette(palette, n_colors=len(unique_cats))

    fig, ax = plt.subplots(figsize=figsize)

    legend_handles = []
    for side, (cat, color) in enumerate(zip(unique_cats, palette_colors)):
        mask = categories == cat
        vals = values[mask]
        if len(vals) < 2:
            continue

        parts = ax.violinplot(
            vals, positions=[0], showmedians=False, showextrema=False, widths=0.8
        )
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_alpha(violin_alpha)
        body.set_edgecolor(color)
        body.set_linewidth(1.0)

        # clip to left (side=0) or right (side=1) half
        verts = body.get_paths()[0].vertices
        if side == 0:
            verts[:, 0] = np.minimum(verts[:, 0], 0.0)
        else:
            verts[:, 0] = np.maximum(verts[:, 0], 0.0)

        # inner quartile indicator
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        sign = -1 if side == 0 else 1
        ax.plot([0, sign * 0.06], [med, med], color=color, lw=2.0, zorder=4)
        ax.vlines(sign * 0.04, q1, q3, color=color, lw=1.5, zorder=4)
        legend_handles.append(
            plt.Line2D([], [], color=color, lw=6, alpha=violin_alpha, label=cat)
        )

    ax.axvline(0, color="#aaaaaa", lw=0.8, zorder=2)
    ax.legend(
        handles=legend_handles,
        loc="best",
        fontsize=LEGEND_FONTSIZE,
        framealpha=LEGEND_FRAMEALPHA,
    )

    ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()
    return _fig_to_plot(fig)


def roc_curve_plot(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = "",
    figsize: tuple[float, float] = (6, 6),
) -> Plot:
    """ROC curve with AUC annotation and random-classifier baseline."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color="#4E79A7", linewidth=2, label=f"AUC = {auc_score:.4f}")
    ax.plot(
        [0, 1], [0, 1], color="#AAAAAA", linewidth=1, linestyle="--", label="Random"
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel("True Positive Rate", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=GRID_ALPHA, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return _fig_to_plot(fig)


def feature_importance_plot(
    importances: dict[str, float],
    title: str = "",
    figsize: tuple[float, float] = (12, 8),
) -> Plot:
    """Horizontal bar chart of feature importances, sorted descending."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    features = [item[0] for item in sorted_items]
    values = np.array([item[1] for item in sorted_items])

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(features, values, color="#4E79A7", edgecolor="white", linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    ax.set_xlabel("Importance", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)
    ax.xaxis.grid(True, alpha=GRID_ALPHA, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout(pad=1.0)
    return _fig_to_plot(fig)


def class_pair_scatter(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    class_labels: np.ndarray,
    noise_mask: np.ndarray,
    failure_rate_by_cluster: dict[int, float],
    class_names: dict,
    title: str = "",
    figsize: tuple[float, float] = (12, 10),
) -> Plot:
    """Each cluster drawn as a circle: fill = class color, edge = failure rate (Reds gradient).

    Radius is the 95th-percentile distance of cluster samples to the 2D centroid.
    Noise samples are drawn as small grey dots behind.
    """
    from matplotlib.patches import Circle

    X_2d = np.asarray(X_2d)
    cluster_labels = np.asarray(cluster_labels)
    class_labels = np.asarray(class_labels)
    noise_mask = np.asarray(noise_mask, dtype=bool)
    if X_2d.ndim != 2 or X_2d.shape[1] != 2:
        raise ValueError("`X_2d` must have shape (n, 2).")

    unique_classes = sorted(int(c) for c in np.unique(class_labels))
    class_cmap = _get_fill_cmap(len(unique_classes))
    class_color = {cls: class_cmap(i) for i, cls in enumerate(unique_classes)}

    valid_ids = sorted(int(c) for c in np.unique(cluster_labels[~noise_mask]))
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    edge_cmap = plt.get_cmap("Reds")

    fig, ax = plt.subplots(figsize=figsize)

    if noise_mask.any():
        ax.scatter(
            X_2d[noise_mask, 0],
            X_2d[noise_mask, 1],
            c="#aaaaaa",
            marker="x",
            s=18,
            alpha=0.4,
            linewidths=0.8,
            zorder=1,
        )

    for cid in valid_ids:
        mask = (cluster_labels == cid) & ~noise_mask
        if not np.any(mask):
            continue
        pts = X_2d[mask]
        center = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        radius = float(np.percentile(dists, 95)) if len(dists) > 1 else 0.5

        cls = int(class_labels[mask][0])  # clusters are intra-class by construction
        fr = float(failure_rate_by_cluster.get(cid, 0.0) or 0.0)
        ax.add_patch(
            Circle(
                center,
                radius,
                facecolor=class_color[cls],
                edgecolor=edge_cmap(norm(fr)),
                linewidth=1.0 + 2.0 * fr,
                alpha=0.6,
                zorder=2,
            )
        )

    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")

    class_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=11,
            markerfacecolor=class_color[cls],
            markeredgecolor="#333333",
            markeredgewidth=1.0,
            label=class_names.get(cls, str(cls)),
        )
        for cls in unique_classes
    ]
    leg = ax.legend(
        handles=class_handles,
        loc="upper left",
        title="Class",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
        framealpha=LEGEND_FRAMEALPHA,
    )
    ax.add_artist(leg)

    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03).set_label(
        "Failure rate", fontsize=LABEL_FONTSIZE
    )

    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
    ax.set_xlabel("D1", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel("D2", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.tick_params(labelsize=TICK_LABELSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    return _fig_to_plot(fig)
