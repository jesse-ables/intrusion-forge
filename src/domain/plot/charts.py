import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from .base import (
    Plot,
    _apply_labels,
    _ensure_ax,
    _fig_to_plot,
    _finalize,
    _smart_legend_loc,
)
from .style import (
    HIGHLIGHT_COLOR,
    MUTED_COLOR,
    NEUTRAL_COLOR,
    PALETTE,
    extended_palette,
)


def bar_plot(
    labels: np.ndarray | list[str],
    values: np.ndarray | list[float],
    *,
    orientation: str = "h",
    color: str | list[str] | None = None,
    sort: str | None = "asc",
    top_k: int | None = None,
    annotate_values: bool = True,
    value_format: str = "{:.3f}",
    color_gradient: bool = False,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    bar_positions: np.ndarray | None = None,
    bar_alpha: float = 1.0,
    axvline: float | None = None,
    axvline_color: str = "black",
    hide_yticks: bool = False,
    hide_left_spine: bool = False,
) -> Plot | None:
    """Bar chart with optional sorting, top-k filtering, and value annotations.

    `color` may be a single color (applied to all bars), a list of per-bar
    colors, or None (defaults to PALETTE[0]). `color_gradient=True` is
    ignored when `color` is a list.
    `bar_positions`: explicit numeric positions for `barh` — use when the
    y-axis is shared with another panel (e.g. `sharey=True`).
    """
    if orientation not in ("h", "v"):
        raise ValueError("`orientation` must be 'h' or 'v'.")
    if sort not in ("asc", "desc", None):
        raise ValueError("`sort` must be 'asc', 'desc', or None.")

    items = list(zip(list(labels), list(values)))
    if sort == "asc":
        items.sort(key=lambda kv: kv[1])
        items = items[-top_k:] if top_k is not None else items
    elif sort == "desc":
        items.sort(key=lambda kv: kv[1], reverse=True)
        items = items[:top_k] if top_k is not None else items
    elif top_k is not None:
        items = items[:top_k]

    plot_labels = [str(name) for name, _ in items]
    plot_values = np.array([v for _, v in items], dtype=float)
    n = len(plot_labels)

    if isinstance(color, list):
        bar_color = list(color)
    else:
        base = color if color is not None else PALETTE[0]
        if color_gradient and n > 0:
            base_rgb = np.array(mcolors.to_rgb(base))
            white = np.ones(3)
            ts = 0.25 + 0.75 * np.arange(n) / max(n - 1, 1)
            bar_color = [mcolors.to_hex(t * base_rgb + (1 - t) * white) for t in ts]
        else:
            bar_color = base

    if figsize is None:
        figsize = (
            (12, max(6, n * 0.4 + 1.5))
            if orientation == "h"
            else (max(8, n * 0.5 + 1.5), 6)
        )

    ax, fig = _ensure_ax(ax, figsize)

    positions = bar_positions if bar_positions is not None else plot_labels
    draw = ax.barh if orientation == "h" else ax.bar
    draw(positions, plot_values, color=bar_color, alpha=bar_alpha, edgecolor="white", linewidth=0.5)

    if orientation == "h":
        ax.set_xlabel(x_label or "Value")
        if y_label:
            ax.set_ylabel(y_label)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
    else:
        if x_label:
            ax.set_xlabel(x_label)
        ax.set_ylabel(y_label or "Value")
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if annotate_values and n:
        value_range = plot_values.max() - plot_values.min() or 1.0
        offset = max(value_range * 0.01, 1e-6)
        for i, v in enumerate(plot_values):
            text = value_format.format(v)
            pos = float(bar_positions[i]) if bar_positions is not None else i
            if orientation == "h":
                ax.text(v + offset, pos, text, va="center", ha="left")
            else:
                ax.text(pos, v + offset, text, ha="center", va="bottom")

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title)
    if axvline is not None:
        ax.axvline(axvline, color=axvline_color, linewidth=1.0, zorder=5)
    if hide_yticks:
        ax.tick_params(axis="y", left=False, labelleft=False)
    if hide_left_spine:
        ax.spines["left"].set_visible(False)

    return _finalize(fig)


def violin_plot(
    categories: np.ndarray,
    values: np.ndarray,
    *,
    category_order: list | None = None,
    split: bool = False,
    inner: str = "box",
    colors: tuple[str, ...] | None = None,
    violin_alpha: float = 0.55,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    show_legend: bool = True,
    figsize: tuple[float, float] = (5, 4),
    ax: Axes | None = None,
) -> Plot | None:
    """Violin plot with optional split layout and inner box/quartile indicators."""
    if inner not in ("box", "quartiles", "none"):
        raise ValueError("`inner` must be 'box', 'quartiles', or 'none'.")

    categories = np.asarray(categories)
    values = np.asarray(values)

    unique_cats = (
        list(category_order)
        if category_order is not None
        else list(dict.fromkeys(categories.tolist()))
    )
    n_cats = len(unique_cats)

    if split and n_cats != 2:
        raise ValueError("split=True requires exactly 2 categories.")

    cat_colors = list(colors) if colors is not None else extended_palette(n_cats)
    if len(cat_colors) < n_cats:
        raise ValueError("`colors` must have at least one entry per category.")

    ax, fig = _ensure_ax(ax, figsize)
    legend_handles = []

    for idx, (cat, color) in enumerate(zip(unique_cats, cat_colors)):
        mask = categories == cat
        vals = values[mask]
        if len(vals) < 2 or np.unique(vals).size < 2:
            continue

        position = 0 if split else idx
        parts = ax.violinplot(
            vals,
            positions=[position],
            showmedians=False,
            showextrema=False,
            widths=0.85,
        )
        body = parts["bodies"][0]
        body.set_facecolor(color)
        body.set_alpha(violin_alpha)
        body.set_edgecolor(color)
        body.set_linewidth(1.0)

        if split and body.get_paths():
            verts = body.get_paths()[0].vertices
            clip = np.minimum if idx == 0 else np.maximum
            verts[:, 0] = clip(verts[:, 0], position)

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        whisker_low = float(np.min(vals))
        whisker_high = float(np.max(vals))

        if inner == "box":
            box_half = 0.05
            ax.add_patch(
                plt.Rectangle(
                    (position - box_half, q1),
                    box_half * 2,
                    q3 - q1,
                    facecolor="white",
                    edgecolor=MUTED_COLOR,
                    linewidth=1.0,
                    zorder=4,
                )
            )
            ax.plot(
                [position - box_half, position + box_half],
                [med, med],
                color=MUTED_COLOR,
                linewidth=1.8,
                zorder=5,
            )
            for y0, y1 in ((whisker_low, q1), (q3, whisker_high)):
                ax.plot(
                    [position, position],
                    [y0, y1],
                    color=MUTED_COLOR,
                    linewidth=0.8,
                    zorder=4,
                )
        elif inner == "quartiles":
            sign = 0 if not split else (-1 if idx == 0 else 1)
            ax.plot(
                [position, position + sign * 0.06],
                [med, med],
                color=color,
                lw=2.0,
                zorder=4,
            )
            ax.vlines(position + sign * 0.04, q1, q3, color=color, lw=1.5, zorder=4)

        legend_handles.append(
            plt.Line2D([], [], color=color, lw=6, alpha=violin_alpha, label=str(cat))
        )

    if split:
        ax.axvline(0, color="#aaaaaa", lw=0.8, zorder=2)
        ax.set_xticks([])
    else:
        ax.set_xticks(range(n_cats))
        ax.set_xticklabels([str(c) for c in unique_cats])

    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, loc="best")

    _apply_labels(ax, x_label, y_label, title)
    return _finalize(fig)


def strip_plot(
    categories: np.ndarray,
    values: np.ndarray,
    fill_categorical_colors: tuple[str, ...],
    *,
    fill_values: np.ndarray | None = None,
    marker_values: np.ndarray | None = None,
    marker_shapes: tuple[str, ...] = ("o", "X"),
    category_order: list | None = None,
    orientation: str = "v",
    show_median: bool = True,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    marker_size: float = 36.0,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Plot | None:
    """Strip plot with categorical fill and optional per-point marker encoding."""
    if orientation not in ("v", "h"):
        raise ValueError("`orientation` must be 'v' or 'h'.")

    categories = np.asarray(categories)
    values = np.asarray(values, dtype=float)

    if category_order is None:
        category_order = list(dict.fromkeys(categories.tolist()))
    n_cats = len(category_order)
    cat_to_pos = {cat: i for i, cat in enumerate(category_order)}

    fill_arr = (
        np.asarray(fill_values, dtype=float) if fill_values is not None else values
    )
    finite = np.isfinite(fill_arr)
    fill_idx = np.clip(
        np.where(finite, fill_arr.astype(int), 0),
        0,
        len(fill_categorical_colors) - 1,
    )
    point_colors = np.array(
        [mcolors.to_rgba(fill_categorical_colors[i]) for i in fill_idx], dtype=float
    )
    point_colors[~finite] = [0.75, 0.75, 0.75, 0.85]

    if marker_values is not None:
        marker_arr = np.asarray(marker_values, dtype=float)
        finite_m = np.isfinite(marker_arr)
        marker_idx = np.clip(
            np.where(finite_m, marker_arr, 0.0).astype(int),
            0,
            len(marker_shapes) - 1,
        )
        per_point_marker = np.array([marker_shapes[i] for i in marker_idx])
    else:
        per_point_marker = None

    rng = np.random.default_rng(seed=42)
    base_positions = np.array([cat_to_pos[c] for c in categories], dtype=float)
    positions = base_positions + rng.uniform(-0.25, 0.25, size=len(categories))

    if figsize is None:
        figsize = (
            (11, max(6.0, 0.35 * n_cats + 2.0))
            if orientation == "h"
            else (max(8.0, 0.6 * n_cats + 2.0), 7.0)
        )

    ax, fig = _ensure_ax(ax, figsize)

    def _xy(pos: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (pos, val) if orientation == "v" else (val, pos)

    scatter_kwargs = dict(
        s=marker_size,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
        alpha=0.85,
    )

    if per_point_marker is None:
        x, y = _xy(positions, values)
        ax.scatter(x, y, c=point_colors, **scatter_kwargs)
    else:
        for shape in np.unique(per_point_marker):
            mask = per_point_marker == shape
            if not mask.any():
                continue
            x, y = _xy(positions[mask], values[mask])
            ax.scatter(x, y, c=point_colors[mask], marker=shape, **scatter_kwargs)

    if show_median:
        for cat in category_order:
            mask = categories == cat
            if not mask.any():
                continue
            pos = cat_to_pos[cat]
            med = float(np.median(values[mask]))
            along = (pos - 0.3, pos + 0.3)
            across = (med, med)
            x, y = (along, across) if orientation == "v" else (across, along)
            ax.plot(x, y, color=MUTED_COLOR, linewidth=1.5, zorder=4)

    cat_labels = [str(c) for c in category_order]
    if orientation == "v":
        ax.set_xticks(range(n_cats))
        ax.set_xticklabels(cat_labels)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
        ax.grid(False, axis="x")
        ax.grid(True, axis="y")
    else:
        ax.set_yticks(range(n_cats))
        ax.set_yticklabels(cat_labels)
        ax.invert_yaxis()
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")

    _apply_labels(ax, x_label, y_label, title)
    return _finalize(fig)


def strip_count_panel_plot(
    categories: np.ndarray,
    values: np.ndarray,
    category_order: list[str],
    counts_by_class: dict[str, int],
    fill_values: np.ndarray,
    fill_categorical_colors: tuple[str, ...],
    x_label: str,
    *,
    marker_values: np.ndarray | None = None,
    marker_shapes: tuple[str, ...] = ("o", "X"),
    failed_counts_by_class: dict[str, int] | None = None,
) -> Plot:
    """Strip plot (left) + horizontal count bar (right) sharing the y-axis.

    When `failed_counts_by_class` is provided, a red overlay bar per class
    shows how many clusters have failure_rate > 0, and a vertical reference
    line is drawn at x=0.
    """
    n_cats = len(category_order)
    height = max(3.0, 0.35 * n_cats + 1.5)
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [3.5, 1.0], "wspace": 0.04},
        figsize=(11, height),
        sharey=True,
    )

    strip_plot(
        categories=categories,
        values=values,
        fill_values=fill_values,
        fill_categorical_colors=fill_categorical_colors,
        marker_values=marker_values,
        marker_shapes=marker_shapes,
        category_order=category_order,
        orientation="h",
        show_median=True,
        x_label=x_label,
        y_label="Class",
        ax=ax_left,
    )

    counts = [int(counts_by_class.get(cat, 0)) for cat in category_order]
    y_positions = np.arange(n_cats)
    max_count = max(counts) if counts else 1

    bar_plot(
        labels=list(category_order),
        values=counts,
        orientation="h",
        sort=None,
        bar_positions=y_positions,
        bar_alpha=0.55,
        color=MUTED_COLOR,
        annotate_values=True,
        value_format="{:.0f}",
        x_label="n clusters",
        hide_yticks=True,
        hide_left_spine=True,
        xlim=(0, max_count * 1.18),
        axvline=0 if failed_counts_by_class is not None else None,
        ax=ax_right,
    )

    if failed_counts_by_class is not None:
        failed_counts = [int(failed_counts_by_class.get(cat, 0)) for cat in category_order]
        if any(failed_counts):
            bar_plot(
                labels=list(category_order),
                values=failed_counts,
                orientation="h",
                sort=None,
                bar_positions=y_positions,
                bar_alpha=0.85,
                color=HIGHLIGHT_COLOR,
                annotate_values=False,
                x_label="n clusters",
                hide_yticks=True,
                hide_left_spine=True,
                ax=ax_right,
            )

    return _fig_to_plot(fig)


def scatter_plot(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    noise_mask: np.ndarray | None = None,
    highlight_mask: np.ndarray | None = None,
    names: dict | None = None,
    palette: str | list[str] = "extended",
    marker_size: float = 14.0,
    marker_alpha: float | None = None,
    minority_fraction: float = 0.05,
    x_label: str = "Dim 1",
    y_label: str = "Dim 2",
    title: str = "",
    show_legend: bool = True,
    legend_max_items: int = 20,
    figsize: tuple[float, float] = (6.5, 5.0),
    ax: Axes | None = None,
) -> Plot | None:
    """Scatter 2D with label-coloring and optional highlight mask."""
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
    n_labels = max(len(unique_labels), 1)
    if isinstance(palette, list):
        color_list = list(palette)
        while len(color_list) < n_labels:
            color_list.extend(palette)
        color_list = color_list[:n_labels]
    else:
        color_list = extended_palette(n_labels)
    color_map = {lbl: color_list[i] for i, lbl in enumerate(unique_labels)}

    ax, fig = _ensure_ax(ax, figsize)
    ax.set_aspect("equal", adjustable="datalim")

    if noise.any():
        ax.scatter(
            X[noise, 0],
            X[noise, 1],
            c=MUTED_COLOR,
            marker=".",
            s=marker_size * 0.45,
            alpha=0.4,
            linewidths=0,
            zorder=1,
        )

    counts = {lbl: int(((labels == lbl) & ~noise).sum()) for lbl in unique_labels}
    total_visible = max(sum(counts.values()), 1)
    draw_order = sorted(unique_labels, key=lambda l: counts[l], reverse=True)

    for lbl in draw_order:
        base = (labels == lbl) & ~noise
        if not base.any():
            continue
        color = color_map[lbl]
        n_class = counts[lbl]
        is_minority = n_class / total_visible < minority_fraction

        alpha_val = (
            float(np.clip(2.0 / np.sqrt(max(n_class, 1)), 0.25, 0.85))
            if marker_alpha is None
            else float(marker_alpha)
        )
        size_val = marker_size * (1.4 if is_minority else 1.0)
        if is_minority:
            alpha_val = min(alpha_val + 0.15, 0.85)

        normal = base & ~highlight
        hot = base & highlight
        if normal.any():
            ax.scatter(
                X[normal, 0],
                X[normal, 1],
                c=[color],
                s=size_val,
                alpha=alpha_val,
                marker="o",
                edgecolors="none",
                zorder=3 if is_minority else 2,
                rasterized=True,
            )
        if hot.any():
            ax.scatter(
                X[hot, 0],
                X[hot, 1],
                c=[color],
                s=size_val,
                alpha=min(alpha_val + 0.25, 0.95),
                marker="o",
                edgecolors=HIGHLIGHT_COLOR,
                linewidths=0.9,
                zorder=10,
                rasterized=True,
            )

    if show_legend and names is not None:
        if len(unique_labels) > legend_max_items:
            ax.text(
                0.99,
                0.02,
                f"{len(unique_labels)} groups",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color=MUTED_COLOR,
            )
        else:
            handles = [
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=8,
                    markerfacecolor=color_map[lbl],
                    markeredgecolor="#444444",
                    markeredgewidth=0.5,
                    label=names.get(int(lbl), str(lbl)),
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
                        markersize=8,
                        markerfacecolor=NEUTRAL_COLOR,
                        markeredgecolor=HIGHLIGHT_COLOR,
                        markeredgewidth=0.9,
                        label="misclassified",
                    )
                )
            ncol = 1 if len(handles) <= 6 else 2 if len(handles) <= 12 else 3
            loc = _smart_legend_loc(ax, X[~noise]) if (~noise).any() else "best"
            ax.legend(handles=handles, loc=loc, ncol=ncol)

    _apply_labels(ax, x_label, y_label, title)
    return _finalize(fig)


def heatmap_plot(
    matrix: np.ndarray,
    labels: list[str],
    *,
    mask: np.ndarray | None = None,
    sidebar_values: np.ndarray | None = None,
    sidebar_label: str = "",
    sidebar_cmap: str = "Reds",
    diagonal_values: np.ndarray | None = None,
    diagonal_label: str = "diagonal",
    diagonal_cmap: str = "Reds",
    matrix_label: str = "value",
    matrix_cmap: str = "viridis_r",
    label_legend: dict[int, str] | None = None,
    legend_title: str = "",
    figsize: tuple[float, float] | None = None,
) -> Plot:
    """Square matrix heatmap with optional sidebar, diagonal overlay, and legend.

    `mask`: bool array same shape as `matrix`; True cells render as NaN.
    `sidebar_values`: one scalar per row, drawn as a horizontal bar next to
    the matrix. None skips the sidebar.
    `diagonal_values`: one scalar per row, overlaid on the diagonal using a
    separate cmap scaled to [0, 1].
    `label_legend`: numeric_id -> name mapping rendered below the matrix.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("`matrix` must be square.")
    if len(labels) != n:
        raise ValueError("`labels` length must match matrix size.")
    if mask is not None and np.asarray(mask).shape != (n, n):
        raise ValueError("`mask` must match matrix shape.")
    if sidebar_values is not None and len(sidebar_values) != n:
        raise ValueError("`sidebar_values` length must match matrix size.")

    has_sidebar = sidebar_values is not None
    has_diag = diagonal_values is not None

    if figsize is None:
        cell = max(0.22, min(0.45, 7.0 / max(n, 1)))
        side = max(4.0, cell * n + 1.0)
        figsize = (side + 1.8, side + 1.3)

    width_ratios = [1.0]
    if has_sidebar:
        width_ratios.append(0.06)
    width_ratios.append(0.04)
    if has_diag:
        width_ratios.append(0.04)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=[1.0, 0.20],
        wspace=0.04,
        hspace=0.10,
    )
    col = 0
    ax_mat = fig.add_subplot(gs[0, col]); col += 1
    ax_side = None
    if has_sidebar:
        ax_side = fig.add_subplot(gs[0, col], sharey=ax_mat); col += 1
    ax_cbar = fig.add_subplot(gs[0, col]); col += 1
    ax_cbar_diag = fig.add_subplot(gs[0, col]) if has_diag else None
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis("off")

    cmap = plt.get_cmap(matrix_cmap).copy()
    cmap.set_bad(color=NEUTRAL_COLOR)
    visible = np.where(np.asarray(mask, dtype=bool), np.nan, matrix) if mask is not None else matrix.copy()
    if has_diag:
        np.fill_diagonal(visible, np.nan)
    im = ax_mat.imshow(visible, cmap=cmap, aspect="auto", interpolation="nearest")

    if has_diag:
        d_cmap = plt.get_cmap(diagonal_cmap).copy()
        d_cmap.set_bad(color="none")
        diag_overlay = np.full((n, n), np.nan)
        np.fill_diagonal(diag_overlay, np.asarray(diagonal_values, dtype=float))
        im_diag = ax_mat.imshow(
            diag_overlay,
            cmap=d_cmap,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        fig.colorbar(im_diag, cax=ax_cbar_diag).set_label(diagonal_label)

    ax_mat.set_xticks(np.arange(n))
    ax_mat.set_yticks(np.arange(n))
    ax_mat.set_xticklabels(labels, rotation=90, fontsize=8)
    ax_mat.set_yticklabels(labels, fontsize=8)
    ax_mat.grid(False)
    fig.colorbar(im, cax=ax_cbar).set_label(matrix_label)

    if has_sidebar:
        sidebar = np.asarray(sidebar_values, dtype=float)
        safe = np.where(np.isfinite(sidebar), sidebar, 0.0)
        s_max = float(safe.max()) if safe.size and safe.max() > 0 else 1.0
        side_cmap = plt.get_cmap(sidebar_cmap)
        side_colors = [
            side_cmap(0.0) if v <= 0 else side_cmap(0.25 + 0.6 * (v / s_max))
            for v in safe
        ]
        ax_side.barh(
            np.arange(n), safe, color=side_colors, edgecolor="white", linewidth=0.3
        )
        ax_side.set_xlim(0, s_max * 1.05 if s_max > 0 else 1.0)
        ax_side.tick_params(axis="y", left=False, labelleft=False)
        ax_side.spines["left"].set_visible(False)
        ax_side.set_xlabel(sidebar_label, fontsize=8)
        ax_side.tick_params(axis="x", labelsize=7)
        ax_side.grid(False)

    if label_legend:
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1,
                facecolor="none",
                edgecolor="none",
                label=f"{num}: {name}",
            )
            for num, name in sorted(label_legend.items())
        ]
        ax_legend.legend(
            handles=handles,
            loc="upper center",
            ncol=min(len(handles), 3),
            frameon=False,
            fontsize=8,
            handlelength=0,
            handletextpad=0,
            columnspacing=1.4,
            title=legend_title or None,
            title_fontsize=8,
        )

    return _fig_to_plot(fig)


def ridgeline_plot(
    distributions: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    order: list[str] | None = None,
    legend_labels: tuple[str, str] = ("a", "b"),
    colors: tuple[str, str] | None = None,
    x_label: str = "value",
    figsize: tuple[float, float] | None = None,
) -> Plot:
    """One horizontal density per key, two overlaid series per row.

    `distributions`: label -> (series_a, series_b). Empty or all-non-finite
    rows are dropped.
    `colors`: fill colors for series a and b; defaults to (PALETTE[0], PALETTE[1]).
    """
    raw_keys = order if order is not None else list(distributions.keys())

    def _finite(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr[np.isfinite(arr)]

    finite_per_key: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k in raw_keys:
        fa, fb = (_finite(s) for s in distributions[k])
        if len(fa) == 0 and len(fb) == 0:
            continue
        finite_per_key[k] = (fa, fb)

    keys = [k for k in raw_keys if k in finite_per_key]
    n = len(keys)
    if figsize is None:
        figsize = (8.0, max(2.5, 0.4 * max(n, 1) + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    if n == 0:
        ax.set_xlabel(x_label)
        return _fig_to_plot(fig)

    concat = np.concatenate(
        [arr for fa, fb in finite_per_key.values() for arr in (fa, fb) if len(arr)]
    )
    x_min, x_max = float(concat.min()), float(concat.max())
    if x_min == x_max:
        x_max = x_min + 1.0
    grid = np.linspace(x_min, x_max, 200)

    spacing = 0.7
    max_kde_height = spacing * 0.9
    rug_height = spacing * 0.25
    color_a, color_b = colors if colors is not None else (PALETTE[0], PALETTE[1])

    def _kde_curve(arr: np.ndarray) -> np.ndarray | None:
        if len(arr) < 2 or arr.std() < 1e-9:
            return None
        try:
            return gaussian_kde(arr, bw_method="scott")(grid)
        except Exception:
            return None

    def _draw_rug(values: np.ndarray, base_y: float, color: str) -> None:
        if not len(values):
            return
        ax.vlines(
            values,
            base_y,
            base_y + rug_height,
            color=color,
            linewidth=1.0,
            alpha=0.85,
            zorder=4,
        )

    for i, k in enumerate(keys):
        base_y = (n - 1 - i) * spacing
        fa, fb = finite_per_key[k]
        curves = [_kde_curve(fa), _kde_curve(fb)]
        row_max = max(
            (float(c.max()) for c in curves if c is not None), default=1e-9
        )
        for curve, raw, color, zorder in zip(
            curves, (fa, fb), (color_a, color_b), (3, 2)
        ):
            if curve is None:
                _draw_rug(raw, base_y, color)
                continue
            heights = (curve / row_max) * max_kde_height
            ax.fill_between(
                grid,
                base_y,
                base_y + heights,
                color=color,
                alpha=0.5,
                linewidth=0.7,
                edgecolor=color,
                zorder=zorder,
            )

    ax.set_yticks([(n - 1 - i) * spacing + max_kde_height * 0.2 for i in range(n)])
    ax.set_yticklabels(keys, fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_xlim(x_min, x_max)
    ax.grid(True, axis="x", alpha=0.15, linewidth=0.5)
    ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_a, alpha=0.5, label=legend_labels[0]),
        plt.Rectangle((0, 0), 1, 1, facecolor=color_b, alpha=0.5, label=legend_labels[1]),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    return _fig_to_plot(fig)


def line_plot(
    series: dict[str, list[float] | np.ndarray],
    *,
    x_label: str = "step",
    y_label: str = "value",
    title: str = "",
    colors: list[str] | None = None,
    linewidth: float = 1.2,
    show_legend: bool = True,
    figsize: tuple[float, float] = (8.0, 4.0),
    ax: Axes | None = None,
) -> Plot | None:
    """Multi-series line plot (e.g. training history curves).

    `series` is a mapping `{name: values}`. Each series is drawn as a line; the
    x axis is the step index for every series. `colors` defaults to the
    extended palette in series-key order.
    """
    ax, fig = _ensure_ax(ax, figsize)
    keys = list(series.keys())
    palette = colors if colors is not None else extended_palette(max(len(keys), 1))

    for i, (name, values) in enumerate(series.items()):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        ax.plot(np.arange(arr.size), arr, color=palette[i], linewidth=linewidth, label=name)

    _apply_labels(ax, x_label=x_label, y_label=y_label, title=title)
    ax.grid(True, axis="both", alpha=0.15, linewidth=0.5)
    if show_legend and len(keys) > 1:
        ax.legend(loc="best", fontsize=8)

    return _finalize(fig)
