import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle, ConnectionPatch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "original"
FIG_DIR = ROOT / "outputs" / "paper" / "figures"

FILES = {
    "Inception": "fid_table_results.csv",
    "CLIP": "fid_table_results_clip-vit-b-16.csv",
    "DINOv2": "fid_table_results_dinov2-vit-b-14.csv",
    "VGG16": "fid_table_results_vgg16.csv",
}

HIGHLIGHT = {
    "BAR-L": "#0072B2",
    "REPA-E": "#D55E00",
    "RAE-DDT": "#009E73",
    "BAR-B": "#CC79A7",
    "xAR-H": "#7E57C2",
    "xAR-L": "#E64B35",
    "xAR-B": "#00A5CF",
}

DISPLAY_NAMES = {
    "VAVAE-800ep": "LightningDiT",
}


def parse_float(value):
    value = value.strip()
    if not value:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def load_table(filename):
    path = DATA_DIR / filename
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    groups = rows[0][1:]
    metrics = rows[1][1:]
    table = {}
    for row in rows[2:]:
        if not row:
            continue
        model = row[0].strip()
        if not model or model == "Val":
            continue
        values = {}
        for i, value in enumerate(row[1:]):
            if i >= len(groups):
                continue
            values[(groups[i].strip(), metrics[i].strip())] = parse_float(value)
        table[model] = values
    return table


def val(table, model, setting, metric):
    return table.get(model, {}).get((setting, metric), math.nan)


def finite(x):
    return x is not None and np.isfinite(x)


def rank_values(values):
    items = [(k, v) for k, v in values.items() if finite(v)]
    items.sort(key=lambda x: x[1])
    return {k: i + 1 for i, (k, _) in enumerate(items)}


def rankdata(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return math.nan, len(x)
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


def set_style():
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#A8B3BD",
            "axes.linewidth": 0.8,
            "grid.color": "#E6EBF0",
            "grid.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def top_models(inception, n=12):
    scores = {
        m: val(inception, m, "CFG", "Train (openai stats)") for m in inception.keys()
    }
    scores = {m: s for m, s in scores.items() if finite(s)}
    return [m for m, _ in sorted(scores.items(), key=lambda x: x[1])[:n]]


def top_entries(table, setting, metric, k=3):
    scores = {
        m: val(table, m, setting, metric)
        for m in table.keys()
    }
    scores = {m: s for m, s in scores.items() if finite(s)}
    return sorted(scores.items(), key=lambda x: x[1])[:k]


def make_train_val_scatter(inception):
    models = top_models(inception, 9)
    xs = [val(inception, m, "CFG", "Train (openai stats)") for m in models]
    ys = [val(inception, m, "CFG", "Val") for m in models]
    colors = [HIGHLIGHT.get(m, "#8F98A3") for m in models]

    fig, ax = plt.subplots(figsize=(7.35, 3.65))
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.scatter(xs, ys, s=92, c=colors, edgecolor="white", linewidth=1.0, zorder=3)

    offsets = {
        "BAR-L": (0.014, 0.004, "left", "center"),
        "REPA-E": (0.014, -0.036, "left", "center"),
        "RAE-DDT": (0.014, 0.022, "left", "center"),
        "BAR-B": (0.014, -0.052, "left", "center"),
        "xAR-H": (0.014, 0.030, "left", "center"),
        "xAR-L": (0.014, -0.050, "left", "center"),
        "BAR-B-2": (0.014, 0.012, "left", "center"),
        "VAVAE-800ep": (0.014, 0.012, "left", "center"),
        "REPA-XL-2": (0.014, 0.012, "left", "center"),
    }
    for model, x, y, color in zip(models, xs, ys, colors):
        dx, dy, ha, va = offsets.get(model, (0.012, 0.010, "left", "center"))
        ax.text(x + dx, y + dy, model, color=color, fontsize=9.0, ha=ha, va=va)

    ax.set_xlabel("Train-reference FID", fontsize=12.5, fontweight="bold", labelpad=7)
    ax.set_ylabel("Validation-reference FD", fontsize=12.5, fontweight="bold", labelpad=7)
    ax.set_xlim(0.95, 1.61)
    ax.set_ylim(2.15, 2.96)
    ax.set_xticks(np.arange(1.0, 1.61, 0.1))
    ax.set_yticks(np.arange(2.2, 3.0, 0.1))
    ax.tick_params(axis="both", labelsize=11.5, colors="#4B5563")
    fig.tight_layout(pad=0.5)
    fig.savefig(FIG_DIR / "train_vs_val_scatter.png", bbox_inches="tight")
    plt.close(fig)


def make_protocol_winner_board(inception):
    protocols = [
        ("Train", "Train (openai stats)"),
        ("Recomputed", "Train (JiT stats)"),
        ("Balanced", "Train (balanced)"),
        ("Balanced732", "Train (balanced732)"),
        ("Validation", "Val"),
    ]
    settings = [("CFG", "CFG"), ("w/o CFG", "w/o CFG")]

    fig, ax = plt.subplots(figsize=(7.35, 2.75))
    ax.set_xlim(-0.72, len(protocols))
    ax.set_ylim(-0.08, 2.28)
    ax.axis("off")

    for col, (label, _) in enumerate(protocols):
        ax.text(col + 0.46, 2.10, label, ha="center", va="bottom", fontsize=8.0, color="#334155")

    for row, (row_label, setting) in enumerate(settings):
        y0 = 1.12 - row * 1.12
        ax.text(-0.42, y0 + 0.46, row_label, ha="right", va="center", fontsize=8.6, fontweight="bold", color="#334155")
        for col, (_, metric) in enumerate(protocols):
            entries = top_entries(inception, setting, metric, 3)
            winner, winner_score = entries[0]
            is_validation = metric == "Val"
            face = "#ECF7F4" if is_validation else "#F8FBFD"
            edge = "#BFDAD4" if is_validation else "#D9E2EA"
            rect = FancyBboxPatch(
                (col + 0.02, y0),
                0.90,
                0.92,
                boxstyle="round,pad=0.018,rounding_size=0.035",
                linewidth=0.75,
                edgecolor=edge,
                facecolor=face,
            )
            ax.add_patch(rect)
            color = HIGHLIGHT.get(winner, "#718096")
            ax.plot([col + 0.07, col + 0.07], [y0 + 0.14, y0 + 0.78], color=color, lw=2.4, solid_capstyle="round")
            ax.text(col + 0.13, y0 + 0.63, winner, ha="left", va="center", fontsize=10.0, fontweight="bold", color=color)
            ax.text(col + 0.13, y0 + 0.36, f"{winner_score:.2f}", ha="left", va="center", fontsize=8.8, color="#475569")
            if setting == "w/o CFG" and metric == "Val":
                runner_up, runner_score = entries[1]
                ax.text(
                    col + 0.13,
                    y0 + 0.17,
                    f"near tie: {runner_up} {runner_score:.2f}",
                    ha="left",
                    va="center",
                    fontsize=6.4,
                    color=HIGHLIGHT.get(runner_up, "#66727D"),
                )

    fig.tight_layout(pad=0.1)
    fig.savefig(FIG_DIR / "protocol_winner_board.png", bbox_inches="tight")
    plt.close(fig)


def make_fid_leaderboard_progress(inception):
    settings = [("CFG", "CFG"), ("w/o CFG", "w/o CFG")]
    figure_highlight = {k: v for k, v in HIGHLIGHT.items() if k != "xAR-B"}
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.0), sharey=True)
    for ax, (title, setting) in zip(axes, settings):
        rows = [
            (m, val(inception, m, setting, "Train (openai stats)"))
            for m in inception
            if finite(val(inception, m, setting, "Train (openai stats)"))
            and val(inception, m, setting, "Train (openai stats)") <= 8
        ]
        rows.sort(key=lambda x: x[1], reverse=True)
        rows = rows[-24:]
        rows.sort(key=lambda x: x[1], reverse=True)
        x = np.arange(1, len(rows) + 1)
        y = np.array([score for _, score in rows])
        ax.axhspan(1.0, 2.0, color="#E8F5F2", zorder=0)
        ax.plot(x, y, color="#94A3B8", lw=1.2, zorder=1)
        for i, (model, score) in enumerate(rows, start=1):
            color = figure_highlight.get(model, "#B8C2CC")
            size = 42 if model in figure_highlight else 20
            ax.scatter(i, score, s=size, color=color, edgecolor="white", lw=0.6, zorder=3)
            if i > len(rows) - 8 or model in ["xAR-H", "xAR-L", "RAE-DDT"]:
                dx, dy, ha = (0.0, 0.08, "center")
                if setting == "w/o CFG" and model == "VAVAE-800ep":
                    ax.annotate(
                        DISPLAY_NAMES.get(model, model),
                        xy=(i, score),
                        xytext=(-3, 3),
                        textcoords="offset points",
                        ha="right",
                        va="bottom",
                        rotation=55,
                        fontsize=5.7,
                        color=color,
                    )
                    continue
                ax.text(i + dx, score + dy, DISPLAY_NAMES.get(model, model), ha=ha, va="bottom", rotation=55, fontsize=5.7, color=color)
        ax.text(0.05, 0.17, "audit frontier\nFID 1-2", transform=ax.transAxes, ha="left", va="center", fontsize=7.0, color="#2F6F66")
        ax.annotate(
            "lower FID",
            xy=(0.94, 0.88),
            xytext=(0.73, 0.88),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=7.0,
            color="#2F6F66",
            arrowprops=dict(arrowstyle="->", color="#2F6F66", lw=1.15, shrinkA=2, shrinkB=2),
        )
        ax.set_title(title, fontsize=8.5, color="#334155")
        ax.set_xlabel("Models ordered from higher to lower FID")
        ax.set_ylabel("Conventional train-reference FID")
        ax.set_xlim(0.4, len(rows) + 0.6)
        ax.set_ylim(0.75, min(8.3, max(4.2, float(np.nanmax(y)) + 0.35)))
        ax.grid(axis="y")
        ax.tick_params(labelsize=6.4)
    fig.tight_layout(pad=0.55)
    fig.savefig(FIG_DIR / "fid_leaderboard_progress.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fid_leaderboard_progress.pdf", bbox_inches="tight")
    plt.close(fig)


def make_rank_shift(inception):
    models = top_models(inception, 12)
    protocols = [
        ("Train", "Train (openai stats)"),
        ("Balanced", "Train (balanced)"),
        ("Validation", "Val"),
    ]
    ranks = {}
    scores = {}
    for label, metric in protocols:
        subset_scores = {m: val(inception, m, "CFG", metric) for m in models}
        ranks[label] = rank_values(subset_scores)
        scores[label] = subset_scores

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    xs = np.arange(len(protocols))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.set_ylim(12.7, 0.3)
    ax.set_yticks(range(1, 13))
    ax.set_ylabel("Rank within conventional top-12")
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in protocols])
    ax.set_xlim(-0.28, 2.72)

    for model in models:
        y = [ranks[label][model] for label, _ in protocols]
        change = y[-1] - y[0]
        if change < 0:
            color = "#2A9D8F"
        elif change > 0:
            color = "#E76F51"
        else:
            color = "#7A8793"
        ax.plot(xs, y, color=color, lw=1.9, alpha=0.86, zorder=2)
        ax.scatter(xs, y, s=34, color="white", edgecolor=color, lw=1.3, zorder=3)
        ax.text(2.08, y[-1], model, va="center", ha="left", fontsize=7.5, color="#263238")
        ax.text(-0.08, y[0], f"{scores['Train'][model]:.2f}", va="center", ha="right", fontsize=6.8, color="#66727D")
        ax.text(2.02, y[-1] + 0.28, f"{scores['Validation'][model]:.2f}", va="center", ha="left", fontsize=6.4, color="#66727D")

    ax.text(-0.08, 0.75, "FID", ha="right", va="center", fontsize=7, color="#66727D")
    ax.text(2.02, 0.75, "FD", ha="left", va="center", fontsize=7, color="#66727D")

    handles = [
        plt.Line2D([0], [0], color="#2A9D8F", lw=2, label="improves under validation"),
        plt.Line2D([0], [0], color="#E76F51", lw=2, label="drops under validation"),
        plt.Line2D([0], [0], color="#7A8793", lw=2, label="stable"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG_DIR / "frontier_protocol_rank_shift.png", bbox_inches="tight")
    plt.close(fig)


def validation_values(tables):
    values = {}
    all_models = set.intersection(*(set(t.keys()) for t in tables.values()))
    for model in all_models:
        row = {
            name: val(table, model, "CFG", "Val")
            for name, table in tables.items()
        }
        if all(finite(v) for v in row.values()):
            values[model] = row
    return values


def make_pairwise_scatter(tables):
    values = validation_values(tables)
    shown = sorted(values, key=lambda m: values[m]["Inception"])[:28]
    pairs = ["CLIP", "DINOv2", "VGG16"]
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.95))

    all_stats = {}
    shown_stats = {}
    for ax, name in zip(axes, pairs):
        all_x = [values[m]["Inception"] for m in values]
        all_y = [values[m][name] for m in values]
        rho_all, n_all = spearman(all_x, all_y)
        x = np.array([values[m]["Inception"] for m in shown])
        y = np.array([values[m][name] for m in shown])
        rho_shown, n_shown = spearman(x, y)
        all_stats[name] = (rho_all, n_all)
        shown_stats[name] = (rho_shown, n_shown)

        colors = [HIGHLIGHT.get(m, "#B8C2CC") for m in shown]
        sizes = [58 if m in HIGHLIGHT else 34 for m in shown]
        ax.scatter(x, y, s=sizes, c=colors, edgecolor="white", linewidth=0.7, alpha=0.95, zorder=3)
        ax.grid(True, zorder=0)
        ax.set_xlabel("Inception-FD")
        ax.set_ylabel(f"{name}-FD")
        ax.text(0.05, 0.94, f"$r_s$={rho_shown:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#334155")

        xmin, xmax = np.nanmin(x), np.nanmax(x)
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        ax.set_xlim(xmin - 0.08 * (xmax - xmin), xmax + 0.10 * (xmax - xmin))
        ax.set_ylim(ymin - 0.08 * (ymax - ymin), ymax + 0.12 * (ymax - ymin))

    legend_order = ["BAR-L", "REPA-E", "RAE-DDT", "xAR-H", "xAR-L", "BAR-B"]
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=HIGHLIGHT[m],
            markeredgecolor="white",
            markersize=6,
            label=m,
        )
        for m in legend_order
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#B8C2CC",
            markeredgecolor="white",
            markersize=6,
            label="other competitive",
        )
    )
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False)
    fig.tight_layout(pad=0.55, rect=(0, 0.10, 1, 1))
    fig.savefig(FIG_DIR / "pairwise_metric_scatter.png", bbox_inches="tight")
    plt.close(fig)
    return shown_stats, all_stats, len(shown), len(values)


def make_spearman_matrix(tables):
    values = validation_values(tables)
    names = ["Inception", "CLIP", "DINOv2", "VGG16"]
    matrix = np.ones((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            rho, _ = spearman([values[m][a] for m in values], [values[m][b] for m in values])
            matrix[i, j] = rho

    cmap = LinearSegmentedColormap.from_list(
        "soft_teal",
        ["#F4F7FA", "#D8F0EA", "#7BC8B6", "#1F8A83"],
    )
    fig, ax = plt.subplots(figsize=(3.65, 3.1))
    im = ax.imshow(matrix, vmin=0.65, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8.4, color="#102A43")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, length=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG_DIR / "backbone_spearman_matrix.png", bbox_inches="tight")
    plt.close(fig)
    return matrix, len(values)


def make_backbone_rank_scatter(tables):
    values = validation_values(tables)
    names = ["CLIP", "DINOv2", "VGG16"]
    ranks = {}
    for name in ["Inception"] + names:
        ranks[name] = rank_values({m: values[m][name] for m in values})

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.9))
    for ax, name in zip(axes, names):
        models = sorted(values, key=lambda m: ranks["Inception"][m])
        x = np.array([ranks["Inception"][m] for m in models])
        y = np.array([ranks[name][m] for m in models])
        rho, n = spearman(x, y)
        colors = [HIGHLIGHT.get(m, "#B8C2CC") for m in models]
        sizes = [56 if m in HIGHLIGHT else 30 for m in models]
        ax.scatter(x, y, s=sizes, c=colors, edgecolor="white", linewidth=0.75, zorder=3)
        lim = (0.5, len(models) + 0.5)
        ax.plot(lim, lim, color="#94A3B8", lw=1.1, ls="--", zorder=1)
        ax.set_xlim(*lim)
        ax.set_ylim(len(models) + 0.5, 0.5)
        ax.set_xlabel("Inception rank")
        ax.set_ylabel(f"{name} rank")
        ax.grid(True, zorder=0)
        ax.text(0.06, 0.94, f"$r_s$={rho:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#334155")

    legend_order = ["BAR-L", "REPA-E", "RAE-DDT", "xAR-H", "xAR-L", "BAR-B"]
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=HIGHLIGHT[m],
            markeredgecolor="white",
            markersize=6,
            label=m,
        )
        for m in legend_order
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#B8C2CC",
            markeredgecolor="white",
            markersize=6,
            label="other models",
        )
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False)
    fig.tight_layout(pad=0.55, rect=(0, 0.10, 1, 1))
    fig.savefig(FIG_DIR / "backbone_rank_scatter.png", bbox_inches="tight")
    plt.close(fig)
    return len(values)


def make_backbone_scatter_panel(tables):
    values = validation_values(tables)
    shown = sorted(values, key=lambda m: values[m]["Inception"])[:28]
    names = ["CLIP", "DINOv2", "VGG16"]

    ranks = {}
    for name in ["Inception"] + names:
        ranks[name] = rank_values({m: values[m][name] for m in values})

    fig, axes = plt.subplots(4, 3, figsize=(7.35, 8.45))
    row_labels = [
        "Validation FD",
        "Validation rank",
        "Conventional FID vs. judge",
        "Reference split",
    ]

    def scatter(ax, models, x, y, name, xlabel, ylabel, diagonal=False):
        colors = [HIGHLIGHT.get(m, "#B8C2CC") for m in models]
        sizes = [42 if m in HIGHLIGHT else 20 for m in models]
        ax.scatter(x, y, s=sizes, c=colors, edgecolor="white", linewidth=0.55, alpha=0.95, zorder=3)
        rho, n = spearman(x, y)
        rank_panel = "rank" in ylabel
        ax.text(0.95 if rank_panel else 0.05, 0.92, f"$r_s$={rho:.2f}", transform=ax.transAxes, ha="right" if rank_panel else "left", va="top", fontsize=6.5, color="#334155")
        ax.grid(True, zorder=0)
        ax.set_xlabel(xlabel, fontsize=6.6)
        ax.set_ylabel(ylabel, fontsize=6.6)
        ax.tick_params(labelsize=6.1, length=2.5)
        if diagonal:
            xmin, xmax = np.nanmin(x), np.nanmax(x)
            ymin, ymax = np.nanmin(y), np.nanmax(y)
            lo, hi = min(xmin, ymin), max(xmax, ymax)
            ax.plot([lo, hi], [lo, hi], color="#94A3B8", lw=0.9, ls="--", zorder=1)
            pad = 0.06 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        else:
            xmin, xmax = np.nanmin(x), np.nanmax(x)
            ymin, ymax = np.nanmin(y), np.nanmax(y)
            xpad = 0.07 * (xmax - xmin) if xmax > xmin else 1.0
            ypad = 0.08 * (ymax - ymin) if ymax > ymin else 1.0
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)

    for col, name in enumerate(names):
        axes[0, col].set_title(name, fontsize=9.2, pad=17, fontweight="bold", color="#24465C")

        models = shown
        x = np.array([values[m]["Inception"] for m in models])
        y = np.array([values[m][name] for m in models])
        scatter(axes[0, col], models, x, y, name, "Inception-FD", f"{name}-FD")

        models = sorted(values, key=lambda m: ranks["Inception"][m])
        x = np.array([ranks["Inception"][m] for m in models])
        y = np.array([ranks[name][m] for m in models])
        scatter(axes[1, col], models, x, y, name, "Inception rank", f"{name} rank", diagonal=True)
        axes[1, col].set_ylim(len(models) + 0.5, 0.5)

        models = [m for m in shown if finite(val(tables["Inception"], m, "CFG", "Train (openai stats)"))]
        x = np.array([val(tables["Inception"], m, "CFG", "Train (openai stats)") for m in models])
        y = np.array([values[m][name] for m in models])
        scatter(axes[2, col], models, x, y, name, "Train FID", f"{name}-FD")

        models = [
            m
            for m in shown
            if finite(val(tables[name], m, "CFG", "Train (balanced)"))
            and finite(val(tables[name], m, "CFG", "Val"))
        ]
        x = np.array([val(tables[name], m, "CFG", "Train (balanced)") for m in models])
        y = np.array([val(tables[name], m, "CFG", "Val") for m in models])
        scatter(axes[3, col], models, x, y, name, "Balanced ref", "Validation ref", diagonal=True)

    for row, label in enumerate(row_labels):
        axes[row, 0].text(
            0.01,
            1.045,
            label,
            transform=axes[row, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=7.9,
            fontweight="bold",
            color="#2F6F66",
        )

    legend_order = ["BAR-L", "REPA-E", "RAE-DDT", "xAR-H", "xAR-L", "BAR-B"]
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=HIGHLIGHT[m],
            markeredgecolor="white",
            markersize=5.2,
            label=m,
        )
        for m in legend_order
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#B8C2CC",
            markeredgecolor="white",
            markersize=5.2,
            label="other models",
        )
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.006), ncol=7, frameon=False, fontsize=7.0)
    fig.tight_layout(pad=0.45, h_pad=0.84, w_pad=0.55, rect=(0.0, 0.035, 1, 1))
    fig.savefig(FIG_DIR / "backbone_scatter_panel.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "backbone_scatter_panel.pdf", bbox_inches="tight")
    plt.close(fig)
    return len(shown), len(values)


def make_protocol_ranges(inception):
    models = top_models(inception, 12)
    protocols = [
        ("Train", "Train (openai stats)"),
        ("Recomputed", "Train (JiT stats)"),
        ("Balanced", "Train (balanced)"),
        ("Balanced732", "Train (balanced732)"),
        ("Validation", "Val"),
    ]
    rows = []
    best = min(val(inception, m, "CFG", "Train (openai stats)") for m in models)
    for m in models:
        vals = [val(inception, m, "CFG", metric) for _, metric in protocols]
        vals = np.array([v for v in vals if finite(v)])
        train = val(inception, m, "CFG", "Train (openai stats)")
        validation = val(inception, m, "CFG", "Val")
        rows.append(
            {
                "model": m,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "range": float(np.max(vals) - np.min(vals)),
                "train": train,
                "val": validation,
                "margin": train - best,
            }
        )

    all_rows = []
    for m in inception:
        vals = [val(inception, m, "CFG", metric) for _, metric in protocols]
        if not all(finite(v) for v in vals):
            continue
        train = val(inception, m, "CFG", "Train (openai stats)")
        if not finite(train) or train > 4.5:
            continue
        all_rows.append(
            {
                "model": m,
                "train": train,
                "range": float(np.max(vals) - np.min(vals)),
            }
        )

    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 4.45), gridspec_kw={"width_ratios": [1.4, 1.0]})
    ax = axes[0]
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    for i, row in enumerate(rows):
        ax.hlines(i, row["min"], row["max"], color="#90A4AE", lw=3.2, alpha=0.85)
        ax.scatter(row["train"], i, s=32, color="#1F77B4", edgecolor="white", lw=0.7, zorder=3)
        ax.scatter(row["val"], i, s=32, color="#E76F51", edgecolor="white", lw=0.7, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_NAMES.get(r["model"], r["model"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Inception FD across reference protocols")

    ax2 = axes[1]
    ax2.set_axisbelow(True)
    ax2.grid(True)
    ax2.axvspan(1.0, 2.0, color="#E8F5F2", zorder=0)
    ax2.text(1.05, 0.96, "audit frontier\nFID 1-2", transform=ax2.get_xaxis_transform(), ha="left", va="top", fontsize=7.2, color="#2F6F66")
    for row in all_rows:
        color = HIGHLIGHT.get(row["model"], "#B8C2CC")
        size = 46 if row["model"] in HIGHLIGHT else 24
        ax2.scatter(row["train"], row["range"], s=size, color=color, edgecolor="white", lw=0.7, alpha=0.95, zorder=3)
    ax2.set_xlabel("Conventional train FID")
    ax2.set_ylabel("Reference-protocol range\n(max - min FD)")
    ax2.set_xlim(0.85, 4.55)
    ymin = min(r["range"] for r in all_rows) * 0.90
    ymax = max(r["range"] for r in all_rows) * 1.12
    ax2.set_ylim(ymin, ymax)
    for row in all_rows:
        if row["model"] in ("BAR-L", "REPA-E", "RAE-DDT"):
            offset = {"BAR-L": (7, 4), "REPA-E": (7, -9), "RAE-DDT": (7, -2)}[row["model"]]
            ax2.annotate(row["model"], (row["train"], row["range"]),
                         xytext=offset, textcoords="offset points", fontsize=6.4,
                         color=HIGHLIGHT[row["model"]])

    handles = [
        plt.Line2D([0, 1], [0, 0], color="#90A4AE", lw=3.2, label="min–max across 5 references"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#1F77B4", markeredgecolor="white", markersize=5, label="conventional train FID"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#E76F51", markeredgecolor="white", markersize=5, label="validation FD"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=3, frameon=False, fontsize=6.7)
    fig.tight_layout(pad=0.55, rect=(0, 0.03, 1, 1))
    fig.savefig(FIG_DIR / "protocol_sensitivity_ranges.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "protocol_sensitivity_ranges.pdf", bbox_inches="tight")
    plt.close(fig)


def make_margin_vs_shift(inception):
    models = top_models(inception, 12)
    best = min(val(inception, m, "CFG", "Train (openai stats)") for m in models)
    rows = []
    for model in models:
        train = val(inception, model, "CFG", "Train (openai stats)")
        validation = val(inception, model, "CFG", "Val")
        rows.append(
            {
                "model": model,
                "margin": train - best,
                "lift": validation - train,
            }
        )

    fig, ax = plt.subplots(figsize=(7.35, 2.85))
    y = np.arange(len(rows))
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ax.axvspan(0, 0.10, color="#FFF4D6", zorder=0)
    ax.text(0.05, -0.68, "0.1 FID", ha="center", va="center", fontsize=7.0, color="#8A6D1D")

    for i, row in enumerate(rows):
        ax.hlines(i, row["margin"], row["lift"], color="#CAD5E0", lw=2.8, zorder=1)
        ax.scatter(row["margin"], i, s=34, color="#1F77B4", edgecolor="white", lw=0.7, zorder=3)
        ax.scatter(row["lift"], i, s=34, color="#E76F51", edgecolor="white", lw=0.7, zorder=3)

    ax.text(0.13, 1.04, "conventional leaderboard margin", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.4, color="#1F77B4")
    ax.text(0.98, 1.04, "validation-reference lift", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4, color="#E76F51")
    ax.set_yticks(y)
    ax.set_yticklabels([r["model"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Difference from conventional train-reference FID")
    ax.set_xlim(0, max(r["lift"] for r in rows) * 1.10)
    fig.tight_layout(pad=0.45)
    fig.savefig(FIG_DIR / "margin_vs_shift.png", bbox_inches="tight")
    plt.close(fig)


def protocol_metrics():
    return [
        ("Train", "Train (openai stats)"),
        ("Recomputed", "Train (JiT stats)"),
        ("Balanced", "Train (balanced)"),
        ("Balanced732", "Train (balanced732)"),
        ("Validation", "Val"),
    ]


def make_protocol_scatter_panel(inception):
    alternatives = [
        ("Recomputed", "Train (JiT stats)"),
        ("Balanced-50", "Train (balanced)"),
        ("Balanced-732", "Train (balanced732)"),
        ("Validation", "Val"),
    ]
    columns = [
        ("CFG (FD <= 8)", "CFG", "all"),
        ("CFG: top-9 zoom", "CFG", "frontier"),
        ("w/o CFG (FD <= 8)", "w/o CFG", "all"),
    ]
    frontier_models = top_models(inception, 9)
    fig, axes = plt.subplots(4, 3, figsize=(7.35, 7.55))
    for row, (row_label, metric) in enumerate(alternatives):
        for col, (col_label, setting, scope) in enumerate(columns):
            ax = axes[row, col]
            if scope == "frontier":
                ax.set_facecolor("#F7FBFA")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#78AFA6")
                    spine.set_linewidth(0.9)
            candidates = frontier_models if scope == "frontier" else list(inception.keys())
            models = []
            for model in candidates:
                x = val(inception, model, setting, "Train (openai stats)")
                y = val(inception, model, setting, metric)
                if finite(x) and finite(y) and x <= 8 and y <= 8:
                    models.append(model)
            x = np.array([val(inception, m, setting, "Train (openai stats)") for m in models])
            y = np.array([val(inception, m, setting, metric) for m in models])
            colors = [HIGHLIGHT.get(m, "#B8C2CC") for m in models]
            sizes = [62 if scope == "frontier" else (36 if m in HIGHLIGHT else 16) for m in models]
            ax.scatter(x, y, s=sizes, c=colors, edgecolor="white", linewidth=0.5, alpha=0.95, zorder=3)
            if len(models):
                lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
                hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
                pad = 0.06 * (hi - lo)
                ax.plot([lo, hi], [lo, hi], color="#94A3B8", lw=0.8, ls="--", zorder=1)
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
                rho, n = spearman(x, y)
                ax.text(0.94 if scope == "frontier" else 0.06,
                        0.08 if scope == "frontier" else 0.92,
                        f"$r_s$={rho:.2f}, $n$={n}", transform=ax.transAxes,
                        ha="right" if scope == "frontier" else "left",
                        va="bottom" if scope == "frontier" else "top", fontsize=6.5, color="#334155")
            if row == 0:
                ax.set_title(
                    col_label,
                    fontsize=8.2,
                    fontweight="bold" if scope == "frontier" else "normal",
                    color="#2F6F66" if scope == "frontier" else "#334155",
                )
            ax.set_xlabel("Conventional train FID" if row == len(alternatives) - 1 else "", fontsize=6.7)
            ax.set_ylabel(f"{row_label} FD" if col == 0 else "", fontsize=6.7, labelpad=3)
            ax.grid(True)
            ax.tick_params(labelsize=6.0, length=2.0)
    legend_order = ["BAR-L", "REPA-E", "RAE-DDT", "xAR-H", "xAR-L", "BAR-B"]
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=HIGHLIGHT[m], markeredgecolor="white", markersize=5.2, label=m)
        for m in legend_order
    ]
    handles.append(
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#B8C2CC", markeredgecolor="white", markersize=5.2, label="other models")
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.012), ncol=7, frameon=False, fontsize=7.0)
    fig.tight_layout(pad=0.50, h_pad=0.66, w_pad=0.56, rect=(0.0, 0.045, 1, 1))
    for row in range(4):
        source, zoom = axes[row, 0], axes[row, 1]
        x0, x1 = zoom.get_xlim()
        y0, y1 = zoom.get_ylim()
        source.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                   edgecolor="#78AFA6", linewidth=0.8, zorder=4))
        fig.add_artist(ConnectionPatch((x1, (y0+y1)/2), (0, 0.5),
                       coordsA="data", coordsB="axes fraction", axesA=source, axesB=zoom,
                       color="#78AFA6", linewidth=0.55, linestyle="--", alpha=0.7))
    fig.savefig(FIG_DIR / "protocol_scatter_panel.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "protocol_scatter_panel.pdf", bbox_inches="tight")
    plt.close(fig)


def models_by_metric(table, setting, metric, n=20):
    rows = [
        (m, val(table, m, setting, metric))
        for m in table
        if finite(val(table, m, setting, metric))
    ]
    rows.sort(key=lambda x: x[1])
    return [m for m, _ in rows[:n]]


def make_appendix_protocol_value_heatmaps(inception):
    protocols = protocol_metrics()
    settings = [("CFG", "CFG"), ("w/o CFG", "w/o CFG")]
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 6.6))
    for ax, (title, setting) in zip(axes, settings):
        models = models_by_metric(inception, setting, "Train (openai stats)", 20)
        data = np.array([[val(inception, m, setting, metric) for _, metric in protocols] for m in models])
        im = ax.imshow(data, cmap="YlGnBu_r", aspect="auto")
        ax.set_title(title, fontsize=9, pad=6, color="#334155")
        ax.set_xticks(range(len(protocols)))
        ax.set_xticklabels([p[0] for p in protocols], rotation=45, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.tick_params(labelsize=6.4, length=0)
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if finite(data[i, j]):
                    norm = (data[i, j] - vmin) / max(vmax - vmin, 1e-6)
                    color = "#F8FAFC" if norm < 0.42 else "#0F172A"
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=5.7, color=color)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(pad=0.45)
    fig.savefig(FIG_DIR / "appendix_protocol_value_heatmaps.png", bbox_inches="tight")
    plt.close(fig)


def make_appendix_protocol_delta_heatmaps(inception):
    protocols = protocol_metrics()[1:]
    settings = [("CFG", "CFG"), ("w/o CFG", "w/o CFG")]
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 6.6))
    cmap = LinearSegmentedColormap.from_list("delta_map", ["#2B6CB0", "#F7FAFC", "#E76F51"])
    for ax, (title, setting) in zip(axes, settings):
        models = models_by_metric(inception, setting, "Train (openai stats)", 20)
        base = np.array([val(inception, m, setting, "Train (openai stats)") for m in models])
        data = np.array([[val(inception, m, setting, metric) for _, metric in protocols] for m in models])
        delta = data - base[:, None]
        vmax = max(0.1, np.nanmax(np.abs(delta)))
        im = ax.imshow(delta, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=9, pad=6, color="#334155")
        ax.set_xticks(range(len(protocols)))
        ax.set_xticklabels([p[0] for p in protocols], rotation=45, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.tick_params(labelsize=6.4, length=0)
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                if finite(delta[i, j]):
                    ax.text(j, i, f"{delta[i, j]:+.2f}", ha="center", va="center", fontsize=5.7, color="#0F172A")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(pad=0.45)
    fig.savefig(FIG_DIR / "appendix_protocol_delta_heatmaps.png", bbox_inches="tight")
    plt.close(fig)


def make_appendix_cfg_nocfg_scatter(inception):
    protocols = protocol_metrics()
    fig, axes = plt.subplots(1, 5, figsize=(7.35, 2.55))
    for ax, (label, metric) in zip(axes, protocols):
        models = []
        for m in inception:
            cfg = val(inception, m, "CFG", metric)
            no_cfg = val(inception, m, "w/o CFG", metric)
            if finite(cfg) and finite(no_cfg) and cfg <= 8 and no_cfg <= 8:
                models.append(m)
        x = np.array([val(inception, m, "CFG", metric) for m in models])
        y = np.array([val(inception, m, "w/o CFG", metric) for m in models])
        colors = [HIGHLIGHT.get(m, "#B8C2CC") for m in models]
        sizes = [35 if m in HIGHLIGHT else 15 for m in models]
        ax.scatter(x, y, s=sizes, c=colors, edgecolor="white", linewidth=0.5, alpha=0.95)
        lo = np.nanmin([x.min(), y.min()])
        hi = np.nanmax([x.max(), y.max()])
        pad = 0.05 * (hi - lo)
        ax.plot([lo, hi], [lo, hi], color="#94A3B8", lw=0.8, ls="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(label, fontsize=7.4, color="#334155")
        ax.set_xlabel("CFG", fontsize=6.7)
        if ax is axes[0]:
            ax.set_ylabel("w/o CFG", fontsize=6.7)
        ax.grid(True)
        ax.tick_params(labelsize=6.0, length=2.0)
    fig.tight_layout(pad=0.55)
    fig.savefig(FIG_DIR / "appendix_cfg_nocfg_scatter.png", bbox_inches="tight")
    plt.close(fig)


def make_appendix_backbone_rank_delta(tables):
    values = validation_values(tables)
    names = ["CLIP", "DINOv2", "VGG16"]
    ranks = {name: rank_values({m: values[m][name] for m in values}) for name in ["Inception"] + names}
    models = sorted(values, key=lambda m: ranks["Inception"][m])
    data = np.array([[ranks[name][m] - ranks["Inception"][m] for name in names] for m in models])
    vmax = max(4, int(np.nanmax(np.abs(data))))
    cmap = LinearSegmentedColormap.from_list("rank_delta", ["#2B6CB0", "#F7FAFC", "#E76F51"])
    fig, ax = plt.subplots(figsize=(4.6, 7.35))
    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.tick_params(labelsize=6.4, length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{int(data[i, j]):+d}", ha="center", va="center", fontsize=5.8, color="#0F172A")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.035)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=6.5, length=0)
    fig.tight_layout(pad=0.45)
    fig.savefig(FIG_DIR / "appendix_backbone_rank_delta.png", bbox_inches="tight")
    plt.close(fig)


def make_appendix_backbone_distributions(tables):
    values = validation_values(tables)
    names = ["Inception", "CLIP", "DINOv2", "VGG16"]
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 4.4))
    for ax, name in zip(axes.ravel(), names):
        vals = np.array([values[m][name] for m in values])
        ax.hist(vals, bins=12, color="#BFD7EA", edgecolor="white")
        median = float(np.median(vals))
        ax.axvline(median, color="#E76F51", lw=1.4)
        ax.text(0.95, 0.90, f"median {median:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color="#E76F51")
        for model, color in HIGHLIGHT.items():
            if model in values:
                ax.axvline(values[model][name], color=color, lw=1.0, alpha=0.9)
        ax.set_title(name, fontsize=8.5, color="#334155")
        ax.set_xlabel("Validation-reference FD", fontsize=7.2)
        ax.set_ylabel("count", fontsize=7.2)
        ax.grid(axis="y")
        ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_DIR / "appendix_backbone_distributions.png", bbox_inches="tight")
    plt.close(fig)


def make_appendix_validation_delta_by_backbone(tables):
    names = ["Inception", "CLIP", "DINOv2", "VGG16"]
    models = top_models(tables["Inception"], 16)
    fig, axes = plt.subplots(1, 4, figsize=(7.35, 5.2), sharey=True)
    y = np.arange(len(models))
    for ax, name in zip(axes, names):
        table = tables[name]
        deltas = np.array([
            val(table, m, "CFG", "Val") - val(table, m, "CFG", "Train (balanced)")
            for m in models
        ])
        ax.axvline(0, color="#94A3B8", lw=0.8)
        for i, (model, d) in enumerate(zip(models, deltas)):
            color = HIGHLIGHT.get(model, "#B8C2CC")
            ax.hlines(i, 0, d, color="#CAD5E0", lw=2.0)
            ax.scatter(d, i, s=30 if model in HIGHLIGHT else 18, color=color, edgecolor="white", lw=0.55, zorder=3)
        ax.set_title(name, fontsize=8.0, color="#334155")
        ax.set_xlabel("Val - balanced", fontsize=6.8)
        ax.grid(axis="x")
        ax.tick_params(labelsize=6.1, length=2.0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(models)
    axes[0].invert_yaxis()
    fig.tight_layout(pad=0.55, w_pad=0.35)
    fig.savefig(FIG_DIR / "appendix_validation_delta_by_backbone.png", bbox_inches="tight")
    plt.close(fig)


def make_appendix_protocol_signature_lines(inception):
    protocols = protocol_metrics()
    settings = [("CFG", "CFG"), ("w/o CFG", "w/o CFG")]
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.4), sharex=True)
    x = np.arange(len(protocols))
    for ax, (title, setting) in zip(axes, settings):
        models = models_by_metric(inception, setting, "Train (openai stats)", 12)
        for model in models:
            y = [val(inception, model, setting, metric) for _, metric in protocols]
            color = HIGHLIGHT.get(model, "#AAB5BF")
            lw = 1.6 if model in HIGHLIGHT else 0.9
            ax.plot(x, y, marker="o", ms=3.0, lw=lw, color=color, alpha=0.9)
            ax.text(x[-1] + 0.05, y[-1], model, va="center", fontsize=5.9, color=color)
        ax.set_title(title, fontsize=8.5, color="#334155")
        ax.set_xticks(x)
        ax.set_xticklabels([p[0] for p in protocols], rotation=30, ha="right")
        ax.set_ylabel("Inception FD")
        ax.grid(True)
        ax.tick_params(labelsize=6.2)
        ax.set_xlim(-0.15, len(protocols) - 0.35)
    fig.tight_layout(pad=0.55)
    fig.savefig(FIG_DIR / "appendix_protocol_signature_lines.png", bbox_inches="tight")
    plt.close(fig)


def make_appendix_reference_shift_summary(inception):
    protocols = protocol_metrics()[1:]
    settings = [("CFG", "CFG"), ("w/o CFG", "w/o CFG")]
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.15), sharey=True)
    for ax, (title, setting) in zip(axes, settings):
        base_metric = "Train (openai stats)"
        models = [
            m
            for m in inception
            if finite(val(inception, m, setting, base_metric))
            and val(inception, m, setting, base_metric) <= 8
        ]
        for j, (label, metric) in enumerate(protocols):
            rows = [
                (m, val(inception, m, setting, metric) - val(inception, m, setting, base_metric))
                for m in models
                if finite(val(inception, m, setting, metric))
            ]
            rows.sort(key=lambda x: x[1])
            deltas = np.array([d for _, d in rows])
            if len(deltas) == 0:
                continue
            jitter = np.linspace(-0.17, 0.17, len(rows))
            for k, (model, delta) in enumerate(rows):
                color = HIGHLIGHT.get(model, "#AAB5BF")
                size = 31 if model in HIGHLIGHT else 13
                alpha = 0.95 if model in HIGHLIGHT else 0.42
                ax.scatter(j + jitter[k], delta, s=size, color=color, edgecolor="white", lw=0.45, alpha=alpha, zorder=3)
            q1, med, q3 = np.percentile(deltas, [25, 50, 75])
            ax.vlines(j, q1, q3, color="#334155", lw=1.2, zorder=4)
            ax.scatter(j, med, s=34, color="#334155", edgecolor="white", lw=0.6, zorder=5)
            ax.text(j, med + 0.08, f"{med:+.2f}", ha="center", va="bottom", fontsize=6.4, color="#334155")
        ax.axhline(0, color="#94A3B8", lw=0.9, ls="--")
        ax.set_title(title, fontsize=8.5, color="#334155")
        ax.set_xticks(range(len(protocols)))
        ax.set_xticklabels([p[0] for p in protocols], rotation=35, ha="right")
        ax.set_ylabel("Protocol shift from conventional FID")
        ax.grid(axis="y")
        ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.55)
    fig.savefig(FIG_DIR / "appendix_reference_shift_summary.png", bbox_inches="tight")
    plt.close(fig)


def make_appendix_backbone_winner_board(tables):
    protocols = [
        ("Balanced", "Train (balanced)"),
        ("Balanced732", "Train (balanced732)"),
        ("Validation", "Val"),
    ]
    names = ["Inception", "CLIP", "DINOv2", "VGG16"]
    fig, ax = plt.subplots(figsize=(7.35, 3.35))
    ax.set_xlim(-0.86, len(protocols))
    ax.set_ylim(-0.08, len(names) + 0.62)
    ax.axis("off")

    for col, (label, _) in enumerate(protocols):
        ax.text(col + 0.46, len(names) + 0.32, label, ha="center", va="bottom", fontsize=8.2, color="#334155")

    for row, name in enumerate(names):
        y0 = len(names) - row - 1
        ax.text(-0.42, y0 + 0.43, name, ha="right", va="center", fontsize=8.4, fontweight="bold", color="#334155")
        table = tables[name]
        for col, (_, metric) in enumerate(protocols):
            entries = top_entries(table, "CFG", metric, 3)
            if not entries:
                continue
            winner, winner_score = entries[0]
            is_validation = metric == "Val"
            face = "#ECF7F4" if is_validation else "#F8FBFD"
            edge = "#BFDAD4" if is_validation else "#D9E2EA"
            rect = FancyBboxPatch(
                (col + 0.02, y0),
                0.90,
                0.82,
                boxstyle="round,pad=0.018,rounding_size=0.035",
                linewidth=0.75,
                edgecolor=edge,
                facecolor=face,
            )
            ax.add_patch(rect)
            color = HIGHLIGHT.get(winner, "#64748B")
            ax.plot([col + 0.07, col + 0.07], [y0 + 0.14, y0 + 0.68], color=color, lw=2.3, solid_capstyle="round")
            ax.text(col + 0.13, y0 + 0.54, winner, ha="left", va="center", fontsize=9.4, fontweight="bold", color=color)
            ax.text(col + 0.13, y0 + 0.28, f"{winner_score:.2f}", ha="left", va="center", fontsize=8.0, color="#475569")
    fig.tight_layout(pad=0.1)
    fig.savefig(FIG_DIR / "appendix_backbone_winner_board.png", bbox_inches="tight")
    plt.close(fig)


def make_appendix_comparable_band_backbones(tables):
    names = ["Inception", "CLIP", "DINOv2", "VGG16"]
    values = validation_values(tables)
    models = [
        m
        for m in values
        if finite(val(tables["Inception"], m, "CFG", "Train (openai stats)"))
        and val(tables["Inception"], m, "CFG", "Train (openai stats)") <= 3.2
    ]
    models.sort(key=lambda m: val(tables["Inception"], m, "CFG", "Train (openai stats)"))
    fig, axes = plt.subplots(1, 4, figsize=(7.35, 2.95), sharex=True)
    for ax, name in zip(axes, names):
        x = np.array([val(tables["Inception"], m, "CFG", "Train (openai stats)") for m in models])
        y = np.array([values[m][name] for m in models])
        colors = [HIGHLIGHT.get(m, "#B8C2CC") for m in models]
        sizes = [42 if m in HIGHLIGHT else 18 for m in models]
        ax.axvspan(1.0, 2.0, color="#E8F5F2", zorder=0)
        ax.scatter(x, y, s=sizes, c=colors, edgecolor="white", linewidth=0.55, alpha=0.95, zorder=3)
        for model in ["BAR-L", "REPA-E", "RAE-DDT", "xAR-H"]:
            if model in models:
                mx = val(tables["Inception"], model, "CFG", "Train (openai stats)")
                my = values[model][name]
                ax.text(mx + 0.025, my, model, fontsize=5.8, color=HIGHLIGHT.get(model, "#334155"), va="center")
        rho, n = spearman(x, y)
        ax.text(0.06, 0.92, f"$r_s$={rho:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color="#334155")
        ax.set_title(name, fontsize=8.0, color="#334155")
        ax.set_xlabel("Train FID", fontsize=6.7)
        ax.set_ylabel("Validation FD", fontsize=6.7)
        ax.grid(True)
        ax.tick_params(labelsize=6.0, length=2.0)
    fig.tight_layout(pad=0.55, w_pad=0.45)
    fig.savefig(FIG_DIR / "appendix_comparable_band_backbones.png", bbox_inches="tight")
    plt.close(fig)


def make_appendix_leaderboard_plateau(inception):
    settings = [("CFG", "CFG"), ("w/o CFG", "w/o CFG")]
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.0), sharey=True)
    for ax, (title, setting) in zip(axes, settings):
        rows = [
            (m, val(inception, m, setting, "Train (openai stats)"))
            for m in inception
            if finite(val(inception, m, setting, "Train (openai stats)"))
            and val(inception, m, setting, "Train (openai stats)") <= 8
        ]
        rows.sort(key=lambda x: x[1])
        rows = rows[:24]
        x = np.arange(1, len(rows) + 1)
        y = np.array([score for _, score in rows])
        ax.axhspan(1.0, 2.0, color="#E8F5F2", zorder=0)
        ax.plot(x, y, color="#94A3B8", lw=1.2, zorder=1)
        for i, (model, score) in enumerate(rows, start=1):
            color = HIGHLIGHT.get(model, "#B8C2CC")
            size = 42 if model in HIGHLIGHT else 20
            ax.scatter(i, score, s=size, color=color, edgecolor="white", lw=0.6, zorder=3)
            if i <= 8 or model in ["xAR-H", "xAR-L"]:
                ax.text(i, score + 0.08, model, ha="center", va="bottom", rotation=60, fontsize=5.7, color=color)
        ax.text(0.04, 0.36, "comparable\nFID band", transform=ax.transAxes, ha="left", va="center", fontsize=7.0, color="#2F6F66")
        ax.set_title(title, fontsize=8.5, color="#334155")
        ax.set_xlabel("Conventional leaderboard rank")
        ax.set_ylabel("Train-reference FID")
        ax.set_xlim(0.4, len(rows) + 0.6)
        ax.set_ylim(0.75, min(8.3, max(4.2, float(np.nanmax(y)) + 0.35)))
        ax.grid(axis="y")
        ax.tick_params(labelsize=6.4)
    fig.tight_layout(pad=0.55)
    fig.savefig(FIG_DIR / "appendix_leaderboard_plateau.png", bbox_inches="tight")
    plt.close(fig)


def make_appendix_backbone_rank_spread(tables):
    values = validation_values(tables)
    names = ["Inception", "CLIP", "DINOv2", "VGG16"]
    ranks = {name: rank_values({m: values[m][name] for m in values}) for name in names}
    rows = []
    for model in values:
        rank_list = [ranks[name][model] for name in names]
        rows.append(
            {
                "model": model,
                "fid": val(tables["Inception"], model, "CFG", "Train (openai stats)"),
                "inception_val": values[model]["Inception"],
                "spread": max(rank_list) - min(rank_list),
            }
        )
    rows = [r for r in rows if finite(r["fid"]) and r["fid"] <= 5.0]
    fig, ax = plt.subplots(figsize=(7.35, 3.05))
    ax.axvspan(1.0, 2.0, color="#E8F5F2", zorder=0)
    for row in rows:
        model = row["model"]
        color = HIGHLIGHT.get(model, "#B8C2CC")
        size = 48 if model in HIGHLIGHT else 22
        ax.scatter(row["fid"], row["spread"], s=size, color=color, edgecolor="white", lw=0.65, alpha=0.95, zorder=3)
    labels = sorted(rows, key=lambda r: (-r["spread"], r["fid"]))[:8]
    labels.extend([r for r in rows if r["model"] in ["BAR-L", "REPA-E", "RAE-DDT"]])
    seen = set()
    for row in labels:
        if row["model"] in seen:
            continue
        seen.add(row["model"])
        ax.text(row["fid"] + 0.04, row["spread"] + 0.15, row["model"], fontsize=6.0, color=HIGHLIGHT.get(row["model"], "#334155"))
    ax.set_xlabel("Conventional train-reference FID")
    ax.set_ylabel("Rank spread across four backbones")
    ax.set_xlim(0.85, 5.1)
    ax.set_ylim(-0.5, max(r["spread"] for r in rows) + 2.0)
    ax.grid(True)
    ax.tick_params(labelsize=6.8)
    fig.tight_layout(pad=0.55)
    fig.savefig(FIG_DIR / "appendix_backbone_rank_spread.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    tables = {name: load_table(filename) for name, filename in FILES.items()}
    inception = tables["Inception"]
    make_fid_leaderboard_progress(inception)
    make_train_val_scatter(inception)
    make_protocol_winner_board(inception)
    make_rank_shift(inception)
    make_protocol_scatter_panel(inception)
    shown_stats, all_stats, shown_n, all_n = make_pairwise_scatter(tables)
    matrix, matrix_n = make_spearman_matrix(tables)
    rank_n = make_backbone_rank_scatter(tables)
    panel_shown, panel_all = make_backbone_scatter_panel(tables)
    make_protocol_ranges(inception)
    make_margin_vs_shift(inception)
    make_appendix_protocol_value_heatmaps(inception)
    make_appendix_protocol_delta_heatmaps(inception)
    make_appendix_cfg_nocfg_scatter(inception)
    make_appendix_backbone_rank_delta(tables)
    make_appendix_backbone_distributions(tables)
    make_appendix_validation_delta_by_backbone(tables)
    make_appendix_protocol_signature_lines(inception)
    make_appendix_reference_shift_summary(inception)
    make_appendix_backbone_winner_board(tables)
    make_appendix_comparable_band_backbones(tables)
    make_appendix_backbone_rank_spread(tables)

    print(f"pairwise scatter shown models: {shown_n}; all overlapping models: {all_n}")
    for name, (rho, n) in shown_stats.items():
        all_rho, all_count = all_stats[name]
        print(f"{name}: shown Spearman={rho:.3f} (n={n}); all Spearman={all_rho:.3f} (n={all_count})")
    print(f"spearman matrix models: {matrix_n}")
    print(np.array2string(matrix, precision=3))
    print(f"backbone rank scatter models: {rank_n}")
    print(f"backbone scatter panel shown models: {panel_shown}; rank models: {panel_all}")


if __name__ == "__main__":
    raise SystemExit('Use: python analysis/reproduce_paper.py --output-dir outputs/paper')
