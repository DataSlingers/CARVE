"""Schematic guidance figures for S4 Text (practical guidance).

figure_guidance_global reads the global stability and generalizability curves.
figure_guidance_clusters reads single clusters (dartboards), compares two
candidate solutions and reads single samples. Everything is drawn from
hand-specified shapes and seeded draws, not from data, so neither function
takes a fit. Both write a PNG and a PDF under vis/guidance/:

    python -c "from benchmarks.figures import figure_guidance_global, \
        figure_guidance_clusters; figure_guidance_global(); figure_guidance_clusters()"
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from .._theme import (
    FOREGROUND_COLOR,
    GUIDE_COLORS,
    MEASURE_LINESTYLES,
    METRIC_COLORS,
    PRINT_WIDTH_IN,
    SEQUENTIAL_CMAP_NAME,
    cluster_colors,
    print_theme_context,
    save_figure,
)
from ._paths import GUIDANCE_DIR, figure_path

STABILITY = METRIC_COLORS["ari_stability"]
GENERALIZABILITY = METRIC_COLORS["ari_generalizability"]
MEASURE_COLORS = {"stability": STABILITY, "generalizability": GENERALIZABILITY}

SEED = 20261016
BODY = 8.0
TITLE = 9.0
MARGIN = 0.05
GAP = 0.1
HEADER_H = 0.24
LINE_H = 0.145  # one 8 pt text line at linespacing 1.25, in inches
BADGE_R = 0.08  # radius of the numbered scenario discs, in inches
AXIS_LABEL = 10.0  # the stable / generalizable labels around the quadrants

K = np.arange(2, 11)


# --------------------------------------------------------------------------
# Canvas: one background axes in inches (origin top left) plus inset axes.
# --------------------------------------------------------------------------


class Canvas:
    """A figure whose background axes is laid out in inches from the top left."""

    def __init__(self, height: float, width: float = PRINT_WIDTH_IN) -> None:
        self.W, self.H = width, height
        self.fig = plt.figure(figsize=(width, height))
        bg = self.fig.add_axes((0, 0, 1, 1))
        bg.set_xlim(0, width)
        bg.set_ylim(height, 0)
        bg.axis("off")
        self.bg = bg

    def axes(self, x: float, y: float, w: float, h: float) -> Axes:
        """An inset axes at (x, y) inches from the top left, w by h inches."""
        return self.fig.add_axes(
            (x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H)
        )

    def box(self, x, y, w, h, color, radius=0.06) -> None:
        self.bg.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=color,
                edgecolor="none",
            )
        )

    def text(self, x, y, s, *, size=BODY, color=FOREGROUND_COLOR, **kw) -> None:
        self.bg.text(x, y, s, size=size, color=color, **kw)

    def para(self, x, y, w, s, *, size=BODY, color=FOREGROUND_COLOR) -> int:
        """Wrap s to width w inches at the top-left anchor; return the line count."""
        if not s:
            return 0
        wrapped = wrap(s, w, size)
        self.text(x, y, wrapped, size=size, color=color, va="top", linespacing=1.25)
        return wrapped.count("\n") + 1

    def header(self, y: float, letter: str, title: str, note: str = "") -> None:
        x0, w = MARGIN, self.W - 2 * MARGIN
        badge_w = HEADER_H * 1.2
        self.box(x0, y, w, HEADER_H, GUIDE_COLORS["header"], radius=0.05)
        self.box(x0, y, badge_w, HEADER_H, GUIDE_COLORS["badge"], radius=0.05)
        self.text(
            x0 + badge_w / 2,
            y + HEADER_H / 2,
            letter,
            size=TITLE,
            color="white",
            weight="bold",
            ha="center",
            va="center",
        )
        self.text(
            x0 + badge_w + 0.1,
            y + HEADER_H / 2,
            title.upper(),
            size=TITLE,
            color=GUIDE_COLORS["badge"],
            weight="bold",
            va="center",
        )
        if note:
            self.text(
                x0 + w - 0.1,
                y + HEADER_H / 2,
                note,
                color=GUIDE_COLORS["badge"],
                ha="right",
                va="center",
            )

    def badge(self, x: float, y: float, number: int, *, on_top: bool = False) -> float:
        """A numbered disc, left edge x and center y; return its right edge.

        on_top draws it in its own axes, above inset axes added before it.
        """
        ax = self.bg
        if on_top:
            ax = self.axes(x, y - BADGE_R, 2 * BADGE_R, 2 * BADGE_R)
            ax.set_xlim(x, x + 2 * BADGE_R)
            ax.set_ylim(y + BADGE_R, y - BADGE_R)
            ax.axis("off")
        ax.add_patch(
            Circle((x + BADGE_R, y), BADGE_R, facecolor=GUIDE_COLORS["badge"], lw=0)
        )
        ax.text(
            x + BADGE_R,
            y,
            str(number),
            size=BODY,
            color="white",
            weight="bold",
            ha="center",
            va="center_baseline",
        )
        return x + 2 * BADGE_R

    def card_title(self, x: float, y: float, title: str, number: int) -> None:
        """A card's scenario number and title, top left of the card at (x, y)."""
        y_mid = y + 0.17
        right = self.badge(x + 0.12, y_mid, number)
        self.text(right + 0.07, y_mid, title, size=TITLE, weight="bold", va="center")


def chars_per_line(width_in: float, size: float) -> int:
    """Characters of running text that fit in width_in at size points."""
    return max(12, int(width_in * 72 / (size * 0.44)))


def wrap(s: str, width_in: float, size: float = BODY) -> str:
    return textwrap.fill(s, chars_per_line(width_in, size), break_on_hyphens=False)


def n_lines(s: str, width_in: float, size: float = BODY) -> int:
    return wrap(s, width_in, size).count("\n") + 1 if s else 0


def bare(ax: Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


# --------------------------------------------------------------------------
# S Fig A: global curves
# --------------------------------------------------------------------------


def bump(center: float, width: float, height: float) -> np.ndarray:
    return height * np.exp(-(((K - center) / width) ** 2))


def spike(at: int, height: float) -> np.ndarray:
    return np.where(K == at, height, 0.0)


def curve_frame(ax: Axes, *, xmax: float = 10.6, ylabel: bool = True) -> None:
    bare(ax)
    ax.set_xlim(1.4, xmax)
    ax.set_ylim(0, 1.08)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(GUIDE_COLORS["axis"])
        ax.spines[side].set_linewidth(0.8)
    muted = GUIDE_COLORS["muted"]
    ax.text(
        1.0,
        -0.03,
        "k",
        transform=ax.transAxes,
        ha="right",
        va="top",
        size=BODY,
        color=muted,
    )
    if ylabel:
        ax.text(
            -0.03,
            0.5,
            "ARI",
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            size=BODY,
            color=muted,
        )


def draw_curve(ax: Axes, mean, se, measure: str, *, faded: bool = False) -> None:
    color = MEASURE_COLORS[measure]
    alpha = 0.4 if faded else 1.0
    ax.fill_between(K, mean - se, mean + se, color=color, alpha=0.18 * alpha, lw=0)
    ax.plot(
        K,
        mean,
        color=color,
        ls=MEASURE_LINESTYLES[measure],
        lw=1.3,
        alpha=alpha,
        marker="o",
        ms=2.4,
    )


def candidate(ax: Axes, k: float) -> None:
    ax.axvline(k, color=GUIDE_COLORS["candidate"], ls=(0, (3, 2)), lw=0.9, zorder=0)


def bracket(ax: Axes, k0: float, k1: float, y: float, label: str) -> None:
    muted = GUIDE_COLORS["muted"]
    ax.annotate(
        "",
        xy=(k0, y),
        xytext=(k1, y),
        arrowprops=dict(arrowstyle="|-|", color=muted, lw=0.8, mutation_scale=2.5),
    )
    ax.text(
        (k0 + k1) / 2, y + 0.03, label, ha="center", va="bottom", size=BODY, color=muted
    )


def se_of(mean, base=0.022):
    return np.full_like(mean, base, dtype=float)


def card_one_peak(ax):
    s = 0.52 + bump(5, 1.5, 0.41)
    g = s - 0.07
    draw_curve(ax, s, se_of(s), "stability")
    draw_curve(ax, g, se_of(g), "generalizability")
    candidate(ax, 5)


def card_plateau(ax):
    s = np.array([0.58, 0.79, 0.9, 0.905, 0.9, 0.9, 0.74, 0.62, 0.55])
    g = s - 0.065
    draw_curve(ax, s, se_of(s), "stability")
    draw_curve(ax, g, se_of(g), "generalizability")
    bracket(ax, 4, 7, 0.95, "plateau")
    candidate(ax, 7)


def card_method_on_top(ax):
    s1 = 0.6 + bump(5, 1.8, 0.3)
    s2 = 0.42 + bump(4, 1.8, 0.14)
    draw_curve(ax, s2, se_of(s2), "stability", faded=True)
    draw_curve(ax, s2 - 0.06, se_of(s2), "generalizability", faded=True)
    draw_curve(ax, s1, se_of(s1), "stability")
    draw_curve(ax, s1 - 0.06, se_of(s1), "generalizability")
    muted = GUIDE_COLORS["muted"]
    ax.text(
        8.6, s1[-2] + 0.06, "method 1", ha="center", va="bottom", size=BODY, color=muted
    )
    ax.text(
        8.6, s2[-2] - 0.13, "method 2", ha="center", va="top", size=BODY, color=muted
    )
    candidate(ax, 5)


def card_disagree(ax):
    s = np.array([0.95, 0.92, 0.88, 0.8, 0.72, 0.66, 0.62, 0.58, 0.55])
    g = np.array([0.74, 0.8, 0.9, 0.79, 0.7, 0.63, 0.58, 0.54, 0.5])
    draw_curve(ax, s, se_of(s), "stability")
    draw_curve(ax, g, se_of(g), "generalizability")
    candidate(ax, 2)
    candidate(ax, 4)


def card_spikes(ax):
    base = np.linspace(0.68, 0.56, K.size)
    s = base + spike(4, 0.2) + spike(7, 0.19)
    g = s - 0.07
    draw_curve(ax, s, se_of(s), "stability")
    draw_curve(ax, g, se_of(g), "generalizability")
    candidate(ax, 4)
    candidate(ax, 7)


def card_high_at_two(ax):
    # k = 2 scores high because the coarsest split is the easiest to
    # reproduce; the grey band shows it excluded (not_two, or a sweep that
    # starts higher), which leaves the later peak as the candidate.
    s = np.array([0.95, 0.74, 0.68, 0.78, 0.88, 0.8, 0.7, 0.63, 0.57])
    g = s - 0.07
    ax.add_patch(
        Rectangle(
            (1.4, 0),
            1.1,
            1.08,
            facecolor=GUIDE_COLORS["untested"],
            edgecolor="none",
            zorder=0,
        )
    )
    ax.text(
        1.95,
        0.32,
        "excluded",
        ha="center",
        va="center",
        rotation=90,
        size=BODY,
        color=GUIDE_COLORS["muted"],
    )
    draw_curve(ax, s, se_of(s), "stability")
    draw_curve(ax, g, se_of(g), "generalizability")
    candidate(ax, 6)


def card_edge(ax):
    s = 0.45 + 0.44 * (K - 2) / 8
    g = s - 0.07
    ax.add_patch(
        Rectangle(
            (10.5, 0),
            1.9,
            1.08,
            facecolor=GUIDE_COLORS["untested"],
            edgecolor="none",
            zorder=0,
        )
    )
    ax.text(
        11.45,
        0.42,
        "not\ntested",
        ha="center",
        va="center",
        size=BODY,
        color=GUIDE_COLORS["muted"],
    )
    draw_curve(ax, s, se_of(s), "stability")
    draw_curve(ax, g, se_of(g), "generalizability")
    ax.annotate(
        "",
        xy=(12.1, 0.97),
        xytext=(10.2, 0.9),
        arrowprops=dict(
            arrowstyle="-|>", color=GUIDE_COLORS["muted"], lw=0.9, mutation_scale=7
        ),
    )


def card_overlap(ax):
    wiggle = np.array([0.0, 0.03, -0.02, 0.025, 0.0, 0.02, -0.015, 0.01, -0.02])
    s = 0.72 + wiggle
    g = 0.64 + wiggle[::-1] * 0.8
    draw_curve(ax, s, se_of(s, 0.075), "stability")
    draw_curve(ax, g, se_of(g, 0.075), "generalizability")
    bracket(ax, 3, 9, 0.93, "tied")


def card_low_flat(ax):
    wiggle = np.array([0.0, 0.02, -0.015, 0.01, 0.02, -0.01, 0.015, -0.02, 0.0])
    s = 0.27 + wiggle
    g = 0.21 + wiggle[::-1]
    draw_curve(ax, s, se_of(s, 0.035), "stability")
    draw_curve(ax, g, se_of(g, 0.035), "generalizability")


GLOBAL_ROWS: list[tuple[str, str, list[tuple]]] = [
    (
        "A",
        "One clear candidate",
        [
            (
                "One clear peak",
                "",
                "Example: Levine",
                card_one_peak,
            ),
            (
                "A plateau",
                "",
                "",
                card_plateau,
            ),
            (
                "One method on top",
                "",
                "Example: Cusanovich, preprocessing",
                card_method_on_top,
            ),
        ],
    ),
    (
        "B",
        "Several plausible candidates",
        [
            (
                "Disagreeing criteria ",
                "",
                "Example: Klein",
                card_disagree,
            ),
            (
                "Several spikes",
                "",
                "Example: Cusanovich",
                card_spikes,
            ),
            (
                "High at k=2, then a peak",
                "",
                "",
                card_high_at_two,
            ),
        ],
    ),
    (
        "C",
        "No clear candidate",
        [
            (
                "Best score at the edge",
                "",
                "",
                card_edge,
            ),
            (
                "Overlapping bands",
                "",
                "",
                card_overlap,
            ),
            (
                "Low and flat everywhere",
                "",
                "",
                card_low_flat,
            ),
        ],
    ),
]


def global_legend(cv: Canvas, y: float) -> None:
    """One key for every card: the two criteria, the band and the candidate line."""
    ax = cv.axes(MARGIN, y, cv.W - 2 * MARGIN, 0.22)
    bare(ax)
    ax.set_xlim(0, cv.W - 2 * MARGIN)
    ax.set_ylim(0, 1)
    entries = [
        (0.1, "stability", "stability ARI"),
        (1.55, "generalizability", "generalizability ARI"),
    ]
    for x, measure, label in entries:
        color = MEASURE_COLORS[measure]
        ax.fill_between([x, x + 0.35], 0.25, 0.75, color=color, alpha=0.18, lw=0)
        ax.plot(
            [x, x + 0.35],
            [0.5, 0.5],
            color=color,
            ls=MEASURE_LINESTYLES[measure],
            lw=1.3,
        )
        ax.text(x + 0.45, 0.5, label, va="center", size=BODY)
    ax.text(
        3.25, 0.5, "band: ±1 SE", va="center", size=BODY, color=GUIDE_COLORS["muted"]
    )
    ax.plot(
        [4.45, 4.45],
        [0.1, 0.9],
        color=GUIDE_COLORS["candidate"],
        ls=(0, (3, 2)),
        lw=0.9,
    )
    ax.text(4.58, 0.5, "candidate k", va="center", size=BODY)


def figure_guidance_global(*, save: bool = True, out_dir: Path | None = None) -> Figure:
    """S Fig A: what to do with each shape of the global curves."""
    card_w = (PRINT_WIDTH_IN - 2 * MARGIN - 2 * GAP) / 3
    plot_h = 1.0
    text_w = card_w - 0.24
    # The tallest text in a row sets that row's card height, so cards line up.
    card_hs = []
    for _, _, cards in GLOBAL_ROWS:
        lines = max(n_lines(text, text_w) for _, text, _, _ in cards)
        has_example = any(example for _, _, example, _ in cards)
        card_hs.append(
            0.36
            + plot_h
            + 0.14
            + LINE_H * lines
            + (0.06 + LINE_H if has_example else 0)
            + 0.1
        )
    legend_h = 0.3
    height = legend_h + sum(HEADER_H + 0.07 + h + 0.14 for h in card_hs) - 0.14

    with print_theme_context():
        cv = Canvas(height)
        global_legend(cv, 0.0)
        y = legend_h
        number = 0
        for (letter, title, cards), card_h in zip(GLOBAL_ROWS, card_hs):
            cv.header(y, letter, title)
            y_card = y + HEADER_H + 0.07
            for i, (card_title, text, example, draw) in enumerate(cards):
                number += 1
                x = MARGIN + i * (card_w + GAP)
                cv.box(x, y_card, card_w, card_h, GUIDE_COLORS["card"])
                cv.card_title(x, y_card, card_title, number)
                ax = cv.axes(x + 0.36, y_card + 0.36, card_w - 0.52, plot_h)
                # One ARI label per row, on its left card.
                curve_frame(ax, xmax=12.4 if draw is card_edge else 10.6, ylabel=i == 0)
                draw(ax)
                y_text = y_card + 0.36 + plot_h + 0.14
                cv.para(x + 0.12, y_text, text_w, text)
                if example:
                    cv.text(
                        x + 0.12,
                        y_card + card_h - 0.08,
                        example,
                        color=GUIDE_COLORS["muted"],
                        va="bottom",
                    )
            y += HEADER_H + 0.07 + card_h + 0.14
        if save:
            for ext in ("png", "pdf"):
                save_figure(
                    cv.fig,
                    figure_path(
                        f"guidance_global.{ext}", subdir=GUIDANCE_DIR, out_dir=out_dir
                    ),
                )
    return cv.fig


# --------------------------------------------------------------------------
# S Fig B: clusters, candidates, samples
# --------------------------------------------------------------------------


def dartboard(ax: Axes, rng, *, offset, spread, color, n: int = 9) -> None:
    """Concentric rings with n darts around offset; darts stay on the board."""
    bare(ax)
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.06, 1.06)
    ax.set_aspect("equal")
    rings = (
        (1.0, GUIDE_COLORS["ring_dark"]),
        (0.75, GUIDE_COLORS["ring_light"]),
        (0.5, GUIDE_COLORS["ring_dark"]),
        (0.25, GUIDE_COLORS["bull"]),
    )
    for radius, face in rings:
        ax.add_patch(
            Circle(
                (0, 0),
                radius,
                facecolor=face,
                edgecolor=GUIDE_COLORS["ring_edge"],
                lw=0.6,
            )
        )
    # Re-centered so the group's mean sits exactly on offset, then pulled in
    # radially onto the board. Redrawing until every dart lands would favor
    # tight draws and shrink the spread the picture is meant to show.
    jitter = rng.normal(0, spread, (n, 2))
    darts = jitter - jitter.mean(axis=0) + np.asarray(offset)
    radius = np.hypot(darts[:, 0], darts[:, 1])
    darts *= np.minimum(1.0, 0.9 / radius)[:, None]
    ax.scatter(
        darts[:, 0],
        darts[:, 1],
        s=13,
        c=color,
        edgecolors=FOREGROUND_COLOR,
        linewidths=0.5,
        zorder=3,
    )


HIGH_S, HIGH_G = (14, 1.6), (12, 1.8)
LOW_S, LOW_G = (5, 2.8), (3, 4.5)


def split_violins(ax: Axes, rng, focal_s, focal_g) -> None:
    """Three clusters, stability left half and generalizability right half; C2 is in question."""
    bare(ax)
    ax.add_patch(
        Rectangle(
            (1.55, 0),
            0.9,
            1.04,
            facecolor=GUIDE_COLORS["focus"],
            edgecolor="none",
            zorder=0,
        )
    )
    stab = [rng.beta(*HIGH_S, 300), rng.beta(*focal_s, 300), rng.beta(*HIGH_S, 300)]
    gen = [rng.beta(*HIGH_G, 300), rng.beta(*focal_g, 300), rng.beta(*HIGH_G, 300)]
    halves: tuple[tuple[list, Literal["low", "high"], str], ...] = (
        (stab, "low", STABILITY),
        (gen, "high", GENERALIZABILITY),
    )
    for data, side, color in halves:
        parts = ax.violinplot(
            data,
            [1, 2, 3],
            side=side,
            widths=0.8,
            showextrema=False,
            bw_method=0.4,
            points=120,
        )
        bodies: list = parts["bodies"]  # type: ignore[assignment]
        for i, body in enumerate(bodies):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            # C1 and C3 are faded so the card reads as being about C2 alone.
            body.set_alpha(0.75 if i == 1 else 0.2)
            body.set_linewidth(0.5)
    ax.set_xlim(0.45, 3.55)
    ax.set_ylim(0, 1.04)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GUIDE_COLORS["axis"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_xticks([1, 2, 3], ["C1", "C2", "C3"])
    ax.tick_params(axis="x", length=0, labelsize=BODY, pad=2)
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(FOREGROUND_COLOR if i == 1 else GUIDE_COLORS["muted"])
        label.set_fontweight("bold" if i == 1 else "normal")


QUADRANTS = [
    # (row, col, offset, spread, focal stability, focal generalizability, title, text)
    # Scenario numbers run row by row: 1 and 2 on top, 3 and 4 below.
    (
        0,
        0,
        (0.0, 0.0),
        0.07,
        HIGH_S,
        HIGH_G,
        "A reliable cluster",
        "",
    ),
    (
        0,
        1,
        (0.5, 0.36),
        0.07,
        HIGH_S,
        LOW_G,
        "A possible rare population",
        "Reappears across resamples, but too small to be learned reliably.",
    ),
    (
        1,
        0,
        (0.0, 0.0),
        0.24,
        LOW_S,
        HIGH_G,
        "A cluster with uncertain membership",
        "Learnable, but cells at the border are inconsistently assigned.",
    ),
    (
        1,
        1,
        (0.22, 0.18),
        0.45,
        LOW_S,
        LOW_G,
        "An unreliable cluster",
        "",
    ),
]


def blob(rng, center, sd, n):
    return rng.normal(0, 1, (n, 2)) * np.asarray(sd) + np.asarray(center)


def scatter_frame(ax: Axes) -> None:
    bare(ax)
    ax.set_aspect("equal", adjustable="datalim")


def candidate_outcomes(rng):
    """Three ways the finer candidate can split cluster C2 of the coarser one.

    Each board is (offset, spread, scenario): the scenario is the number of the
    quadrant in section A whose dart pattern the board repeats.
    """
    c1 = blob(rng, (-1.7, 0.0), (0.42, 0.5), 110)
    outcomes = []
    # 1: two real sub-groups.
    a = blob(rng, (0.85, 0.6), (0.3, 0.28), 70)
    b = blob(rng, (1.6, -0.55), (0.3, 0.28), 70)
    outcomes.append(
        dict(
            parts=[c1, a, b],
            boards=[((0.0, 0.0), 0.07, 1), ((0.0, 0.0), 0.08, 1)],
            title="Reliable structure",
            text="",
        )
    )
    # 2: one continuous group cut in two. The cut moves between resamples, so
    # labels interleave near it (not stable), while each half is still
    # learnable (generalizable): darts spread around the center.
    whole = blob(rng, (1.2, 0.0), (0.62, 0.36), 140)
    cut = whole[:, 0] - 1.2 + rng.normal(0, 0.2, len(whole)) < 0
    outcomes.append(
        dict(
            parts=[c1, whole[cut], whole[~cut]],
            boards=[((0.0, 0.0), 0.24, 3), ((0.0, 0.0), 0.24, 3)],
            title="Continuum or splitting artifact",
            text="",
        )
    )
    # 3: a small tight sliver at the edge.
    rest = blob(rng, (1.05, -0.1), (0.42, 0.45), 130)
    sliver = blob(rng, (2.05, 0.95), (0.09, 0.09), 11)
    outcomes.append(
        dict(
            parts=[c1, rest, sliver],
            boards=[((0.0, 0.0), 0.07, 1), ((0.5, -0.38), 0.07, 2)],
            title="Potential rare population",
            text="",
        )
    )
    return outcomes


def sample_cases(rng):
    """Per-sample stability for three patterns; values in [0, 1], 1 most stable."""
    cases = []
    # Border cells between two touching clusters.
    left = blob(rng, (-0.85, 0.0), (0.42, 0.5), 130)
    right = blob(rng, (0.85, 0.0), (0.42, 0.5), 130)
    pts = np.vstack([left, right])
    stab = 1 - 0.75 * np.exp(-(pts[:, 0] ** 2) / (2 * 0.28**2))
    cases.append(
        dict(
            pts=pts,
            stab=np.clip(stab + rng.normal(0, 0.03, len(pts)), 0, 1),
            title="Unreliable cells along a border",
            text="",
        )
    )
    # Isolated unreliable cells.
    left = blob(rng, (-1.1, 0.1), (0.35, 0.4), 120)
    right = blob(rng, (1.1, -0.1), (0.35, 0.4), 120)
    lone = np.array(
        [
            [-0.1, 1.15],
            [0.2, -1.2],
            [-2.0, -1.05],
            [1.95, 1.05],
            [0.05, 0.1],
            [-1.85, 1.2],
        ]
    )
    pts = np.vstack([left, right, lone])
    stab = np.concatenate([rng.beta(30, 1.5, 240), rng.uniform(0.15, 0.35, len(lone))])
    cases.append(
        dict(
            pts=pts,
            stab=stab,
            title="Isolated unreliable cells",
            text="",
        )
    )
    # A whole cluster is unreliable.
    left = blob(rng, (-0.95, 0.0), (0.4, 0.48), 130)
    right = blob(rng, (0.95, 0.0), (0.4, 0.48), 130)
    pts = np.vstack([left, right])
    stab = np.concatenate(
        [rng.beta(30, 1.5, 130), np.clip(rng.normal(0.5, 0.09, 130), 0, 1)]
    )
    cases.append(
        dict(
            pts=pts,
            stab=stab,
            title="A whole cluster is unreliable",
            text="",
        )
    )
    return cases


# The note in header B. The darts, violins and "reliability" are explained in
# the caption, as for every other figure.
NOTE_B = "numbers refer to the scenarios in A"


def stability_key(cv: Canvas, y: float, cmap) -> None:
    """Per-sample stability color key, drawn inside the right end of a header bar."""
    x_right = cv.W - MARGIN - 0.1
    bar_w, bar_h = 1.1, 0.09
    cv.text(x_right, y + HEADER_H / 2, "more reliable", ha="right", va="center")
    bar_x = x_right - 0.72 - bar_w
    cbar = cv.axes(bar_x, y + (HEADER_H - bar_h) / 2, bar_w, bar_h)
    cbar.imshow(
        np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap, extent=(0, 1, 0, 1)
    )
    bare(cbar)
    for spine in cbar.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color(GUIDE_COLORS["axis"])
    cv.text(
        bar_x - 0.08,
        y + HEADER_H / 2,
        "per-sample reliability: less reliable",
        ha="right",
        va="center",
    )


def figure_guidance_clusters(
    *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """S Fig B: read each cluster, compare two candidates, read the samples."""
    rng = np.random.default_rng(SEED)
    full_w = PRINT_WIDTH_IN - 2 * MARGIN
    cmap = mpl.colormaps[SEQUENTIAL_CMAP_NAME].reversed()

    # Section A: a 2 x 2 grid of quadrant cards.
    label_w = 0.24
    quad_w = (full_w - label_w - GAP) / 2
    board = 0.78
    quad_text_w = quad_w - 0.24
    quad_hs = [
        0.32
        + board
        + 0.14
        + LINE_H * max(n_lines(q[7], quad_text_w) for q in QUADRANTS if q[0] == row)
        + 0.08
        for row in (0, 1)
    ]
    sec_a_h = HEADER_H + 0.07 + 0.22 + sum(quad_hs) + GAP

    # Section B: three outcome cards.
    col_w = (full_w - 2 * GAP) / 3
    out_scatter = 0.85
    outcomes = candidate_outcomes(rng)
    out_lines = max(n_lines(o["text"], col_w - 0.24) for o in outcomes)
    out_h = 0.32 + out_scatter + 0.06 + LINE_H * out_lines + 0.08
    sec_b_h = HEADER_H + 0.07 + out_h

    # Section C: three sample cards.
    samp_scatter = 0.74
    cases = sample_cases(rng)
    samp_lines = max(n_lines(c["text"], col_w - 0.24) for c in cases)
    samp_h = 0.32 + samp_scatter + 0.06 + LINE_H * samp_lines + 0.08
    sec_c_h = HEADER_H + 0.07 + samp_h

    height = sec_a_h + 0.12 + sec_b_h + 0.12 + sec_c_h

    with print_theme_context():
        cv = Canvas(height)

        # ---- A: cluster-level analysis ----
        y = 0.0
        cv.header(y, "A", "Cluster-level analysis")
        y += HEADER_H + 0.07
        x_grid = MARGIN + label_w

        def axis_label(x: float, y: float, label: str, rotation: float = 0) -> None:
            cv.text(
                x,
                y,
                label,
                size=AXIS_LABEL,
                color=GUIDE_COLORS["badge"],
                weight="bold",
                ha="center",
                va="center",
                rotation=rotation,
            )

        for col, label in enumerate(("generalizable", "not generalizable")):
            axis_label(x_grid + col * (quad_w + GAP) + quad_w / 2, y + 0.11, label)
        y += 0.22
        row_y = (y, y + quad_hs[0] + GAP)
        for row, label in enumerate(("stable", "not stable")):
            axis_label(
                MARGIN + label_w / 2, row_y[row] + quad_hs[row] / 2, label, rotation=90
            )
        for row, col, offset, spread, fs, fg, title, text in QUADRANTS:
            qx = x_grid + col * (quad_w + GAP)
            qy = row_y[row]
            cv.box(qx, qy, quad_w, quad_hs[row], GUIDE_COLORS["card"])
            cv.card_title(qx, qy, title, 2 * row + col + 1)
            dartboard(
                cv.axes(qx + 0.25, qy + 0.32, board, board),
                rng,
                offset=offset,
                spread=spread,
                color=cluster_colors(2)[1],
            )
            split_violins(
                cv.axes(
                    qx + 0.25 + board + 0.4,
                    qy + 0.32,
                    quad_w - board - 0.85,
                    board * 0.86,
                ),
                rng,
                fs,
                fg,
            )
            cv.para(qx + 0.12, qy + 0.32 + board + 0.14, quad_text_w, text)
        y += sum(quad_hs) + GAP + 0.12

        # ---- B: understanding stability and generalizability ----
        cv.header(y, "B", "Understanding stability and generalizability", note=NOTE_B)
        y += HEADER_H + 0.07
        colors = cluster_colors(3)
        board_s = 0.44
        for i, outcome in enumerate(outcomes):
            ox = MARGIN + i * (col_w + GAP)
            cv.box(ox, y, col_w, out_h, GUIDE_COLORS["card"])
            cv.text(
                ox + 0.12,
                y + 0.17,
                outcome["title"],
                size=TITLE,
                weight="bold",
                va="center",
            )
            ax = cv.axes(ox + 0.08, y + 0.32, col_w - board_s - 0.26, out_scatter)
            scatter_frame(ax)
            for pts, color in zip(outcome["parts"], colors):
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=6,
                    c=color,
                    edgecolors=FOREGROUND_COLOR,
                    linewidths=0.2,
                )
            for j, (offset, spread, scenario) in enumerate(outcome["boards"]):
                bx = ox + col_w - board_s - 0.1
                by = y + 0.3 + j * (board_s + 0.06)
                dartboard(
                    cv.axes(bx, by, board_s, board_s),
                    rng,
                    offset=offset,
                    spread=spread,
                    color=colors[j + 1],
                    n=7,
                )
                # The matching scenario in section A, at the board's top left.
                cv.badge(bx - 0.05, by + 0.05, scenario, on_top=True)
            cv.para(
                ox + 0.12, y + 0.32 + out_scatter + 0.06, col_w - 0.24, outcome["text"]
            )
        y += out_h + 0.12

        # ---- C: sample-level analysis ----
        cv.header(y, "C", "Sample-level analysis")
        stability_key(cv, y, cmap)
        y += HEADER_H + 0.07
        for i, case in enumerate(cases):
            sx = MARGIN + i * (col_w + GAP)
            cv.box(sx, y, col_w, samp_h, GUIDE_COLORS["card"])
            cv.card_title(sx, y, case["title"], len(QUADRANTS) + i + 1)
            ax = cv.axes(sx + 0.12, y + 0.32, col_w - 0.24, samp_scatter)
            scatter_frame(ax)
            order = np.argsort(-case["stab"])  # least stable drawn on top
            ax.scatter(
                case["pts"][order, 0],
                case["pts"][order, 1],
                s=8,
                c=cmap(case["stab"][order]),
                edgecolors=FOREGROUND_COLOR,
                linewidths=0.2,
            )
            cv.para(
                sx + 0.12, y + 0.32 + samp_scatter + 0.06, col_w - 0.24, case["text"]
            )

        if save:
            for ext in ("png", "pdf"):
                save_figure(
                    cv.fig,
                    figure_path(
                        f"guidance_clusters.{ext}", subdir=GUIDANCE_DIR, out_dir=out_dir
                    ),
                )
    return cv.fig
