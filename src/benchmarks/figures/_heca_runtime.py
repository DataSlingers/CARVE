"""Where the hECA fit's CPU time went, and its memory over time.

Left: the whole fit's core-hours by component, as one stacked bar. Middle:
the graph and Leiden components per setting. Right: the resident memory of
the fit process and its workers, as sampled during the run.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .._theme import FOREGROUND_COLOR, cluster_colors, save_figure, theme_context
from ._paths import CASE_STUDY_DIR, figure_path


def figure_heca_runtime(
    components: pd.DataFrame,
    memory: pd.DataFrame,
    *,
    save: bool = True,
    out_dir: Path | None = None,
    save_name: str = "heca_runtime.png",
) -> Figure:
    """Draw the component breakdown and the memory curve.

    components is _heca_report.component_table's frame; memory is the fit's
    memory.csv.
    """
    if components.empty:
        raise ValueError("figure_heca_runtime needs a non-empty component table.")

    totals = components.groupby("component", sort=False)["core_hours"].sum()
    palette = dict(zip(totals.index, cluster_colors(len(totals)), strict=True))

    with theme_context():
        fig, (total_ax, setting_ax, memory_ax) = plt.subplots(1, 3, figsize=(15.0, 3.8))

        left = 0.0
        for name, hours in totals.items():
            total_ax.barh(0, hours, left=left, color=palette[name], label=name)
            left += hours
        total_ax.set_yticks([])
        total_ax.set_xlabel("Core-hours, whole fit")
        total_ax.legend(
            fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3
        )

        per_setting = components[components["setting"] != "all"]
        settings = list(dict.fromkeys(per_setting["setting"]))
        bottoms = [0.0] * len(settings)
        for name in dict.fromkeys(per_setting["component"]):
            rows = per_setting[per_setting["component"] == name].set_index("setting")
            heights = [float(rows["core_hours"].get(s, 0.0)) for s in settings]
            setting_ax.bar(
                settings, heights, bottom=bottoms, color=palette[name], label=name
            )
            bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]
        setting_ax.set_ylabel("Core-hours")
        setting_ax.set_title("Graph and Leiden, by setting")

        hours = memory["elapsed_s"] / 3600
        total_gb = (memory["own_rss_bytes"] + memory["children_rss_bytes"]) / 1e9
        memory_ax.plot(hours, total_gb, color=FOREGROUND_COLOR)
        memory_ax.set_xlabel("Hours since the fit started")
        memory_ax.set_ylabel("Resident memory (GB)")

        fig.tight_layout()
        if save:
            save_figure(
                fig, figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir)
            )
    return fig
