"""Fig 6: the Levine 32-dimension case study.

Panel F is an ARI lollipop against the reported labels. Panels A to C use the
precomputed t-SNE embedding carried on CompositeInputs.Z and 8-point markers,
because the dataset has an order of magnitude more cells than Klein.
"""

from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import adjusted_rand_score

from .._panels import ari_lollipop
from ._case_study import CompositeInputs, composite_figure

MARKER_SIZE = 8.0
AXIS_LABELS = ("t-SNE 1", "t-SNE 2")


def _ari_table(inputs: CompositeInputs) -> pd.DataFrame:
    """ARI of each selection against the reported labels."""
    rows = [
        {
            "method": "CARVE",
            "metric": "ari_stability_1se",
            "ari": float(adjusted_rand_score(inputs.y, inputs.carve_labels)),
            "k": int(len(set(inputs.carve_labels.tolist()))),
        }
    ]
    for _, row in inputs.best_df.iterrows():
        rows.append(
            {
                "method": str(row["metric"]).replace("_", " ").title(),
                "metric": str(row["metric"]),
                "ari": float(row["ari"]),
                "k": int(row["k"]),
            }
        )
    return pd.DataFrame(rows)


def _ari_panel(ax: Axes, inputs: CompositeInputs) -> Axes:
    return ari_lollipop(
        ax, _ari_table(inputs), title="Agreement with Reported Labels (ARI)"
    )


def figure_levine_results(
    inputs: CompositeInputs, *, save: bool = True, out_dir: Path | None = None
) -> Figure:
    """Build Fig 6."""
    return composite_figure(
        inputs,
        bottom_panel=_ari_panel,
        marker_size=MARKER_SIZE,
        axis_labels=AXIS_LABELS,
        save_name="levine_results.png",
        save=save,
        out_dir=out_dir,
    )
