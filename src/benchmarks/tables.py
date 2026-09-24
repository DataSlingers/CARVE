"""Write the supplementary tables as .tex fragments.

Under their manuscript names, to disk, rather than printed for copy-paste.
Each fragment is the caption-free float the manuscript's S2-S9 use, so it
replaces the committed table without hand-editing. The caption, with k*,
goes in the fragment's first line as a LaTeX comment that identifies the
file and never renders.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from ._registry import DIFFICULTY_AXIS, SCALING_TABLE_ROW_GROUPS, TABLE_ROW_GROUPS
from ._tables import write_tables

EXCLUDED_METRICS: frozenset[str] = frozenset(
    {
        "ari_average",
        "ari_average_1se",
        "ari_average_quant",
        "consensus_pac_stability",
        "consensus_ce_stability",
    }
)

TABLE_NAMES: dict[str, str] = {
    "gaussians": "S2_table",
    "t_dist": "S3_table",
    "t_dist_noise": "S4_table",
    "circles": "S5_table",
    "moons": "S6_table",
    "swiss_rolls": "S7_table",
    "gaussians_samples": "S8_table",
    "gaussians_dimensionality": "S9_table",
}

TABLE_CAPTIONS: dict[str, str] = {
    "gaussians": "Gaussian mixtures: ARI at the selected k and k-recovery by difficulty.",
    "t_dist": "t-distributed clusters: ARI at the selected k and k-recovery by difficulty.",
    "t_dist_noise": (
        "t-distributed clusters with nuisance dimensions: ARI at the selected k and "
        "k-recovery by difficulty."
    ),
    "circles": "RFF-embedded circles: ARI at the selected k and k-recovery by difficulty.",
    "moons": "RFF-embedded moons: ARI at the selected k and k-recovery by difficulty.",
    "swiss_rolls": (
        "RFF-embedded Swiss rolls: ARI at the selected k and k-recovery by difficulty."
    ),
    "gaussians_samples": (
        "Gaussian mixtures over sample size: ARI at the selected k and k-recovery."
    ),
    "gaussians_dimensionality": (
        "Gaussian mixtures over feature dimension: ARI at the selected k and k-recovery."
    ),
}


def table_metrics() -> tuple[str, ...]:
    """Metrics that appear in the manuscript tables, in the published order.

    Flattens TABLE_ROW_GROUPS, so the order a table renders in and the order
    a caller requests are one declaration rather than two that can drift.
    """
    return tuple(metric for group in TABLE_ROW_GROUPS for metric in group)


# render_paired_tex writes the caption into the fragment's first line, a
# LaTeX comment, through _comment_safe, which folds newlines into spaces and
# drops percent signs; nothing escapes it. The placeholder is substituted
# with "$k^\star = ...$" on the rendered file. It dates from
# render_grouped_tex, which passed the caption through _tex_escape into
# \caption{...} and would have mangled the math ("\$" for the dollar signs,
# "\textasciicircum{}" for "^"); under render_paired_tex the round trip
# writes exactly what passing the math directly would.
_K_STAR_PLACEHOLDER = "KSTARPLACEHOLDER"


def _axis_value_headers(frame: pd.DataFrame) -> dict[str, str]:
    """Each axis label's swept value as column header text: 1000, not 1000.0."""
    pairs = frame[["axis_label", "axis_value"]].drop_duplicates()
    return {
        str(label): str(int(value)) if float(value).is_integer() else str(value)
        for label, value in zip(pairs["axis_label"], pairs["axis_value"])
    }


def write_all_tables(
    results_by_scenario: Mapping[str, pd.DataFrame],
    out_dir: Path,
    *,
    metrics: Sequence[str] | None = None,
) -> list[Path]:
    """Write one .tex fragment per scenario under its manuscript table name.

    A difficulty scenario's table is headed easy/medium/hard and follows
    TABLE_ROW_GROUPS. A scaling scenario's is headed with its swept values
    and follows SCALING_TABLE_ROW_GROUPS, as the committed S8 and S9 are.
    """
    metrics = tuple(metrics) if metrics is not None else table_metrics()
    written: list[Path] = []

    for name, frame in results_by_scenario.items():
        if frame.empty:
            continue
        # "baseline_oracle" names the oracle_ari schema column, not a value
        # of metric_name, so it is always considered present rather than
        # filtered out by the metric_name membership check every other name
        # goes through.
        present = [
            m
            for m in metrics
            if m == "baseline_oracle" or (frame["metric_name"] == m).any()
        ]
        if not present:
            continue

        if frame["axis_name"].iloc[0] == DIFFICULTY_AXIS.name:
            groups, headers = TABLE_ROW_GROUPS, None
        else:
            groups, headers = SCALING_TABLE_ROW_GROUPS, _axis_value_headers(frame)

        k_star = int(frame["k_star"].iloc[0])
        table_name = TABLE_NAMES.get(name, f"{name}_table")
        path = write_tables(
            frame,
            Path(out_dir),
            table_name,
            caption=(f"{TABLE_CAPTIONS.get(name, name)} {_K_STAR_PLACEHOLDER}."),
            label=f"tab:{table_name.lower()}",
            metrics=present,
            groups=groups,
            column_headers=headers,
        )
        path.write_text(
            path.read_text().replace(_K_STAR_PLACEHOLDER, f"$k^\\star = {k_star}$")
        )
        written.append(path)
    return written
