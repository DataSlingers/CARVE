"""Write the supplementary tables as .tex fragments.

Under their manuscript names, to disk, rather than printed for copy-paste.
The committed versions have the generated k-star header row stripped by hand,
which is the drift this ends.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from ._registry import CARVE_METRICS_ALL, CVI_METRICS
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
        "Gaussian mixtures over embedding dimension: ARI at the selected k and k-recovery."
    ),
}


def table_metrics() -> tuple[str, ...]:
    """Metrics that appear in the manuscript tables, in a stable order.

    "baseline_oracle" leads the tuple so it renders as the tables' first
    data row, matching every published S2-S9 table.
    """
    return (
        "baseline_oracle",
        *(m for m in (*CARVE_METRICS_ALL, *CVI_METRICS) if m not in EXCLUDED_METRICS),
    )


# render_grouped_tex escapes the whole caption through _tex_escape before
# it emits \caption{...}, so LaTeX math handed to write_tables' caption=
# verbatim -- "$k^\star = 5$" -- comes back mangled: "\$" for the dollar
# signs, "\textasciicircum{}" for "^", "\textbackslash{}" for the "\" that
# starts \star. _tex_escape has no notion of math mode; it treats every
# backslash and special character as literal text to protect, which is
# correct for the plain-prose captions in TABLE_CAPTIONS but wrong for this
# one deliberately-inserted math snippet.
#
# The placeholder below is plain letters, none of which _tex_escape touches,
# so it survives the caption's escaping pass unchanged. Substituting the
# real "$k^\star = ...$" back in afterwards, on the rendered file, is what
# keeps that k-star header a generated, unescaped math header rather than
# the garbled text escaping it wholesale would produce.
_K_STAR_PLACEHOLDER = "KSTARPLACEHOLDER"


def write_all_tables(
    results_by_scenario: Mapping[str, pd.DataFrame],
    out_dir: Path,
    *,
    metrics: Sequence[str] | None = None,
) -> list[Path]:
    """Write one .tex fragment per scenario under its manuscript table name."""
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

        k_star = int(frame["k_star"].iloc[0])
        table_name = TABLE_NAMES.get(name, f"{name}_table")
        path = write_tables(
            frame,
            Path(out_dir),
            table_name,
            caption=(f"{TABLE_CAPTIONS.get(name, name)} {_K_STAR_PLACEHOLDER}."),
            label=f"tab:{table_name.lower()}",
            metrics=present,
        )
        path.write_text(
            path.read_text().replace(_K_STAR_PLACEHOLDER, f"$k^\\star = {k_star}$")
        )
        written.append(path)
    return written
