"""Write the supplementary tables as .tex fragments, and show them in a notebook.

Each fragment is written under its manuscript name as the caption-free float
the supplementary tables use. The caption, with k*, goes in the fragment's
first line as a LaTeX comment. manuscript_tables builds the same tables from
the same cells for display.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from ._registry import DIFFICULTY_AXIS, SCALING_TABLE_ROW_GROUPS, TABLE_ROW_GROUPS
from ._tables import DisplayTable, paired_table, summarize, write_tables

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
    "gaussians_samples": "S11_table",
    "gaussians_dimensionality": "S12_table",
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

    Flattens TABLE_ROW_GROUPS.
    """
    return tuple(metric for group in TABLE_ROW_GROUPS for metric in group)


# render_paired_tex writes the caption into the fragment's first line, a
# LaTeX comment, through _comment_safe, which folds newlines into spaces and
# drops percent signs. The placeholder is replaced with "$k^\star = ...$" in
# the rendered file.
_K_STAR_PLACEHOLDER = "KSTARPLACEHOLDER"


def _axis_value_headers(frame: pd.DataFrame) -> dict[str, str]:
    """Each axis label's swept value as column header text: 1000, not 1000.0."""
    pairs = frame[["axis_label", "axis_value"]].drop_duplicates()
    return {
        str(label): str(int(value)) if float(value).is_integer() else str(value)
        for label, value in zip(pairs["axis_label"], pairs["axis_value"])
    }


def _table_specs(
    results_by_scenario: Mapping[str, pd.DataFrame], metrics: Sequence[str] | None
) -> list[dict]:
    """Per scenario with rows, what its table holds, in the order given."""
    metrics = tuple(metrics) if metrics is not None else table_metrics()
    specs = []
    for name, frame in results_by_scenario.items():
        if frame.empty:
            continue
        # "baseline_oracle" names the oracle_ari column, not a metric_name
        # value, so it is always present.
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
        specs.append(
            {
                "name": name,
                "frame": frame,
                "table_name": TABLE_NAMES.get(name, f"{name}_table"),
                "k_star": int(frame["k_star"].iloc[0]),
                "metrics": present,
                "groups": groups,
                "headers": headers,
            }
        )
    return specs


def _number(table_name: str) -> int:
    """S2_table -> 2; a name without a number sorts last."""
    head = table_name.split("_")[0]
    return int(head[1:]) if head[:1] == "S" and head[1:].isdigit() else 10**6


def manuscript_tables(
    results_by_scenario: Mapping[str, pd.DataFrame],
    *,
    metrics: Sequence[str] | None = None,
) -> list[DisplayTable]:
    """The tables write_all_tables writes, for a notebook, in the order of
    their manuscript numbers."""
    specs = sorted(
        _table_specs(results_by_scenario, metrics),
        key=lambda s: _number(s["table_name"]),
    )
    tables = []
    for spec in specs:
        number = spec["table_name"].replace("_table", " Table")
        caption = TABLE_CAPTIONS.get(spec["name"], spec["name"])
        tables.append(
            paired_table(
                summarize(spec["frame"], metrics=spec["metrics"]),
                caption=f"{number}. {caption} k* = {spec['k_star']}.",
                groups=spec["groups"],
                column_headers=spec["headers"],
            )
        )
    return tables


#: Row labels of the runtime table, by the runtime artifact's column.
RUNTIME_CRITERIA: dict[str, str] = {
    "t_stability_s": "CARVE stability",
    "t_generalizability_s": "CARVE generalizability",
}

#: How the runtime table names a swept axis.
_AXIS_SYMBOLS: dict[str, str] = {"n_total": "n"}


def runtime_seconds_table(runtimes: Mapping[str, pd.DataFrame]) -> DisplayTable:
    """S13 Table: mean (SD) seconds of one CARVE.fit() per criterion, over the
    data sets at each of a scaling scenario's three axis values.

    runtimes maps a scaling scenario to its read_runtimes frame. Settings
    follow their own tables' numbers, so sample size comes before feature
    dimension.
    """
    index, rows = [], []
    for name in sorted(runtimes, key=lambda n: _number(TABLE_NAMES.get(n, n))):
        frame = runtimes[name]
        values = sorted(frame["axis_value"].unique())
        if len(values) != 3:
            raise ValueError(
                f"{name} has {len(values)} axis values; the table needs 3."
            )
        axis_name = frame["axis_name"].iloc[0]
        axis = _AXIS_SYMBOLS.get(axis_name, axis_name)
        title = TABLE_CAPTIONS.get(name, name).split(":")[0]
        setting = f"{title} ({axis} = {' / '.join(f'{v:g}' for v in values)})"
        for column, criterion in RUNTIME_CRITERIA.items():
            stats = frame.groupby("axis_value")[column].agg(["mean", "std"]).loc[values]
            index.append((setting, criterion))
            rows.append(
                [f"{m:.2f} ({sd:.2f})" for m, sd in zip(stats["mean"], stats["std"])]
            )
    cells = pd.DataFrame(
        rows, index=pd.MultiIndex.from_tuples(index), columns=["low", "mid", "high"]
    )
    return DisplayTable(
        caption=(
            "S13 Table. CARVE runtime (seconds) across the Gaussian-mixture scaling "
            "experiments: mean (SD) over data sets of one CARVE.fit() call per criterion."
        ),
        cells=cells,
    )


def write_all_tables(
    results_by_scenario: Mapping[str, pd.DataFrame],
    out_dir: Path,
    *,
    metrics: Sequence[str] | None = None,
) -> list[Path]:
    """Write one .tex fragment per scenario under its manuscript table name.

    A difficulty scenario's table is headed easy/medium/hard and follows
    TABLE_ROW_GROUPS. A scaling scenario's is headed with its swept values
    and follows SCALING_TABLE_ROW_GROUPS.
    """
    written: list[Path] = []
    for spec in _table_specs(results_by_scenario, metrics):
        name, table_name = spec["name"], spec["table_name"]
        path = write_tables(
            spec["frame"],
            Path(out_dir),
            table_name,
            caption=(f"{TABLE_CAPTIONS.get(name, name)} {_K_STAR_PLACEHOLDER}."),
            label=f"tab:{table_name.lower()}",
            metrics=spec["metrics"],
            groups=spec["groups"],
            column_headers=spec["headers"],
        )
        path.write_text(
            path.read_text().replace(
                _K_STAR_PLACEHOLDER, f"$k^\\star = {spec['k_star']}$"
            )
        )
        written.append(path)
    return written
