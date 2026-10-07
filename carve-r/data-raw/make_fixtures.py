"""Write the Python reference outputs that the R tests compare against.

Run from code/ with the repository's virtual environment:

    .venv/bin/python carve-r/data-raw/make_fixtures.py

Each fixture is a JSON file in carve-r/tests/testthat/fixtures/. NaN is
written as null. Sample indices and labels are 0-based, as in Python; the
R tests add 1 to indices before using them.
"""

import json
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from carve import CARVE, SpectralClustering, _utils
from carve._accuracy import compute_generalizability_scores
from carve._consensus import (
    compute_consensus_matrix,
    compute_consensus_pac,
    consensus_anchor_block,
    stability_from_consensus,
    stability_from_runs_anchored,
)
from carve._grids import estimate_knn_gamma
from carve._selection import select_best_k, select_best_row_by_rule
from carve._sweep import format_method_label, resolve_sweep
from carve._utils import (
    _summarize_ari_scores,
    align_cluster_labels,
    noise_mask,
    resolve_core_budget,
    scale_neighbor_count,
)
from carve._pipeline import PipelineSpec, PipelineStep
from carve.cluster import build_knn_graph

OUT = Path(__file__).resolve().parents[1] / "tests" / "testthat" / "fixtures"


def clean(x):
    """Convert numpy values to JSON types, with NaN as None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    return x


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(clean(obj), f, indent=1, allow_nan=False)
        f.write("\n")


def consensus():
    rng = np.random.default_rng(0)
    n = 12
    runs = []
    for _ in range(6):
        # Sample 11 is never drawn, so its row of the matrix is NaN.
        idx = rng.choice(n - 1, size=8, replace=False)
        labels = rng.integers(0, 3, size=8)
        runs.append((idx, labels))
    M = compute_consensus_matrix(n, runs)
    gini, ce = stability_from_consensus(M)
    write(
        "consensus",
        {
            "n_samples": n,
            "runs": [{"indices": i, "labels": lab} for i, lab in runs],
            "consensus": M,
            "gini": gini,
            "ce": ce,
            "pac_005": compute_consensus_pac(M),
            "pac_010": compute_consensus_pac(M, tau=0.1),
        },
    )


def ari():
    rng = np.random.default_rng(1)
    cases = [
        ([0, 0, 1, 1], [0, 0, 1, 1]),
        ([0, 0, 1, 1], [1, 1, 0, 0]),
        ([0, 0, 0, 0], [0, 1, 2, 3]),
        ([0, 0, 1, 1], [0, 1, 0, 1]),
        ([0, 0, 0], [0, 0, 0]),
        ([0, 1, 2, 3], [0, 1, 2, 3]),
    ]
    for _ in range(5):
        cases.append((rng.integers(0, 4, size=40), rng.integers(0, 3, size=40)))
    a = rng.integers(0, 4, size=60)
    b = a.copy()
    flip = rng.choice(60, size=10, replace=False)
    b[flip] = rng.integers(0, 4, size=10)
    cases.append((a, b))
    write(
        "ari",
        {"cases": [{"a": a, "b": b, "ari": adjusted_rand_score(a, b)} for a, b in cases]},
    )


def align():
    # Every case has a unique best matching, so scipy and clue must agree.
    ref = np.repeat([0, 1, 2], 10)
    noisy = (ref + 2) % 3
    noisy[[0, 25]] = [1, 2]
    cases = [
        ("permuted", ref, (ref + 1) % 3),
        ("noisy", ref, noisy),
        ("more_clusters", ref, (np.repeat([0, 1, 2, 3], [10, 10, 6, 4]) + 1) % 4),
        ("fewer_clusters", ref, np.repeat([0, 1], [18, 12])),
        ("sparse_reference_values", np.repeat([3, 7, 9], 10), np.repeat([2, 0, 1], 10)),
    ]
    write(
        "align",
        {
            "cases": [
                {"name": n, "reference": r, "labels": lab, "aligned": align_cluster_labels(r, lab)}
                for n, r, lab in cases
            ]
        },
    )


def accuracy():
    rng = np.random.default_rng(2)
    n = 20
    runs = []
    for _ in range(5):
        # Sample 19 is never held out, so its score is 0.
        idx = rng.choice(n - 1, size=12, replace=False)
        true = rng.permutation(np.repeat([0, 1, 2], 4))
        pred = rng.permutation(3)[true]
        j = int(rng.integers(0, 12))
        pred[j] = (pred[j] + 1) % 3
        runs.append((idx, true, pred))
    write(
        "accuracy",
        {
            "n_samples": n,
            "runs": [{"indices": i, "true": t, "predicted": p} for i, t, p in runs],
            "scores": compute_generalizability_scores(n, runs),
        },
    )


def selection():
    rng = np.random.default_rng(3)
    rows = []
    for m, estimator in enumerate(["KMeans", "AgglomerativeClustering"]):
        for rank, k in enumerate(range(2, 7)):
            row = {
                "config_id": 5 * m + rank,
                "method_id": f"m{m}",
                "method_label": estimator,
                "estimator": estimator,
                "n_clusters": k,
                "sweep_param": "n_clusters",
                "sweep_value": k,
                "sweep_rank": rank,
                "n_clusters_observed": float(k),
                "n_clusters_observed_se": 0.0,
                "noise_fraction": 0.0,
            }
            for prefix in ("ari_stability", "ari_generalizability", "ari_average"):
                mean = rng.uniform(0.6, 0.95)
                row[prefix] = mean
                row[f"{prefix}_se"] = rng.uniform(0.005, 0.05)
                row[f"{prefix}_upper"] = mean + rng.uniform(0.01, 0.08)
                row[f"{prefix}_lower"] = mean - rng.uniform(0.01, 0.08)
            for col in (
                "consensus_pac_stability",
                "consensus_gini_stability",
                "consensus_ce_stability",
                "accuracy_generalizability",
            ):
                row[col] = rng.uniform(0.5, 1.0)
            rows.append(row)
    # Shuffled, so table order differs from config_id order.
    table = pd.DataFrame(rows).sample(frac=1, random_state=4).reset_index(drop=True)
    expected = []
    for measure in ("stability", "generalizability", "average", "pac", "gini", "ce", "accuracy"):
        for rule in ("max", "1se", "quantile"):
            for not_two in (False, True):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    row = select_best_row_by_rule(
                        table, measure=measure, rule=rule, not_two=not_two
                    )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    k = select_best_k(table, measure=measure, rule=rule, not_two=not_two)
                expected.append(
                    {
                        "measure": measure,
                        "rule": rule,
                        "not_two": not_two,
                        "config_id": int(row["config_id"]),
                        "k": int(k),
                        "warning": str(caught[0].message) if caught else None,
                    }
                )
    write("selection", {"table": table.to_dict(orient="list"), "expected": expected})


def core_budget():
    cases = []
    for cores in (1, 4, 11):
        for n_jobs in (None, 1, 2, 4, -1, -2, 16):
            for n_resamples in (3, 100):
                with mock.patch.object(_utils, "cpu_count", return_value=cores):
                    outer, inner = resolve_core_budget(n_jobs, n_resamples=n_resamples)
                cases.append(
                    {
                        "cores": cores,
                        "n_jobs": n_jobs,
                        "n_resamples": n_resamples,
                        "outer": outer,
                        "inner": inner,
                    }
                )
    write("core_budget", {"cases": cases})


def summarize_ari():
    arrays = [
        [0.8, 0.9, 1.0, 0.7],
        [0.5],
        [np.nan, 0.6, 0.8],
        [np.nan, np.nan],
        [0.3, 0.3, 0.3],
    ]
    cases = []
    for a in arrays:
        mean, se, upper, lower = _summarize_ari_scores(a, len(a))
        cases.append({"x": a, "mean": mean, "se": se, "upper": upper, "lower": lower})
    write("summarize_ari", {"cases": cases})


class _Neighbors:
    def __init__(self, n_neighbors=7, n_clusters=2):
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters


def scale_neighbor():
    cases = []
    for params, n_fit, n_full in [
        ({}, 618, 1000),
        ({"n_neighbors": 15}, 618, 1000),
        ({"n_neighbors": 3}, 10, 1000),
        ({"n_neighbors": 15}, 1000, 1000),
        ({"n_neighbors": 9}, 500, 1000),
        ({"n_neighbors": 15}, 1, 1000),
    ]:
        scaled = scale_neighbor_count(_Neighbors, params, n_fit=n_fit, n_full=n_full)
        cases.append(
            {
                "params": params,
                "n_fit": n_fit,
                "n_full": n_full,
                "n_neighbors": scaled.get("n_neighbors"),
            }
        )
    write("scale_neighbor", {"cases": cases})


def sweep():
    labels = [
        ("KMeans", {"n_clusters": 3}, "n_clusters"),
        ("AgglomerativeClustering", {"n_clusters": 3, "linkage": "ward"}, "n_clusters"),
        ("SpectralClustering", {"n_clusters": 3, "affinity": "rbf", "gamma": 0.5}, "n_clusters"),
        ("SpectralClustering", {"n_clusters": 3, "gamma": 1.0, "n_neighbors": 7}, "n_clusters"),
        ("Custom", {"tol": 1e-5, "scale": True, "b": 1234.5, "resolution": 0.25}, "resolution"),
    ]
    grid = {"n_clusters": [2, 3], "linkage": ["ward", "average"], "affinity": ["x"]}
    specs = [
        resolve_sweep(n_clusters=np.array([5, 2, 3])),
        resolve_sweep(resolution=[0.5, 0.1, 1.0]),
        resolve_sweep(sweep="min_cluster_size", sweep_values=[10, 5, 20]),
    ]
    write(
        "sweep",
        {
            "labels": [
                {"estimator": e, "params": p, "sweep_param": s, "label": format_method_label(e, p, s)}
                for e, p, s in labels
            ],
            "grid": grid,
            "grid_order": list(ParameterGrid(grid)),
            "ranks": [{"param": s.param, "values": s.values, "ranks": s.ranks()} for s in specs],
        },
    )


def kmeans():
    X, _ = make_blobs(n_samples=60, centers=3, cluster_std=1.5, random_state=5)
    starts = {
        "data_points": X[[0, 1, 2]],
        # No point is closer to the third center than to the other two, so
        # Lloyd's first step leaves it empty and relocates it.
        "empty_cluster": np.vstack([X[0], X[1], [100.0, 100.0]]),
    }
    cases = []
    for name, init in starts.items():
        km = KMeans(
            n_clusters=3, init=init, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd"
        ).fit(X)
        cases.append(
            {
                "name": name,
                "init": init,
                "labels": km.labels_,
                "centers": km.cluster_centers_,
                "n_iter": km.n_iter_,
                "inertia": km.inertia_,
            }
        )
    write("kmeans", {"X": X, "cases": cases})


def agglomerative():
    # This seed gives four pairwise different partitions, one per linkage, so
    # swapping or misnaming a linkage in the R port changes the result.
    X, _ = make_blobs(n_samples=40, centers=4, cluster_std=2.0, random_state=11)
    labels = {
        linkage: AgglomerativeClustering(n_clusters=4, linkage=linkage).fit_predict(X)
        for linkage in ("ward", "average", "single", "complete")
    }
    write("agglomerative", {"X": X, "labels": labels})


def spectral():
    X, _ = make_blobs(n_samples=40, centers=3, cluster_std=1.0, random_state=7)
    out = {
        "X": X,
        "scaled": StandardScaler().fit_transform(X),
        "knn_gamma": estimate_knn_gamma(X),
    }
    settings = {
        "self_tuning": {"affinity": "self_tuning"},
        "rbf": {"affinity": "rbf"},
        "rbf_gamma": {"affinity": "rbf", "gamma": 0.3},
        "knn": {"affinity": "knn"},
    }
    for name, kwargs in settings.items():
        sc = SpectralClustering(n_clusters=3, random_state=0, **kwargs).fit(X)
        W = sc.affinity_.toarray() if sparse.issparse(sc.affinity_) else sc.affinity_
        out[name] = {"W": W, "gamma": sc.gamma_, "evals": sc.evals_}
    write("spectral", out)


def consensus_cut():
    X, _ = make_blobs(n_samples=45, centers=3, cluster_std=2.5, random_state=8)
    model = CARVE(
        n_clusters=np.array([2, 3, 4]),
        n_resamples=5,
        estimator_param_grids=[(KMeans, {"n_clusters": [2, 3, 4]})],
        random_state=0,
    ).fit(X)
    results = model.estimator_results_
    cases = []
    for k in (2, 3, 4):
        config_id = int(results.loc[results["n_clusters"] == k, "config_id"].iloc[0])
        cases.append(
            {
                "k": k,
                "consensus": model.consensus_matrices_[config_id],
                "labels": model.get_labels(k=k),
            }
        )
    write("consensus_cut", {"cases": cases})


def anchored():
    rng = np.random.default_rng(1)
    n = 15
    runs = []
    for _ in range(6):
        # Sample 14 is never drawn, so its scores are NaN. The indices stay
        # unsorted, as the runner passes them.
        idx = rng.choice(n - 1, size=9, replace=False)
        labels = rng.integers(0, 3, size=9)
        runs.append((idx, labels))
    anchors = np.array([0, 2, 5, 9, 13, 14])
    gini, ce = stability_from_runs_anchored(n, runs, anchors, chunk_size=4)
    write(
        "anchored",
        {
            "n": n,
            "runs": [{"indices": idx, "labels": labels} for idx, labels in runs],
            "anchors": anchors,
            "block": consensus_anchor_block(n, runs, anchors),
            "gini": gini,
            "ce": ce,
        },
    )


def noise_masks():
    ties = np.ones(90)
    ties[[3, 41, 84]] = 0.5
    ties[[10, 25, 35, 50, 65, 80]] = 0.9
    with_nan = ties.copy()
    with_nan[[20, 60]] = np.nan
    spread = np.random.default_rng(2).uniform(0.0, 1.0, 40)
    # Every score is within 0.03 of 1, so the margin flags nothing even
    # though some scores lie below the quantile.
    near_one = np.linspace(0.97, 1.0, 40)
    flat = np.full(30, 0.8)
    cases = []
    for name, scores, quantile in [
        ("ties", ties, 0.05),
        ("with_nan", with_nan, 0.05),
        ("spread", spread, 0.1),
        ("spread_quarter", spread, 0.25),
        ("near_one", near_one, 0.1),
        ("flat", flat, 0.05),
    ]:
        cut = noise_mask(scores, quantile=quantile)
        cases.append(
            {
                "name": name,
                "scores": scores,
                "quantile": quantile,
                "mask": cut.mask,
                "n_target": cut.n_target,
                "cutoff": cut.cutoff,
                "median": cut.median,
            }
        )
    write("noise_mask", {"cases": cases})


def knn_graph():
    X = np.random.RandomState(3).randn(25, 2)
    cases = []
    for weighting, n_neighbors in [("connectivity", 4), ("jaccard", 4), ("jaccard", 7)]:
        graph = build_knn_graph(X, n_neighbors=n_neighbors, weighting=weighting)
        cases.append(
            {
                "weighting": weighting,
                "n_neighbors": n_neighbors,
                "edges": np.array(graph.get_edgelist()),
                "weights": graph.es["weight"],
            }
        )
    write("knn_graph", {"X": X, "cases": cases})


def hdbscan():
    # Two close groups and a far one. On these data scikit-learn's eom and
    # leaf selections differ. On these data dbscan::hdbscan() orders its tied
    # merges as scikit-learn does, so the package's labels match both
    # selections (checked with dbscan 1.2.7 when this fixture was written).
    # Other data can differ where distances tie; the HDBSCAN help page
    # explains why.
    X, _ = make_blobs(
        n_samples=[40, 40, 40],
        centers=[[0, 0], [2.0, 0], [8, 0]],
        cluster_std=[0.4, 0.4, 0.8],
        random_state=0,
    )
    labels = {}
    for m in (5, 10):
        for method in ("eom", "leaf"):
            labels[f"{method}_{m}"] = HDBSCAN(
                min_cluster_size=m, cluster_selection_method=method, copy=True
            ).fit_predict(X)
    write("hdbscan", {"X": X, "labels": labels})


def hdbscan_split():
    # Four groups of 20. At min_cluster_size 15 a cluster of the tree splits
    # into two parts both smaller than 15. scikit-learn ends the cluster
    # there and selects two clusters by eom; dbscan::hdbscan()'s own
    # selection keeps the cluster alive and selects three (dbscan 1.2.7).
    X, _ = make_blobs(
        n_samples=80, n_features=2, centers=4, cluster_std=1.0, random_state=13
    )
    labels = {
        method: HDBSCAN(
            min_cluster_size=15, cluster_selection_method=method, copy=True
        ).fit_predict(X)
        for method in ("eom", "leaf")
    }
    write("hdbscan_split", {"X": X, "labels": labels})


def pipeline_labels():
    # A user-supplied name renders like a class name, so each step is built
    # with a name and a stand-in class; the label depends only on the name
    # and the sampled values.
    steps = [
        ("identity", {}),
        ("log1p", {}),
        ("StandardScaler", {}),
        ("PCA", {"n_components": 10}),
        ("TSNE", {"n_components": 2, "perplexity": 30}),
        ("TSNE", {"perplexity": 12.5, "n_components": 2}),
        ("UMAP", {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1}),
        ("Scaled", {"factor": 0.000123}),
        ("Flag", {"center": True}),
    ]
    cases = [
        {
            "name": name,
            "params": params,
            "label": PipelineStep(cls=object, params=params, name=name).label,
        }
        for name, params in steps
    ]
    spec = PipelineSpec(
        normalization=PipelineStep(cls=object, params={}, name="identity"),
        dim_reduction=PipelineStep(cls=object, params={"n_components": 10}, name="PCA"),
    )
    write("pipeline_labels", {"steps": cases, "spec": {"label": spec.label}})


if __name__ == "__main__":
    for make in (
        consensus,
        ari,
        align,
        accuracy,
        selection,
        core_budget,
        summarize_ari,
        scale_neighbor,
        sweep,
        kmeans,
        agglomerative,
        spectral,
        consensus_cut,
        anchored,
        noise_masks,
        knn_graph,
        hdbscan,
        hdbscan_split,
        pipeline_labels,
    ):
        make()
        print(f"wrote {make.__name__}.json")
