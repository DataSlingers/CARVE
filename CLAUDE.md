# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CARVE** (Cluster Analysis with Resampling for Validation and Exploration) is a Python library for stability- and generalizability-based clustering validation. It evaluates clustering robustness by:
1. Repeatedly subsampling input data
2. Running clustering algorithms on each subsample
3. Building consensus matrices from aligned clustering results
4. Computing stability (intra-subsample consistency) and generalizability (held-out predictive performance) metrics

The primary use case is validating clustering methods on high-dimensional biological data (scRNA-seq, genomics).

## Environment Setup

Python 3.12 is required (pinned in `pyproject.toml`).

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -e .           # standard install
pip install -e ".[dev]"    # includes ruff + pytest
```

## Common Commands

```bash
pytest                     # run all tests
pytest path/to/test.py::test_name  # run a single test
ruff check src/            # lint
ruff format src/           # format
jupyter lab                # launch notebooks
```

## Architecture

### Core Validation Pipeline (`src/carve/`)

The main entry point is `CARVE` in `api.py`, a scikit-learn–compatible estimator:

```python
carve = CARVE(n_clusters=10, n_resamples=100, subsample_ratio=0.8)
carve.fit(X)
labels = carve.get_labels(measure="stability", rule="1se")
```

**Data flow:**
- `api.py` → `_runner.py` (`run_validation`) → per-estimator loop over `_grids.py` defaults
- Each iteration: subsample → optional preprocessing (`_pipeline.py`) → cluster → align labels → compute ARI
- After all iterations: `_consensus.py` builds consensus matrices; `_misclassification.py` scores generalizability
- `_selection.py` applies selection rules (max, 1se, quantile) to pick best k/estimator
- `_plotting.py` renders metric-over-k and cluster summary figures; `_output.py` handles console formatting

### Key Files

| File | Role |
|------|------|
| `src/carve/api.py` | `CARVE` dataclass; `fit()`, `get_labels()`, `get_k()`, `save()`/`load()`, plotting methods |
| `src/carve/_runner.py` | `run_validation()` — core resampling loop |
| `src/carve/config.py` | `ValidatorConfig` frozen dataclass (legacy config path) |
| `src/carve/_grids.py` | Default estimator/parameter grids (KMeans, Agglomerative, Spectral) |
| `src/carve/_pipeline.py` | Builds sklearn preprocessing pipelines (normalization + dim reduction) |
| `src/carve/_consensus.py` | Consensus matrix computation; PAC, Gini, CE stability metrics |
| `src/carve/_misclassification.py` | RandomForest-based generalizability scoring |
| `src/carve/_selection.py` | `select_best_k()`, `select_best_estimator()`, rule-based selection |
| `src/carve/cluster.py` | `SpectralClusteringCARVE` — custom spectral clustering variant |
| `src/carve/sim/` | Synthetic data generation submodule (distributions, covariance, noise, outliers) |

### Metrics

- **Stability:** `ari_stability`, `consensus_pac_stability`, `consensus_gini_stability`, `consensus_ce_stability`
- **Generalizability:** `ari_generalizability`, `misclassification_generalizability`

All metrics are stored in `carve.estimator_results_` (a DataFrame) after `fit()`.

### Preprocessing Options

Normalization: identity, StandardScaler, log1p
Dimensionality reduction: identity, PCA, t-SNE (2D/3D), UMAP
Pass `randomize_preprocessing=True` to `fit()` to sample a random pipeline per resample iteration.

### Persistence

```python
carve.save("results.carve")
carve = CARVE.load("results.carve")
```

## Notebooks

- `notebooks/Benchmarking.ipynb` — main benchmarking analysis
- `notebooks/case_studies/` — per-dataset analyses (Samusik, Zheng PBMC14K, Klein, Tasic)
- `notebooks/legacy/` — archived development notebooks

## Legacy Directories

Any directory named `legacy/` (e.g. `src/carve/legacy/`, `notebooks/legacy/`) contains out-of-date code or notebooks and should not be used as a reference for current patterns.

## Working Directory Note

The repository has two working directories used by Claude Code:
- `code/` — the Python package (primary source)
- `_claude_work/` — planning templates and AI workflow files (not source code)
