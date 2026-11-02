# Notebooks

Paths below are relative to the repository root. Install the package from there with
`pip install -e ".[benchmarks,notebooks]"`; individual notebooks name any further extras or R
packages they need.

## Tutorials

| Notebook | Contents |
|---|---|
| [Tutorial.ipynb](Tutorial.ipynb) | The k-based mode: fitting CARVE, the criteria and selection rules, the plots, saving and loading |
| [Resolution_Tutorial.ipynb](Resolution_Tutorial.ipynb) | Sweeps other than k on PBMC 3k: Leiden and Louvain resolution, HDBSCAN's `min_samples`, custom grids |
| [R_Tutorial.Rmd](R_Tutorial.Rmd) | The R package, rendered as `R_Tutorial.html` |

The tutorials run in a few minutes and use no data beyond what scikit-learn and scanpy provide.

## Manuscript notebooks

These notebooks produce the figures, tables and numbers of the manuscript. Figures are written
under `vis/` and tables under `vis/tables/`, `vis/case_studies/` or `results/tables/`; copying
them into the manuscript is a manual step. Case-study fits are cached in
`case_studies/carve_state_saves/`, so a second run reads them instead of fitting again.

| Notebook | Manuscript items | Inputs | Cost |
|---|---|---|---|
| [Benchmarking.ipynb](Benchmarking.ipynb) | Fig 4, S1 Fig, S4 Fig, S5 Fig, S2 to S7 Table, S11 Table, S12 Table | Run artifacts in `results/runs/` from `python -m benchmarks.run --all` | Reads the artifacts only |
| [Resampling_Ablation.ipynb](Resampling_Ablation.ipynb) | S2 Fig, S3 Fig, S8 to S10 Table | A run from `python -m benchmarks.run --ablation rho_b` | Reads the run only |
| [case_studies/Motivation.ipynb](case_studies/Motivation.ipynb) | Fig 1 | Levine data through HDCytoData, Klein data in `data/Klein/` | KMeans fits and one t-SNE |
| [case_studies/Klein.ipynb](case_studies/Klein.ipynb) | Fig 3, Fig 5, S7 Fig, S15 Table | GEO GSE65525 in `data/Klein/`; R with M3C and SC3 | Cached; M3C took 17 minutes and SC3 5 minutes |
| [case_studies/Levine_32dim.ipynb](case_studies/Levine_32dim.ipynb) | S9 Fig, S10 Fig, S11 Fig, S12 Fig | R with HDCytoData, M3C and SC3 | Cached |
| [case_studies/Cusanovich.ipynb](case_studies/Cusanovich.ipynb) | Fig 6, S8 Fig, S16 Table, S17 Table, S14 Table | The atlas data release in `data/Cusanovich/`; the graph extra | Cached; the fit takes about 28 hours at 5 workers on an 11-core Mac |
| [case_studies/hECA.ipynb](case_studies/hECA.ipynb) | S6 Fig, the numbers in S10 Text | The Longleaf run directory in `results/runs/heca/` and the embedding in `data/hECA/`; the umap extra | Reads the run, plus one UMAP |

The R packages install through `make levine-setup`, `make m3c-setup` and `make sc3-setup`. S5 Text and
S7 to S13 Text describe the studies behind these notebooks.
