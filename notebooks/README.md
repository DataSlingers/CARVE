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
| [Benchmarking.ipynb](Benchmarking.ipynb) | Fig 4, S1 Fig, S4 Fig, S5 Fig, S2 to S7 Table, S11 to S13 Table | Run artifacts in `results/runs/` from `python -m benchmarks.run --all` | Reads the artifacts only |
| [Resampling_Ablation.ipynb](Resampling_Ablation.ipynb) | S2 Fig, S3 Fig, S8 to S10 Table | A run from `python -m benchmarks.run --ablation rho_b` | Reads the run only |
| [case_studies/Motivation.ipynb](case_studies/Motivation.ipynb) | Fig 1 | Levine data through HDCytoData, Klein data in `data/Klein/` | KMeans fits and one t-SNE |
| [case_studies/Klein.ipynb](case_studies/Klein.ipynb) | Fig 3, Fig 5, S7 Fig, S15 Table | GEO GSE65525 in `data/Klein/`; R with M3C and SC3 | Cached; M3C took 17 minutes and SC3 5 minutes |
| [case_studies/Levine_32dim.ipynb](case_studies/Levine_32dim.ipynb) | S9 Fig, S10 Fig, S11 Fig, S12 Fig | R with HDCytoData, M3C and SC3 | Cached |
| [case_studies/Cusanovich.ipynb](case_studies/Cusanovich.ipynb) | Fig 6, S8 Fig, S16 Table, S17 Table, S14 Table | The atlas data release in `data/Cusanovich/`; the graph extra | Cached; the fit takes about 28 hours at 5 workers on an 11-core Mac |
| [case_studies/hECA.ipynb](case_studies/hECA.ipynb) | S6 Fig, the numbers in S10 Text | The Longleaf run directory in `results/runs/heca/` and the embedding in `data/hECA/`; the umap extra | Reads the run, plus one UMAP |

The R packages install through `make levine-setup`, `make m3c-setup` and `make sc3-setup`. S5 Text and
S7 to S13 Text describe the studies behind these notebooks.

## Terms

| Term | Meaning |
|---|---|
| Configuration | One clustering estimator with fixed hyperparameters at one value of the swept parameter, k or a resolution. |
| Resample | One draw of two overlapping subsamples, each holding a fraction rho (`subsample_ratio`, 0.618 by default) of the samples, and a test set, the samples outside the first subsample. B (`n_resamples`, 100 by default) is the number of resamples. |
| ARI | The adjusted Rand index: 1 for two labelings that agree up to the names of the clusters, near 0 for unrelated ones. |
| Stability | The mean over resamples of the ARI between the clusterings of the two subsamples, on the samples they share. |
| Generalizability | The mean over resamples of the ARI between a classifier's predictions for the test set, trained on the first subsample's clusters, and the test set's own clustering. |
| Consensus matrix | For each pair of samples, the share of the resamples drawing both in which the two fell in the same cluster. CARVE keeps one built from the subsample clusterings (stability) and one from the classifier's predictions on the test sets (generalizability). |
| Consensus labels | The labels `get_labels()` returns: an average-linkage tree built from the consensus matrix, cut at the selected number of clusters. |
| Per-sample scores | Gini and cross-entropy stability, from a sample's row of the consensus matrix, are 1 when every other sample drawn with it fell in its cluster either every time or never. Accuracy is the share of the test sets holding the sample in which the classifier predicted its cluster. The case studies call these per-cell scores. |
| PAC | The proportion of ambiguous clustering, the share of consensus entries strictly between 0.05 and 0.95. CARVE reports 1 - PAC. |
| Selection rules | `max` takes the best mean score. `1se` takes the finest configuration whose score is within one standard error of the best, and `quantile` the finest within the 5th to 95th percentiles of the best configuration's resample scores. Finest means the most clusters on a k sweep and the highest `sweep_rank` on other sweeps. `not_two` leaves k = 2 out. |
| Randomized preprocessing | Each resample draws one normalization and dimensionality-reduction pipeline and fits it on its subsamples, so that the scores can be split by pipeline. |
| Anchors | On large data, the samples over which the consensus matrices are kept (`consensus_anchors`). Every sample still gets its per-sample scores. |
| k* and k-hat | The true number of clusters of a simulation, and the number a method selects. |
| Oracle | The ARI of a scenario's own clustering method run at k*. |
| k-recovery | The share of data sets in which a method selects k*. |
| Difficulty | The easy, medium and hard settings of a simulation, at which the oracle reaches an ARI of about 0.9 to 1.0, 0.8 to 0.9 and 0.7 to 0.8. |
| CVIs | The classical cluster validity indices: silhouette and Calinski-Harabasz (higher is better), Davies-Bouldin (lower is better) and the gap statistic (Tibshirani's rule). |
| M3C, SC3 | Consensus clustering tools in R. M3C picks the k with the largest relative cluster stability index (RCSI); SC3's k here is its Tracy-Widom estimate from the eigenvalues of the data. |

The guide to reading the curves of the criteria over k is under Reading the curves in section 4 of [Tutorial.ipynb](Tutorial.ipynb). S4 Text of the manuscript gives the same guidance with figures.
