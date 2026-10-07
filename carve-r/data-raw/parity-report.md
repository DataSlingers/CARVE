# R and Python parity report

Generated on 2026-10-06 with CARVE 2.0.0 in R and carve 1.0.0 in Python, 100 resamples per configuration and random_state 0. The k cases use the light preset over k = 2 to 6.

## easy blobs

Rows compared: 15 of 15.

| Column | Largest difference | Tolerance |
|---|---|---|
| ari_stability | 0.067 (over) | 0.05 |
| ari_generalizability | 0.073 (over) | 0.05 |
| ari_average | 0.049 | 0.05 |
| consensus_pac_stability | 0.124 (over) | 0.03 |
| consensus_gini_stability | 0.064 (over) | 0.03 |
| consensus_ce_stability | 0.069 (over) | 0.03 |
| accuracy_generalizability | 0.026 | 0.03 |
| n_clusters_observed | 0.000 | 0.50 |
| noise_fraction | 0.000 | 0.02 |

| Measure | Rule | R | Python |
|---|---|---|---|
| stability | max | 3 | 3 |
| generalizability | max | 2 | 2 |
| average | max | 2 | 2 |
| stability | 1se | 3 | 3 |
| generalizability | 1se | 2 | 2 |
| average | 1se | 2 | 2 |

ARI between the R and Python labels at the default selection (stability, 1se): 1.000

## hard blobs

Rows compared: 15 of 15.

| Column | Largest difference | Tolerance |
|---|---|---|
| ari_stability | 0.032 | 0.05 |
| ari_generalizability | 0.047 | 0.05 |
| ari_average | 0.032 | 0.05 |
| consensus_pac_stability | 0.027 | 0.03 |
| consensus_gini_stability | 0.024 | 0.03 |
| consensus_ce_stability | 0.023 | 0.03 |
| accuracy_generalizability | 0.020 | 0.03 |
| n_clusters_observed | 0.000 | 0.50 |
| noise_fraction | 0.000 | 0.02 |

| Measure | Rule | R | Python |
|---|---|---|---|
| stability | max | 2 | 2 |
| generalizability | max | 2 | 2 |
| average | max | 2 | 2 |
| stability | 1se | 2 | 2 |
| generalizability | 1se | 2 | 2 |
| average | 1se | 2 | 2 |

ARI between the R and Python labels at the default selection (stability, 1se): 1.000

## circles

Rows compared: 15 of 15.

| Column | Largest difference | Tolerance |
|---|---|---|
| ari_stability | 0.084 (over) | 0.05 |
| ari_generalizability | 0.036 | 0.05 |
| ari_average | 0.040 | 0.05 |
| consensus_pac_stability | 0.057 (over) | 0.03 |
| consensus_gini_stability | 0.041 (over) | 0.03 |
| consensus_ce_stability | 0.038 (over) | 0.03 |
| accuracy_generalizability | 0.015 | 0.03 |
| n_clusters_observed | 0.000 | 0.50 |
| noise_fraction | 0.000 | 0.02 |

| Measure | Rule | R | Python |
|---|---|---|---|
| stability | max | 2 | 2 |
| generalizability | max | 2 | 5 |
| average | max | 2 | 2 |
| stability | 1se | 2 | 2 |
| generalizability | 1se | 2 | 5 |
| average | 1se | 2 | 2 |

ARI between the R and Python labels at the default selection (stability, 1se): 1.000

## resolution sweep, 10-dimensional blobs

Leiden runs in igraph in R and in leidenalg in Python. Both optimize modularity at each resolution.

Rows compared: 10 of 10.

| Column | Largest difference | Tolerance |
|---|---|---|
| ari_stability | 0.002 | 0.05 |
| ari_generalizability | 0.008 | 0.05 |
| ari_average | 0.004 | 0.05 |
| consensus_pac_stability | 0.008 | 0.03 |
| consensus_gini_stability | 0.005 | 0.03 |
| consensus_ce_stability | 0.008 | 0.03 |
| accuracy_generalizability | 0.010 | 0.03 |
| n_clusters_observed | 0.070 | 0.50 |
| noise_fraction | 0.000 | 0.02 |

| Measure | Rule | R | Python |
|---|---|---|---|
| stability | max | 0.1 | 0.1 |
| generalizability | max | 0.25 | 0.25 |
| average | max | 0.25 | 0.25 |
| stability | 1se | 1 | 1 |
| generalizability | 1se | 1 | 1 |
| average | 1se | 1 | 1 |

ARI between the R and Python labels at the default selection (stability, 1se): 1.000

## min_cluster_size sweep, HDBSCAN

dbscan and scikit-learn can merge tied distances in a different order, so HDBSCAN can select different clusters on some subsamples (see the HDBSCAN help page).

Rows compared: 4 of 4.

| Column | Largest difference | Tolerance |
|---|---|---|
| ari_stability | 0.006 | 0.05 |
| ari_generalizability | 0.003 | 0.05 |
| ari_average | 0.002 | 0.05 |
| consensus_pac_stability | 0.007 | 0.03 |
| consensus_gini_stability | 0.005 | 0.03 |
| consensus_ce_stability | 0.005 | 0.03 |
| accuracy_generalizability | 0.001 | 0.03 |
| n_clusters_observed | 0.010 | 0.50 |
| noise_fraction | 0.001 | 0.02 |

| Measure | Rule | R | Python |
|---|---|---|---|
| stability | max | 5 | 10 |
| generalizability | max | 5 | 5 |
| average | max | 5 | 10 |
| stability | 1se | 5 | 10 |
| generalizability | 1se | 5 | 5 |
| average | 1se | 5 | 5 |

ARI between the R and Python labels at the default selection (stability, 1se): 1.000

## easy blobs, anchored over 200 of 500 samples

The two languages draw different anchors. PAC covers anchor pairs only, so it can differ more than in the exact case.

Rows compared: 15 of 15.

| Column | Largest difference | Tolerance |
|---|---|---|
| ari_stability | 0.067 (over) | 0.05 |
| ari_generalizability | 0.073 (over) | 0.05 |
| ari_average | 0.049 | 0.05 |
| consensus_pac_stability | 0.144 (over) | 0.03 |
| consensus_gini_stability | 0.065 (over) | 0.03 |
| consensus_ce_stability | 0.070 (over) | 0.03 |
| accuracy_generalizability | 0.026 | 0.03 |
| n_clusters_observed | 0.000 | 0.50 |
| noise_fraction | 0.000 | 0.02 |

| Measure | Rule | R | Python |
|---|---|---|---|
| stability | max | 3 | 3 |
| generalizability | max | 2 | 2 |
| average | max | 2 | 2 |
| stability | 1se | 3 | 3 |
| generalizability | 1se | 2 | 2 |
| average | 1se | 2 | 2 |

ARI between the R and Python labels at the default selection (stability, 1se): 1.000

## easy blobs, randomized preprocessing

Each of the four pipelines gets 25 of the 100 resamples in both languages, so n_resamples must agree exactly.

Rows compared: 9 of 9.

| Column | Largest difference | Tolerance |
|---|---|---|
| ari_stability | 0.083 (over) | 0.05 |
| ari_generalizability | 0.084 (over) | 0.05 |
| ari_average | 0.084 (over) | 0.05 |
| consensus_pac_stability | 0.330 (over) | 0.03 |
| consensus_gini_stability | 0.097 (over) | 0.03 |
| consensus_ce_stability | 0.129 (over) | 0.03 |
| accuracy_generalizability | 0.032 (over) | 0.03 |
| n_clusters_observed | 0.000 | 0.50 |
| noise_fraction | 0.000 | 0.02 |

| Measure | Rule | R | Python |
|---|---|---|---|
| stability | max | 3 | 3 |
| generalizability | max | 2 | 2 |
| average | max | 2 | 2 |
| stability | 1se | 3 | 3 |
| generalizability | 1se | 2 | 2 |
| average | 1se | 2 | 2 |

ARI between the R and Python labels at the default selection (stability, 1se): 1.000

Rows of preprocessing_results():

Rows compared: 36 of 36.

| Column | Largest difference | Tolerance |
|---|---|---|
| n_resamples | 0.000 | 0.00 |
| ari_stability | 0.191 (over) | 0.05 |
| ari_generalizability | 0.222 (over) | 0.05 |

