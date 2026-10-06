# R and Python parity report

Generated on 2026-10-06 with CARVE 2.0.0 in R and carve 1.0.0 in Python: the light preset, k from 2 to 6, 100 resamples, random_state 0.

## easy blobs

Configurations compared: 15 of 15.

| Column | Largest difference | Tolerance |
|---|---|---|
| ari_stability | 0.067 (over) | 0.05 |
| ari_generalizability | 0.073 (over) | 0.05 |
| ari_average | 0.049 | 0.05 |
| consensus_pac_stability | 0.124 (over) | 0.03 |
| consensus_gini_stability | 0.064 (over) | 0.03 |
| consensus_ce_stability | 0.069 (over) | 0.03 |
| accuracy_generalizability | 0.026 | 0.03 |

| Measure | Rule | k in R | k in Python |
|---|---|---|---|
| stability | max | 3 | 3 |
| generalizability | max | 2 | 2 |
| average | max | 2 | 2 |
| stability | 1se | 3 | 3 |
| generalizability | 1se | 2 | 2 |
| average | 1se | 2 | 2 |

ARI between the R and Python labels at the default selection (stability, 1se): 1.000

## hard blobs

Configurations compared: 15 of 15.

| Column | Largest difference | Tolerance |
|---|---|---|
| ari_stability | 0.032 | 0.05 |
| ari_generalizability | 0.047 | 0.05 |
| ari_average | 0.032 | 0.05 |
| consensus_pac_stability | 0.027 | 0.03 |
| consensus_gini_stability | 0.024 | 0.03 |
| consensus_ce_stability | 0.023 | 0.03 |
| accuracy_generalizability | 0.020 | 0.03 |

| Measure | Rule | k in R | k in Python |
|---|---|---|---|
| stability | max | 2 | 2 |
| generalizability | max | 2 | 2 |
| average | max | 2 | 2 |
| stability | 1se | 2 | 2 |
| generalizability | 1se | 2 | 2 |
| average | 1se | 2 | 2 |

ARI between the R and Python labels at the default selection (stability, 1se): 1.000

## circles

Configurations compared: 15 of 15.

| Column | Largest difference | Tolerance |
|---|---|---|
| ari_stability | 0.084 (over) | 0.05 |
| ari_generalizability | 0.036 | 0.05 |
| ari_average | 0.040 | 0.05 |
| consensus_pac_stability | 0.057 (over) | 0.03 |
| consensus_gini_stability | 0.041 (over) | 0.03 |
| consensus_ce_stability | 0.038 (over) | 0.03 |
| accuracy_generalizability | 0.015 | 0.03 |

| Measure | Rule | k in R | k in Python |
|---|---|---|---|
| stability | max | 2 | 2 |
| generalizability | max | 2 | 5 |
| average | max | 2 | 2 |
| stability | 1se | 2 | 2 |
| generalizability | 1se | 2 | 5 |
| average | 1se | 2 | 2 |

ARI between the R and Python labels at the default selection (stability, 1se): 1.000

