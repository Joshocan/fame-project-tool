# Draft result tables for MDE Intelligence
## Table A. Model-wise breakdown (Federation dataset, cleaned subset)
| Model | Runs | Well-formed rate | Satisfiable rate | Semantic F1 | Semantic Precision | Semantic Recall | Coverage |
|---|---:|---:|---:|---|---|---|---|
| deepseek-v3.2:cloud | 117 | 0.709 | 1.000 | 0.647 ± 0.067 (95% CI ± 0.014) | 0.818 ± 0.064 (95% CI ± 0.013) | 0.549 ± 0.109 (95% CI ± 0.023) | 33.881 ± 7.432 (95% CI ± 1.544) |
| gemini-3.1-pro-preview | 70 | 0.400 | 1.000 | 0.569 ± 0.082 (95% CI ± 0.019) | 0.849 ± 0.062 (95% CI ± 0.014) | 0.443 ± 0.127 (95% CI ± 0.030) | 26.197 ± 8.691 (95% CI ± 2.036) |
| glm-4.7:cloud | 127 | 0.535 | 1.000 | 0.628 ± 0.102 (95% CI ± 0.019) | 0.831 ± 0.061 (95% CI ± 0.011) | 0.530 ± 0.167 (95% CI ± 0.031) | 32.438 ± 12.633 (95% CI ± 2.361) |
| gpt-4.1 | 82 | 0.732 | 1.000 | 0.616 ± 0.059 (95% CI ± 0.015) | 0.847 ± 0.070 (95% CI ± 0.017) | 0.488 ± 0.069 (95% CI ± 0.017) | 30.468 ± 5.707 (95% CI ± 1.421) |
| gpt-oss:120b-cloud | 128 | 0.828 | 1.000 | 0.633 ± 0.094 (95% CI ± 0.017) | 0.822 ± 0.055 (95% CI ± 0.010) | 0.526 ± 0.121 (95% CI ± 0.021) | 34.472 ± 10.054 (95% CI ± 1.770) |

## Table B. Pipeline-wise breakdown (Federation dataset, cleaned subset)
| Pipeline | Runs | Well-formed rate | Satisfiable rate | Semantic F1 | Semantic Precision | Semantic Recall | Coverage |
|---|---:|---:|---:|---|---|---|---|
| is_nonrag | 59 | 0.932 | 1.000 | 0.665 ± 0.116 (95% CI ± 0.030) | 0.777 ± 0.050 (95% CI ± 0.013) | 0.613 ± 0.183 (95% CI ± 0.048) | 38.483 ± 14.473 (95% CI ± 3.791) |
| is_rag | 86 | 0.895 | 1.000 | 0.715 ± 0.055 (95% CI ± 0.012) | 0.779 ± 0.032 (95% CI ± 0.007) | 0.669 ± 0.088 (95% CI ± 0.019) | 42.885 ± 6.617 (95% CI ± 1.450) |
| ss_nonrag | 196 | 0.495 | 1.000 | 0.569 ± 0.058 (95% CI ± 0.009) | 0.828 ± 0.053 (95% CI ± 0.008) | 0.437 ± 0.067 (95% CI ± 0.010) | 26.280 ± 5.512 (95% CI ± 0.851) |
| ss_rag | 183 | 0.634 | 1.000 | 0.615 ± 0.070 (95% CI ± 0.011) | 0.878 ± 0.047 (95% CI ± 0.007) | 0.478 ± 0.083 (95% CI ± 0.013) | 30.150 ± 7.139 (95% CI ± 1.113) |

## Table C. Expert agreement
| Measure | Value | Notes |
|---|---:|---|
| Cohen's kappa on Relevance (Y/N) | 1.000 | Binary agreement across 14 shared FM evaluations |
| Weighted kappa on feature correctness | -0.144 | Ordinal 0--10 expert scores |
| Weighted kappa on structural quality | 0.349 | Ordinal 0--10 expert scores |
| Weighted kappa on constraint validity | 0.571 | Ordinal 0--10 expert scores |
| Weighted kappa on overall score | 0.225 | Ordinal 0--10 expert scores |
