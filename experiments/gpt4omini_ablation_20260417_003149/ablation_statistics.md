# REPRO Ablation — Statistical Analysis

## Target: GPT-4o-mini  (n = 50)

### Summary Table

| Configuration | ASR (k/n) | ASR % | 95% Wilson CI | Median Iters | Mean TQ | Median TQ | Max TQ |
|---|---|---|---|---|---|---|---|
| REPRO (Full) | 50/50 | 100.0% | [92.9%, 100.0%] | 3.0 | N/A | N/A | N/A |
| −Failure Analysis | 50/50 | 100.0% | [92.9%, 100.0%] | 2.0 | N/A | N/A | N/A |
| −Semantic Crossover | 45/50 | 90.0% | [78.6%, 95.7%] | 1.0 | N/A | N/A | N/A |
| −Bayesian Strategy | 44/50 | 88.0% | [76.2%, 94.4%] | 1.0 | N/A | N/A | N/A |
| −Elitism | 45/50 | 90.0% | [78.6%, 95.7%] | 3.0 | N/A | N/A | N/A |

### McNemar's Test + Effect Size + Bootstrap ASR Difference (Bonferroni α = 0.0125)

| Configuration | b | c | Cohen's g | OR | McNemar p | Bonf. p | Holm adj.p | Sig. | ΔASR | Bootstrap 95% CI | CI excl. 0? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| −Failure Analysis | 0 | 0 | 0.000 (none) | ∞ | 1.000 | 1.000 | 1.000 | ✗ | +0.0% | [+0.0%, +0.0%] | ✗ |
| −Semantic Crossover | 5 | 0 | 0.500 (large) | ∞ | 0.062 | 0.250 | 0.188 | ✗ | +10.0% | [+2.0%, +18.0%] | ✓ |
| −Bayesian Strategy | 6 | 0 | 0.500 (large) | ∞ | <0.05 | 0.125 | 0.125 | ✗ | +12.0% | [+4.0%, +22.0%] | ✓ |
| −Elitism | 5 | 0 | 0.500 (large) | ∞ | 0.062 | 0.250 | 0.188 | ✗ | +10.0% | [+2.0%, +20.0%] | ✓ |

### McNemar Power Analysis (α = 0.0125, target 80% power)

| Configuration | min disc. to reject | Observed disc. | Can reject? | Achieved power | n for 80% power |
|---|---|---|---|---|---|
| −Failure Analysis | 8 | 0 | ✗ | 0.000 | — |
| −Semantic Crossover | 8 | 5 | ✗ | 0.000 | 112 |
| −Bayesian Strategy | 8 | 6 | ✗ | 0.000 | 93 |
| −Elitism | 8 | 5 | ✗ | 0.000 | 112 |

### Log-Rank Test — Convergence Curves (all 50 behaviors, failures censored, Bonferroni α = 0.0125)

| Configuration | χ² | p-value | Bonf. p | Sig. |
|---|---|---|---|---|
| −Failure Analysis | 4.533 | <0.05 | 0.133 | ✗ |
| −Semantic Crossover | 0.029 | 0.865 | 1.000 | ✗ |
| −Bayesian Strategy | 0.001 | 0.981 | 1.000 | ✗ |
| −Elitism | 0.516 | 0.473 | 1.000 | ✗ |

### Wilcoxon Signed-Rank + Cliff's δ — Iterations to Convergence

> Paired on behaviors where **both** Full and ablated succeeded. Cliff's δ (full dist.) uses all 50 behaviors; failed → 101.

| Configuration | Paired n | Median Full | Median Abl | Wilcoxon p | Bonf. p | Sig. | Cliff's δ (paired) | Cliff's δ (all) | Magnitude |
|---|---|---|---|---|---|---|---|---|---|
| −Failure Analysis | 50 | 3.0 | 2.0 | <0.05 | 0.103 | ✗ | +0.160 | +0.160 | small |
| −Semantic Crossover | 45 | 3.0 | 1.0 | <0.05 | <0.05 | ✗ | +0.174 | +0.123 | negligible |
| −Bayesian Strategy | 44 | 2.5 | 1.0 | 0.124 | 0.496 | ✗ | +0.139 | +0.077 | negligible |
| −Elitism | 45 | 3.0 | 3.0 | 0.393 | 1.000 | ✗ | -0.029 | -0.034 | negligible |

### Omnibus Tests (All 5 Configurations)

**Cochran's Q** (binary ASR): Q = 16.571, df = 4, p = <0.01 (p < 0.05)

**Kruskal-Wallis** (iterations, failed→101): H = 4.125, K = 5, p = 0.389 (ns)

**Holm-Bonferroni adjusted McNemar p-values** (family-wise α = 0.05):

| Configuration | raw p | Holm adj.p | Sig. |
|---|---|---|---|
| −Failure Analysis | 1.000 | 1.000 | ✗ |
| −Semantic Crossover | 0.062 | 0.188 | ✗ |
| −Bayesian Strategy | <0.05 | 0.125 | ✗ |
| −Elitism | 0.062 | 0.188 | ✗ |

---

## Target: Llama-3.3-70B  (n = 50)

### Summary Table

| Configuration | ASR (k/n) | ASR % | 95% Wilson CI | Median Iters | Mean TQ | Median TQ | Max TQ |
|---|---|---|---|---|---|---|---|
| REPRO (Full) | 49/50 | 98.0% | [89.5%, 99.6%] | 3.0 | N/A | N/A | N/A |
| −Failure Analysis | 47/50 | 94.0% | [83.8%, 97.9%] | 1.0 | N/A | N/A | N/A |
| −Semantic Crossover | 45/50 | 90.0% | [78.6%, 95.7%] | 2.0 | N/A | N/A | N/A |
| −Bayesian Strategy | 45/50 | 90.0% | [78.6%, 95.7%] | 1.0 | N/A | N/A | N/A |
| −Elitism | 47/50 | 94.0% | [83.8%, 97.9%] | 1.0 | N/A | N/A | N/A |

### McNemar's Test + Effect Size + Bootstrap ASR Difference (Bonferroni α = 0.0125)

| Configuration | b | c | Cohen's g | OR | McNemar p | Bonf. p | Holm adj.p | Sig. | ΔASR | Bootstrap 95% CI | CI excl. 0? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| −Failure Analysis | 2 | 0 | 0.500 (large) | ∞ | 0.500 | 1.000 | 1.000 | ✗ | +4.0% | [+0.0%, +10.0%] | ✗ |
| −Semantic Crossover | 4 | 0 | 0.500 (large) | ∞ | 0.125 | 0.500 | 0.500 | ✗ | +8.0% | [+2.0%, +16.0%] | ✓ |
| −Bayesian Strategy | 4 | 0 | 0.500 (large) | ∞ | 0.125 | 0.500 | 0.500 | ✗ | +8.0% | [+2.0%, +16.0%] | ✓ |
| −Elitism | 3 | 1 | 0.250 (large) | 3.00 | 0.625 | 1.000 | 1.000 | ✗ | +4.0% | [-4.0%, +12.0%] | ✗ |

### McNemar Power Analysis (α = 0.0125, target 80% power)

| Configuration | min disc. to reject | Observed disc. | Can reject? | Achieved power | n for 80% power |
|---|---|---|---|---|---|
| −Failure Analysis | 8 | 2 | ✗ | 0.000 | 279 |
| −Semantic Crossover | 8 | 4 | ✗ | 0.000 | 140 |
| −Bayesian Strategy | 8 | 4 | ✗ | 0.000 | 140 |
| −Elitism | 8 | 4 | ✗ | 0.000 | 558 |

### Log-Rank Test — Convergence Curves (all 50 behaviors, failures censored, Bonferroni α = 0.0125)

| Configuration | χ² | p-value | Bonf. p | Sig. |
|---|---|---|---|---|
| −Failure Analysis | 0.464 | 0.496 | 1.000 | ✗ |
| −Semantic Crossover | 0.352 | 0.553 | 1.000 | ✗ |
| −Bayesian Strategy | 3.146 | 0.076 | 0.304 | ✗ |
| −Elitism | 0.356 | 0.550 | 1.000 | ✗ |

### Wilcoxon Signed-Rank + Cliff's δ — Iterations to Convergence

> Paired on behaviors where **both** Full and ablated succeeded. Cliff's δ (full dist.) uses all 50 behaviors; failed → 101.

| Configuration | Paired n | Median Full | Median Abl | Wilcoxon p | Bonf. p | Sig. | Cliff's δ (paired) | Cliff's δ (all) | Magnitude |
|---|---|---|---|---|---|---|---|---|---|
| −Failure Analysis | 47 | 2.0 | 1.0 | 0.566 | 1.000 | ✗ | +0.055 | +0.032 | negligible |
| −Semantic Crossover | 45 | 2.0 | 2.0 | 0.811 | 1.000 | ✗ | -0.011 | -0.031 | negligible |
| −Bayesian Strategy | 45 | 2.0 | 1.0 | <0.01 | <0.05 | ✗ | +0.245 | +0.172 | small |
| −Elitism | 46 | 2.0 | 1.0 | <0.05 | 0.086 | ✗ | +0.321 | +0.265 | small |

### Omnibus Tests (All 5 Configurations)

**Cochran's Q** (binary ASR): Q = 4.870, df = 4, p = 0.301 (ns)

**Kruskal-Wallis** (iterations, failed→101): H = 10.237, K = 5, p = <0.05 (p < 0.05)

**Holm-Bonferroni adjusted McNemar p-values** (family-wise α = 0.05):

| Configuration | raw p | Holm adj.p | Sig. |
|---|---|---|---|
| −Failure Analysis | 0.500 | 1.000 | ✗ |
| −Semantic Crossover | 0.125 | 0.500 | ✗ |
| −Bayesian Strategy | 0.125 | 0.500 | ✗ |
| −Elitism | 0.625 | 1.000 | ✗ |

---
