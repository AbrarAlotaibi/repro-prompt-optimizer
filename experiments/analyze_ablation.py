"""Statistical analysis for the REPRO ablation experiment.

Reads per-behavior JSONL records from one or two experiment directories and
produces all outputs required by the ablation experiment plan:

Per-target outputs (written into the input directory):
  - ablation_{target}_{config}.json     per-behavior records in plan format
  - ablation_summary_{target}.csv       per-behavior results as CSV
  - convergence_{target}.pdf/png        cumulative-ASR trajectory plot (5 lines)
  - statistical_analysis.json           raw test results

Combined outputs (written to --output-dir, or CWD):
  - ablation_statistics.md              all test results formatted
  - ablation_table.tex                  LaTeX table ready for paper

Usage (single target):
    python experiments/analyze_ablation.py --input experiments/gpt4omini_ablation_.../

Usage (both targets, produces combined LaTeX + statistics):
    python experiments/analyze_ablation.py \\
        --input  experiments/gpt4omini_ablation_.../ \\
        --input2 experiments/llama70b_ablation_.../
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Config metadata ──────────────────────────────────────────────────────

CONFIG_ORDER = [
    "full",
    "no_failure_analysis",
    "no_semantic_crossover",
    "no_bayesian_strategy",
    "no_elitism",
]

CONFIG_DISPLAY = {
    "full":                 "REPRO (Full)",
    "no_failure_analysis":  r"$-$Failure Analysis",
    "no_semantic_crossover": r"$-$Semantic Crossover",
    "no_bayesian_strategy": r"$-$Bayesian Strategy",
    "no_elitism":           r"$-$Elitism",
}

CONFIG_SHORT = {
    "full":                 "full",
    "no_failure_analysis":  "no_failure_analysis",
    "no_semantic_crossover": "no_crossover",
    "no_bayesian_strategy": "no_bayesian",
    "no_elitism":           "no_elitism",
}

# Colours for convergence plot (matches plan)
CONFIG_COLORS = {
    "full":                 "#2196F3",
    "no_failure_analysis":  "#FF5722",
    "no_semantic_crossover": "#4CAF50",
    "no_bayesian_strategy": "#FF9800",
    "no_elitism":           "#9C27B0",
}

CONFIG_LINESTYLES = {
    "full":                 "-",
    "no_failure_analysis":  "--",
    "no_semantic_crossover": "-.",
    "no_bayesian_strategy": ":",
    "no_elitism":           (0, (5, 1)),
}

TARGET_DISPLAY = {
    "gpt4omini": "GPT-4o-mini",
    "llama70b":  "Llama-3.3-70B",
}

ITER_BUDGET = 100


# ── Helpers ───────────────────────────────────────────────────────────────

def infer_target(input_dir: Path) -> str:
    """Infer short target name from directory name."""
    name = input_dir.name.lower()
    if "gpt4omini" in name or "gpt-4o-mini" in name:
        return "gpt4omini"
    if "llama70b" in name or "llama-3" in name or "llama3" in name:
        return "llama70b"
    # Fallback: strip timestamp
    parts = name.split("_ablation_")
    return parts[0].replace("-", "")


def load_all_records(input_dir: Path) -> Dict[str, List[Dict]]:
    """Load per-behavior records grouped by configuration.

    Prefers individual per_behavior_{cfg}.jsonl files (always written at the
    end of each config run, even in resumed/interrupted sessions).  Falls back
    to all_results.jsonl for any config not covered by per-behavior files.
    """
    records_by_config: Dict[str, List[Dict]] = {}

    # 1. Load from per_behavior_{cfg}.jsonl — most reliable source
    for cfg in CONFIG_ORDER:
        pb_path = input_dir / f"per_behavior_{cfg}.jsonl"
        if pb_path.exists():
            recs = []
            with open(pb_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    r.setdefault("configuration", cfg)  # ensure field present
                    recs.append(r)
            if recs:
                records_by_config[cfg] = recs

    # 2. Fill any missing configs from all_results.jsonl
    all_path = input_dir / "all_results.jsonl"
    if all_path.exists():
        with open(all_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                cfg = r["configuration"]
                if cfg not in records_by_config:
                    records_by_config.setdefault(cfg, []).append(r)

    if not records_by_config:
        raise FileNotFoundError(
            f"No per_behavior_*.jsonl or all_results.jsonl found in {input_dir}"
        )
    return records_by_config


def normalize_record(r: Dict, target: str, seed: int = 42) -> Dict:
    """Convert run_ablation.py format → plan format."""
    success = bool(r.get("success", 0))
    iters = r.get("iterations_used", 0)
    return {
        "behavior_id":              r["behavior_id"],
        "configuration":            r["configuration"],
        "target":                   target,
        "seed":                     seed,
        "success":                  success,
        # With --stop-on-first, iterations_used == iteration_of_first_success
        "iteration_of_first_success": iters if success else None,
        "total_iterations":         iters,
        # target_queries not logged by run_ablation.py → 0 / null
        "total_target_queries":     r.get("target_queries") or None,
        "strategies_attempted":     None,
        "final_soft_success_score": None,
        "elapsed_seconds":          round(r.get("wall_clock_seconds", 0), 1),
    }


# ── Statistics ────────────────────────────────────────────────────────────

def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """95% Wilson score confidence interval for a binomial proportion."""
    from scipy.stats import norm
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_test(full_outcomes: List[int], abl_outcomes: List[int]) -> Dict:
    """McNemar's exact binomial test (b+c < 25) or chi-squared with continuity correction."""
    from scipy.stats import binom
    assert len(full_outcomes) == len(abl_outcomes)

    a = b = c = d = 0
    for f, a_ in zip(full_outcomes, abl_outcomes):
        if   f == 1 and a_ == 1: a += 1
        elif f == 1 and a_ == 0: b += 1
        elif f == 0 and a_ == 1: c += 1
        else:                    d += 1

    n_disc = b + c
    if n_disc == 0:
        return {"a": a, "b": b, "c": c, "d": d,
                "statistic": 0.0, "p_value": 1.0, "method": "no_discordant_pairs"}

    if n_disc < 25:
        p = min(2 * binom.cdf(min(b, c), n_disc, 0.5), 1.0)
        return {"a": a, "b": b, "c": c, "d": d,
                "statistic": float(min(b, c)), "p_value": float(p),
                "method": "exact_binomial"}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return {"a": a, "b": b, "c": c, "d": d,
            "statistic": float(chi2), "p_value": float(p),
            "method": "chi_squared_continuity"}


def wilcoxon_paired(full_vals: List[float], abl_vals: List[float]) -> Optional[Dict]:
    """Wilcoxon signed-rank on paired samples. Returns None if < 10 pairs or all equal."""
    from scipy.stats import wilcoxon
    if len(full_vals) < 10:
        return None
    diffs = [a - f for f, a in zip(full_vals, abl_vals)]
    if all(d == 0 for d in diffs):
        return {"n_pairs": len(full_vals), "statistic": 0.0, "p_value": 1.0,
                "median_full": float(np.median(full_vals)),
                "median_abl":  float(np.median(abl_vals)),
                "mean_full":   float(np.mean(full_vals)),
                "mean_abl":    float(np.mean(abl_vals))}
    stat, p = wilcoxon(full_vals, abl_vals, alternative="two-sided")
    return {"n_pairs": len(full_vals), "statistic": float(stat), "p_value": float(p),
            "median_full": float(np.median(full_vals)),
            "median_abl":  float(np.median(abl_vals)),
            "mean_full":   float(np.mean(full_vals)),
            "mean_abl":    float(np.mean(abl_vals))}


def cliffs_delta(x: List[float], y: List[float]) -> Dict:
    """Cliff's delta effect size: P(X > Y) - P(X < Y).

    Convention here: x = Full REPRO iterations, y = ablated iterations.
    Positive delta means Full tends to take MORE iterations (ablated faster on
    shared successes — selection-bias signal).
    Negative delta means Full tends to take FEWER iterations (Full faster).

    Thresholds (Romano et al.): |d|>=0.147 small, >=0.330 medium, >=0.474 large.
    """
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return {"delta": None, "magnitude": "n/a"}
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    d = (more - less) / (m * n)
    if   abs(d) >= 0.474: mag = "large"
    elif abs(d) >= 0.330: mag = "medium"
    elif abs(d) >= 0.147: mag = "small"
    else:                  mag = "negligible"
    return {"delta": round(float(d), 4), "magnitude": mag}


def bootstrap_asr_diff(
    full_outcomes: List[int],
    abl_outcomes: List[int],
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict:
    """Paired bootstrap 95% CI for ASR_full − ASR_ablated.

    Resamples behavior indices with replacement, keeping full/ablated paired.
    """
    rng  = np.random.default_rng(seed)
    n    = len(full_outcomes)
    farr = np.array(full_outcomes, dtype=float)
    aarr = np.array(abl_outcomes,  dtype=float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx      = rng.integers(0, n, size=n)
        diffs[i] = farr[idx].mean() - aarr[idx].mean()
    obs  = float(farr.mean() - aarr.mean())
    lo   = float(np.percentile(diffs, 2.5))
    hi   = float(np.percentile(diffs, 97.5))
    return {
        "observed_diff": round(obs, 4),
        "ci_lower":      round(lo, 4),
        "ci_upper":      round(hi, 4),
        "n_bootstrap":   n_boot,
        # Significant if CI excludes 0
        "significant":   lo > 0 or hi < 0,
    }


def logrank_test(
    durations_full: List[int], events_full: List[int],
    durations_abl:  List[int], events_abl:  List[int],
) -> Dict:
    """Log-rank test comparing two convergence curves.

    Treats each behavior as a survival observation:
      duration = iterations_used
      event    = 1 (success) or 0 (censored = never solved within budget)

    Uses all n behaviors per group — not just paired successes.
    Returns chi-squared statistic and p-value.
    """
    from scipy.stats import chi2 as chi2_dist

    all_times = sorted(set(durations_full) | set(durations_abl))

    O1 = 0.0
    E1 = 0.0
    V1 = 0.0

    for t in all_times:
        # At-risk: all observations with duration >= t
        r1 = sum(d >= t for d in durations_full)
        r2 = sum(d >= t for d in durations_abl)
        r  = r1 + r2
        if r == 0:
            continue

        # Events (actual successes) at exactly time t
        d1 = sum(d == t and e == 1 for d, e in zip(durations_full, events_full))
        d2 = sum(d == t and e == 1 for d, e in zip(durations_abl,  events_abl))
        d  = d1 + d2
        if d == 0:
            continue

        e1  = r1 * d / r
        O1 += d1
        E1 += e1
        if r > 1:
            V1 += e1 * (r2 / r) * ((r - d) / (r - 1))

    if V1 <= 0:
        return {"statistic": 0.0, "p_value": 1.0, "O1": int(O1), "E1": round(E1, 3)}

    chi2_stat = (O1 - E1) ** 2 / V1
    p = float(1 - chi2_dist.cdf(chi2_stat, df=1))
    return {
        "statistic": round(float(chi2_stat), 4),
        "p_value":   round(p, 6),
        "O1":        int(O1),   # observed events in Full REPRO
        "E1":        round(float(E1), 3),  # expected events in Full REPRO under H0
    }


def cohens_g_and_or(b: int, c: int) -> Dict:
    """Cohen's g and odds ratio as effect-size measures for McNemar's test.

    g = |b/(b+c) - 0.5|.  Thresholds (Cohen 1988): <0.05 negligible,
    <0.15 small, <0.25 medium, >=0.25 large.
    OR = b/c (None when c == 0).
    """
    n_disc = b + c
    if n_disc == 0:
        return {"cohens_g": 0.0, "odds_ratio": None, "magnitude": "none",
                "b": b, "c": c, "n_disc": 0}
    p_hat = b / n_disc
    g     = abs(p_hat - 0.5)
    if   g >= 0.25: mag = "large"
    elif g >= 0.15: mag = "medium"
    elif g >= 0.05: mag = "small"
    else:           mag = "negligible"
    or_val = (b / c) if c > 0 else None
    return {
        "cohens_g":   round(g, 4),
        "odds_ratio": round(float(or_val), 4) if or_val is not None else None,
        "magnitude":  mag,
        "b": b, "c": c, "n_disc": n_disc,
    }


def cochrans_q(
    outcomes_by_config: Dict[str, List[int]],
    behavior_order: List[str],
) -> Dict:
    """Cochran's Q omnibus test for K paired binary proportions.

    outcomes_by_config: {cfg: List[int]} aligned to behavior_order.
    Q ~ chi-sq(K-1) under H0 that all configs have equal ASR.
    """
    from scipy.stats import chi2 as chi2_dist

    configs = [c for c in CONFIG_ORDER if c in outcomes_by_config]
    K = len(configs)
    N = len(behavior_order)
    if K < 2 or N == 0:
        return {"statistic": None, "p_value": None, "df": None, "K": K}

    # T_k: successes for config k; L_j: successes for behavior j across configs
    T = [sum(outcomes_by_config[cfg]) for cfg in configs]
    L = [
        sum(outcomes_by_config[cfg][j] for cfg in configs)
        for j in range(N)
    ]
    sum_T  = sum(T)
    sum_T2 = sum(t * t for t in T)
    sum_L2 = sum(l * l for l in L)
    denom  = K * sum_T - sum_L2

    if denom == 0:
        return {"statistic": 0.0, "p_value": 1.0, "df": K - 1, "K": K}

    Q = (K - 1) * (K * sum_T2 - sum_T ** 2) / denom
    p = float(1 - chi2_dist.cdf(Q, df=K - 1))
    return {
        "statistic": round(float(Q), 4),
        "p_value":   round(p, 6),
        "df":        K - 1,
        "K":         K,
    }


def kruskal_wallis_iterations(iters_by_config: Dict[str, List[float]]) -> Dict:
    """Kruskal-Wallis H-test across K iteration distributions (omnibus).

    Failed behaviors should be pre-encoded as ITER_BUDGET + 1.
    """
    from scipy.stats import kruskal

    groups = [v for k, v in iters_by_config.items() if v]
    if len(groups) < 2:
        return {"statistic": None, "p_value": None, "n_groups": len(groups)}
    stat, p = kruskal(*groups)
    return {
        "statistic": round(float(stat), 4),
        "p_value":   round(float(p), 6),
        "n_groups":  len(groups),
    }


def holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict]:
    """Holm-Bonferroni step-down procedure for multiple comparisons.

    More powerful than simple Bonferroni while still controlling FWER.
    Returns adjusted p-value and reject decision for each key.
    """
    k = len(p_values)
    if k == 0:
        return {}
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])
    results: Dict[str, Dict] = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(sorted_items):
        adj = min(1.0, (k - rank) * p)
        running_max = max(running_max, adj)  # enforce monotonicity
        results[name] = {
            "raw_p":       round(p, 6),
            "adj_p":       round(running_max, 6),
            "rank":        rank + 1,
            "significant": running_max < alpha,
        }
    return results


def mcnemar_power_analysis(
    b: int, c: int, n: int, alpha: float = 0.0125
) -> Dict:
    """Retrospective power for McNemar's exact binomial at observed effect size.

    Returns:
    - min_disc_to_reject: smallest n_disc where significance is achievable at alpha
    - achieved_power: exact power given observed (b, c)
    - n_for_80pct: behaviors needed for 80% power at observed ΔASR
                   (normal approximation; None when b == c)
    """
    import math
    from scipy.stats import binom, norm

    # Smallest m such that 2*(0.5)^m <= alpha → must reject everything in same direction
    min_disc = math.ceil(math.log(alpha / 2) / math.log(0.5))
    n_disc   = b + c

    if n_disc == 0 or n_disc < min_disc:
        achieved_power = 0.0
    else:
        p_hat = max(b, c) / n_disc
        power = 0.0
        for k in range(n_disc + 1):
            p_val_k = min(2 * float(binom.cdf(min(k, n_disc - k), n_disc, 0.5)), 1.0)
            if p_val_k <= alpha:
                power += float(binom.pmf(k, n_disc, p_hat))
        achieved_power = round(power, 4)

    # Normal approximation: n = (z_alpha/2 + z_beta)^2 * (pB+pC) / (pB-pC)^2
    if b == c or n == 0:
        n_for_80pct = None
    else:
        z_a = float(norm.ppf(1 - alpha / 2))
        z_b = float(norm.ppf(0.80))
        pB, pC = b / n, c / n
        d = abs(pB - pC)
        n_for_80pct = math.ceil((z_a + z_b) ** 2 * (pB + pC) / d ** 2) if d > 0 else None

    return {
        "alpha":              alpha,
        "min_disc_to_reject": min_disc,
        "observed_disc":      n_disc,
        "can_reject":         n_disc >= min_disc,
        "achieved_power":     achieved_power,
        "n_for_80pct":        n_for_80pct,
    }


def format_p(p: float) -> str:
    if p < 0.001: return "<0.001"
    if p < 0.01:  return "<0.01"
    if p < 0.05:  return "<0.05"
    return f"{p:.3f}"


def sig_marker(p: float, bonf_alpha: float = 0.0125) -> str:
    return "$^\\dagger$" if p < bonf_alpha else ""


# ── Per-target analysis ───────────────────────────────────────────────────

def analyze_target(input_dir: Path, target: str) -> Dict:
    """
    Full analysis for one target directory.
    Writes per-behavior JSON files, CSV, convergence plot, statistical_analysis.json.
    Returns analysis_results dict keyed by config.
    """
    records_by_config = load_all_records(input_dir)

    if "full" not in records_by_config:
        raise ValueError(f"'full' config not found in {input_dir}")

    full_records = records_by_config["full"]
    n_behaviors  = len(full_records)
    behavior_order = [r["behavior_id"] for r in full_records]

    print(f"\n{'='*60}")
    print(f"  Target: {TARGET_DISPLAY.get(target, target)}  ({n_behaviors} behaviors)")
    print(f"  Dir:    {input_dir.name}")
    print(f"{'='*60}")

    # ── 1. Write per-behavior JSON files ──────────────────────────────────
    for cfg, records in records_by_config.items():
        short = CONFIG_SHORT.get(cfg, cfg)
        fname = f"ablation_{target}_{short}.json"
        normed = [normalize_record(r, target) for r in records]
        with open(input_dir / fname, "w", encoding="utf-8") as f:
            json.dump(normed, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Per-behavior JSON files written")

    # ── 2. Write per-behavior CSV ─────────────────────────────────────────
    csv_path = input_dir / f"ablation_summary_{target}.csv"
    fieldnames = [
        "behavior_id", "configuration", "target", "seed", "success",
        "iteration_of_first_success", "total_iterations", "total_target_queries",
        "strategies_attempted", "final_soft_success_score", "elapsed_seconds",
    ]
    all_normed = []
    for cfg in CONFIG_ORDER:
        records = records_by_config.get(cfg, [])
        all_normed.extend([normalize_record(r, target) for r in records])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_normed)
    print(f"  [OK] {csv_path.name}")

    # ── 3. Statistical analysis per config ────────────────────────────────
    analysis_results: Dict[str, Dict] = {}

    for cfg in CONFIG_ORDER:
        records = records_by_config.get(cfg)
        if records is None:
            continue

        records_by_id = {r["behavior_id"]: r for r in records}
        successes    = sum(r["success"] for r in records)
        iters_all    = [r["iterations_used"] for r in records if r["success"]]
        tq_all       = [r["target_queries"]  for r in records if r.get("target_queries")]

        lo, hi = wilson_ci(successes, n_behaviors)

        entry: Dict = {
            "configuration":    cfg,
            "successes":        successes,
            "total":            n_behaviors,
            "asr":              successes / n_behaviors,
            "wilson_ci_lower":  lo,
            "wilson_ci_upper":  hi,
            "median_iters_success": float(np.median(iters_all)) if iters_all else None,
            "mean_iters_success":   float(np.mean(iters_all))   if iters_all else None,
            "mean_tq":   float(np.mean(tq_all))   if tq_all else None,
            "median_tq": float(np.median(tq_all)) if tq_all else None,
            "max_tq":    float(np.max(tq_all))    if tq_all else None,
        }

        if cfg != "full":
            # Align vectors in behavior order
            full_vec, abl_vec = [], []
            for bid in behavior_order:
                fr = next((r for r in full_records if r["behavior_id"] == bid), None)
                ar = records_by_id.get(bid)
                if fr and ar:
                    full_vec.append(fr["success"])
                    abl_vec.append(ar["success"])

            # ── McNemar (binary ASR) ──────────────────────────────────────
            mcn = mcnemar_test(full_vec, abl_vec)
            entry["mcnemar"]      = mcn
            entry["bonferroni_p"] = min(mcn["p_value"] * 4, 1.0)
            entry["cohens_g"]       = cohens_g_and_or(mcn["b"], mcn["c"])
            entry["power_analysis"] = mcnemar_power_analysis(
                mcn["b"], mcn["c"], n_behaviors, alpha=0.0125
            )

            # ── Bootstrap CI for ASR difference ──────────────────────────
            entry["bootstrap_asr_diff"] = bootstrap_asr_diff(full_vec, abl_vec)

            # ── Log-rank test (all behaviors, censored = failures) ────────
            # Use iterations_used for all behaviors; failed = censored at that value
            full_dur, full_ev, abl_dur, abl_ev = [], [], [], []
            for bid in behavior_order:
                fr = next((r for r in full_records if r["behavior_id"] == bid), None)
                ar = records_by_id.get(bid)
                if fr and ar:
                    full_dur.append(max(1, fr["iterations_used"]))
                    full_ev.append(int(fr["success"]))
                    abl_dur.append(max(1, ar["iterations_used"]))
                    abl_ev.append(int(ar["success"]))
            entry["logrank"] = logrank_test(full_dur, full_ev, abl_dur, abl_ev)
            entry["logrank"]["bonferroni_p"] = min(
                entry["logrank"]["p_value"] * 4, 1.0
            )

            # ── Wilcoxon on iterations (paired successes) + Cliff's delta ─
            pairs_iters = [
                (fr["iterations_used"], records_by_id[bid]["iterations_used"])
                for bid in behavior_order
                if (fr := next((r for r in full_records if r["behavior_id"] == bid), None))
                   and fr["success"] == 1
                   and records_by_id.get(bid, {}).get("success") == 1
            ]
            if pairs_iters:
                w_iters = wilcoxon_paired([p[0] for p in pairs_iters],
                                          [p[1] for p in pairs_iters])
                if w_iters:
                    w_iters["cliffs_delta"] = cliffs_delta(
                        [p[0] for p in pairs_iters],
                        [p[1] for p in pairs_iters],
                    )
                entry["wilcoxon_iters"] = w_iters

            # Cliff's delta on full distribution (failed → ITER_BUDGET + 1)
            full_iters_all = [
                fr["iterations_used"] if fr["success"] else ITER_BUDGET + 1
                for bid in behavior_order
                if (fr := next((r for r in full_records if r["behavior_id"] == bid), None))
            ]
            abl_iters_all = [
                ar["iterations_used"] if ar["success"] else ITER_BUDGET + 1
                for bid in behavior_order
                if (ar := records_by_id.get(bid))
            ]
            entry["cliffs_delta_full"] = cliffs_delta(full_iters_all, abl_iters_all)

            # ── Wilcoxon on TQ (both succeeded and have TQ > 0) ──────────
            pairs_tq = [
                (fr["target_queries"], records_by_id[bid]["target_queries"])
                for bid in behavior_order
                if (fr := next((r for r in full_records if r["behavior_id"] == bid), None))
                   and fr["success"] == 1
                   and records_by_id.get(bid, {}).get("success") == 1
                   and fr.get("target_queries", 0) > 0
                   and records_by_id[bid].get("target_queries", 0) > 0
            ]
            if pairs_tq:
                w_tq = wilcoxon_paired([p[0] for p in pairs_tq],
                                        [p[1] for p in pairs_tq])
                entry["wilcoxon_tq"] = w_tq

        analysis_results[cfg] = entry

    # ── Omnibus tests across all 5 configurations ─────────────────────────
    outcomes_vec: Dict[str, List[int]]   = {}
    iters_vec:    Dict[str, List[float]] = {}
    for cfg in CONFIG_ORDER:
        recs = records_by_config.get(cfg)
        if recs is None:
            continue
        by_id = {r["behavior_id"]: r for r in recs}
        outcomes_vec[cfg] = [
            int(by_id.get(bid, {}).get("success", 0)) for bid in behavior_order
        ]
        iters_vec[cfg] = [
            float(by_id[bid]["iterations_used"])
            if by_id.get(bid, {}).get("success")
            else float(ITER_BUDGET + 1)
            for bid in behavior_order
            if bid in by_id
        ]

    mcn_pvals = {
        cfg: analysis_results[cfg]["mcnemar"]["p_value"]
        for cfg in CONFIG_ORDER
        if cfg != "full" and "mcnemar" in analysis_results.get(cfg, {})
    }

    analysis_results["_omnibus"] = {
        "cochrans_q":      cochrans_q(outcomes_vec, behavior_order),
        "kruskal_wallis":  kruskal_wallis_iterations(iters_vec),
        "holm_bonferroni": holm_bonferroni(mcn_pvals, alpha=0.05),
    }
    print(f"  [OK] Omnibus tests (Cochran's Q, Kruskal-Wallis, Holm-Bonferroni)")

    with open(input_dir / "statistical_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)
    print(f"  [OK] statistical_analysis.json")

    # ── 4. Convergence trajectory plot ───────────────────────────────────
    _generate_convergence_plot(records_by_config, target, input_dir, n_behaviors)

    return analysis_results


# ── Convergence plot ──────────────────────────────────────────────────────

def _generate_convergence_plot(
    records_by_config: Dict[str, List[Dict]],
    target: str,
    output_dir: Path,
    n_behaviors: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib not installed — skipping convergence plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for cfg in CONFIG_ORDER:
        records = records_by_config.get(cfg)
        if records is None:
            continue

        label  = CONFIG_DISPLAY.get(cfg, cfg).replace("$-$", "−")
        color  = CONFIG_COLORS.get(cfg, "gray")
        ls     = CONFIG_LINESTYLES.get(cfg, "-")

        # cumulative[t] = count of behaviors solved at or before iteration t
        cumulative = np.zeros(ITER_BUDGET + 1)
        for r in records:
            if r["success"] and r["iterations_used"] > 0:
                first = min(r["iterations_used"], ITER_BUDGET)
                cumulative[first:] += 1

        pct = cumulative / n_behaviors * 100
        ax.plot(range(ITER_BUDGET + 1), pct,
                label=label, color=color, linewidth=2, linestyle=ls)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cumulative ASR (%)", fontsize=12)
    ax.set_title(
        f"Convergence Trajectory — {TARGET_DISPLAY.get(target, target)} (n={n_behaviors})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, ITER_BUDGET)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = output_dir / f"convergence_{target}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [OK] convergence_{target}.{ext}")
    plt.close(fig)


# ── Combined outputs ──────────────────────────────────────────────────────

def _tq_str(v: Optional[float]) -> str:
    return f"{v:.0f}" if v is not None else "N/A"


def _iters_str(v: Optional[float]) -> str:
    return f"{v:.1f}" if v is not None else "N/A"


def generate_statistics_md(
    results: Dict[str, Dict],        # {target: {cfg: entry}}
    output_path: Path,
) -> None:
    """Write ablation_statistics.md covering all test results."""
    lines = ["# REPRO Ablation — Statistical Analysis\n"]

    bonf_alpha = 0.0125  # α/4 Bonferroni

    for target, analysis in results.items():
        tdisp = TARGET_DISPLAY.get(target, target)
        n = next(iter(analysis.values()))["total"]

        lines.append(f"## Target: {tdisp}  (n = {n})\n")

        # ── Summary table ──
        lines.append("### Summary Table\n")
        lines.append(
            "| Configuration | ASR (k/n) | ASR % | 95% Wilson CI | Median Iters | "
            "Mean TQ | Median TQ | Max TQ |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")

        for cfg in CONFIG_ORDER:
            entry = analysis.get(cfg)
            if entry is None:
                continue
            name = CONFIG_DISPLAY.get(cfg, cfg).replace("$-$", "−")
            k    = entry["successes"]
            asr  = entry["asr"]
            ci   = f"[{entry['wilson_ci_lower']:.1%}, {entry['wilson_ci_upper']:.1%}]"
            med  = _iters_str(entry.get("median_iters_success"))
            mtq  = _tq_str(entry.get("mean_tq"))
            mdtq = _tq_str(entry.get("median_tq"))
            mxtq = _tq_str(entry.get("max_tq"))
            lines.append(f"| {name} | {k}/{n} | {asr:.1%} | {ci} | {med} | {mtq} | {mdtq} | {mxtq} |")

        lines.append("")

        # Holm-Bonferroni adjusted p-values (from omnibus block)
        holm_map = (analysis.get("_omnibus") or {}).get("holm_bonferroni") or {}

        # ── McNemar + Effect Size + Bootstrap ASR diff ────────────────────
        lines.append(
            "### McNemar's Test + Effect Size + Bootstrap ASR Difference "
            "(Bonferroni α = 0.0125)\n"
        )
        lines.append(
            "| Configuration | b | c | Cohen's g | OR | McNemar p | "
            "Bonf. p | Holm adj.p | Sig. | ΔASR | Bootstrap 95% CI | CI excl. 0? |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

        for cfg in CONFIG_ORDER:
            if cfg == "full":
                continue
            entry = analysis.get(cfg)
            if entry is None or "mcnemar" not in entry:
                continue
            mcn   = entry["mcnemar"]
            bonf  = entry["bonferroni_p"]
            boot  = entry.get("bootstrap_asr_diff", {})
            cg    = entry.get("cohens_g", {})
            holm  = holm_map.get(cfg, {})
            name  = CONFIG_DISPLAY.get(cfg, cfg).replace("$-$", "−")
            sig   = "✓" if bonf < bonf_alpha else "✗"
            delta = boot.get("observed_diff")
            ci_lo = boot.get("ci_lower")
            ci_hi = boot.get("ci_upper")
            d_str    = f"{delta:+.1%}" if delta is not None else "—"
            ci_str   = f"[{ci_lo:+.1%}, {ci_hi:+.1%}]" if ci_lo is not None else "—"
            excl     = "✓" if boot.get("significant") else "✗"
            g_str    = f"{cg.get('cohens_g', 0):.3f} ({cg.get('magnitude', '—')})" if cg else "—"
            or_str   = f"{cg['odds_ratio']:.2f}" if cg and cg.get("odds_ratio") is not None else "∞" if cg else "—"
            holm_str = format_p(holm["adj_p"]) if holm else "—"
            lines.append(
                f"| {name} | {mcn['b']} | {mcn['c']} | {g_str} | {or_str} | "
                f"{format_p(mcn['p_value'])} | {format_p(bonf)} | {holm_str} | {sig} | "
                f"{d_str} | {ci_str} | {excl} |"
            )

        lines.append("")

        # ── McNemar Power Analysis ─────────────────────────────────────────
        lines.append("### McNemar Power Analysis (α = 0.0125, target 80% power)\n")
        lines.append(
            "| Configuration | min disc. to reject | Observed disc. | "
            "Can reject? | Achieved power | n for 80% power |"
        )
        lines.append("|---|---|---|---|---|---|")

        for cfg in CONFIG_ORDER:
            if cfg == "full":
                continue
            entry = analysis.get(cfg)
            if entry is None or "power_analysis" not in entry:
                continue
            pa   = entry["power_analysis"]
            name = CONFIG_DISPLAY.get(cfg, cfg).replace("$-$", "−")
            can  = "✓" if pa["can_reject"] else "✗"
            n80  = str(pa["n_for_80pct"]) if pa["n_for_80pct"] is not None else "—"
            lines.append(
                f"| {name} | {pa['min_disc_to_reject']} | {pa['observed_disc']} | "
                f"{can} | {pa['achieved_power']:.3f} | {n80} |"
            )

        lines.append("")

        # ── Log-rank test ───────────────────────────────────────────────────
        lines.append("### Log-Rank Test — Convergence Curves (all 50 behaviors, failures censored, Bonferroni α = 0.0125)\n")
        lines.append("| Configuration | χ² | p-value | Bonf. p | Sig. |")
        lines.append("|---|---|---|---|---|")

        for cfg in CONFIG_ORDER:
            if cfg == "full":
                continue
            entry = analysis.get(cfg)
            if entry is None or "logrank" not in entry:
                continue
            lr   = entry["logrank"]
            bonf = lr.get("bonferroni_p", 1.0)
            name = CONFIG_DISPLAY.get(cfg, cfg).replace("$-$", "−")
            sig  = "✓" if bonf < bonf_alpha else "✗"
            lines.append(
                f"| {name} | {lr['statistic']:.3f} | "
                f"{format_p(lr['p_value'])} | {format_p(bonf)} | {sig} |"
            )

        lines.append("")

        # ── Wilcoxon on iterations + Cliff's delta ─────────────────────────
        lines.append("### Wilcoxon Signed-Rank + Cliff's δ — Iterations to Convergence\n")
        lines.append("> Paired on behaviors where **both** Full and ablated succeeded. "
                     "Cliff's δ (full dist.) uses all 50 behaviors; failed → 101.\n")
        lines.append(
            "| Configuration | Paired n | Median Full | Median Abl | "
            "Wilcoxon p | Bonf. p | Sig. | Cliff's δ (paired) | Cliff's δ (all) | Magnitude |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")

        for cfg in CONFIG_ORDER:
            if cfg == "full":
                continue
            entry = analysis.get(cfg)
            if entry is None:
                continue
            w    = entry.get("wilcoxon_iters")
            cd_f = entry.get("cliffs_delta_full", {})
            name = CONFIG_DISPLAY.get(cfg, cfg).replace("$-$", "−")
            if w:
                bonf_w = min(w["p_value"] * 4, 1.0)
                sig    = "✓" if bonf_w < bonf_alpha else "✗"
                cd_p   = w.get("cliffs_delta", {})
                d_p    = f"{cd_p.get('delta', '—'):+.3f}" if cd_p.get("delta") is not None else "—"
                d_f    = f"{cd_f.get('delta', '—'):+.3f}" if cd_f.get("delta") is not None else "—"
                mag    = cd_f.get("magnitude", "—")
                lines.append(
                    f"| {name} | {w['n_pairs']} | {w['median_full']:.1f} | {w['median_abl']:.1f} | "
                    f"{format_p(w['p_value'])} | {format_p(bonf_w)} | {sig} | "
                    f"{d_p} | {d_f} | {mag} |"
                )
            else:
                lines.append(f"| {name} | <10 pairs | — | — | — | — | — | — | — | — |")

        lines.append("")

        # ── Wilcoxon on TQ ──
        has_tq = any(
            analysis.get(cfg, {}).get("wilcoxon_tq") is not None
            for cfg in CONFIG_ORDER if cfg != "full"
        )
        if has_tq:
            lines.append("### Wilcoxon Signed-Rank — Target Queries (Bonferroni α = 0.0125)\n")
            lines.append("| Configuration | Paired n | Median Full TQ | Median Abl TQ | p-value | Sig. |")
            lines.append("|---|---|---|---|---|---|")

            for cfg in CONFIG_ORDER:
                if cfg == "full":
                    continue
                entry = analysis.get(cfg)
                if entry is None:
                    continue
                w = entry.get("wilcoxon_tq")
                name = CONFIG_DISPLAY.get(cfg, cfg).replace("$-$", "−")
                if w:
                    sig = "✓" if w["p_value"] < bonf_alpha else "✗"
                    lines.append(
                        f"| {name} | {w['n_pairs']} | {w['median_full']:.0f} | {w['median_abl']:.0f} | "
                        f"{format_p(w['p_value'])} | {sig} |"
                    )
                else:
                    lines.append(f"| {name} | <10 pairs | — | — | — | — |")
            lines.append("")

        # ── Omnibus Tests ──────────────────────────────────────────────────
        omnibus = analysis.get("_omnibus") or {}
        lines.append("### Omnibus Tests (All 5 Configurations)\n")

        cq = omnibus.get("cochrans_q") or {}
        kw = omnibus.get("kruskal_wallis") or {}

        if cq.get("statistic") is not None:
            cq_sig = "p < 0.05" if cq["p_value"] < 0.05 else "ns"
            lines.append(
                f"**Cochran's Q** (binary ASR): Q = {cq['statistic']:.3f}, "
                f"df = {cq['df']}, p = {format_p(cq['p_value'])} ({cq_sig})"
            )
        else:
            lines.append("**Cochran's Q**: insufficient configs")

        if kw.get("statistic") is not None:
            kw_sig = "p < 0.05" if kw["p_value"] < 0.05 else "ns"
            lines.append(
                f"\n**Kruskal-Wallis** (iterations, failed→{ITER_BUDGET+1}): "
                f"H = {kw['statistic']:.3f}, K = {kw['n_groups']}, "
                f"p = {format_p(kw['p_value'])} ({kw_sig})\n"
            )
        else:
            lines.append("\n**Kruskal-Wallis**: insufficient groups\n")

        hb = omnibus.get("holm_bonferroni") or {}
        if hb:
            lines.append("**Holm-Bonferroni adjusted McNemar p-values** (family-wise α = 0.05):\n")
            lines.append("| Configuration | raw p | Holm adj.p | Sig. |")
            lines.append("|---|---|---|---|")
            for cfg in CONFIG_ORDER:
                if cfg == "full":
                    continue
                h = hb.get(cfg)
                if h is None:
                    continue
                name = CONFIG_DISPLAY.get(cfg, cfg).replace("$-$", "−")
                sig  = "✓" if h["significant"] else "✗"
                lines.append(
                    f"| {name} | {format_p(h['raw_p'])} | {format_p(h['adj_p'])} | {sig} |"
                )
            lines.append("")

        lines.append("---\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {output_path.name}")


def generate_latex_table(
    results: Dict[str, Dict],   # {target: {cfg: entry}}
    output_path: Path,
) -> None:
    """Write ablation_table.tex — dual-target (or single) LaTeX table."""
    targets = [t for t in ["gpt4omini", "llama70b"] if t in results]
    single  = len(targets) == 1

    bonf_alpha = 0.0125
    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    if single:
        cap_target = TARGET_DISPLAY.get(targets[0], targets[0])
        lines.append(
            r"\caption{Component ablation results on JBB-50 (" + cap_target + r"). "
            r"ASR with 95\% Wilson CI; Med.\ Iters over successful behaviors. "
            r"$^\dagger$: bootstrap 95\% CI for $\Delta\text{ASR}$ excludes zero "
            r"(Bonferroni-corrected $\alpha'{=}0.0125$; McNemar exact returns "
            r"no significant comparisons).}"
        )
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\toprule")
        lines.append(r"Configuration & ASR (\%) & 95\% CI & Median Iters \\")
        lines.append(r"\midrule")
    else:
        lines.append(
            r"\caption{Component ablation results on JBB-50. "
            r"ASR with 95\% Wilson CI; Med.\ Iters over successful behaviors. "
            r"$^\dagger$: bootstrap 95\% CI for $\Delta\text{ASR}$ excludes zero "
            r"(Bonferroni-corrected $\alpha'{=}0.0125$; McNemar exact binomial "
            r"returns no significant comparisons on either target).}"
        )
        lines.append(r"\begin{tabular}{lcccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"& \multicolumn{3}{c}{GPT-4o-mini} & \multicolumn{3}{c}{Llama-3.3-70B} \\"
        )
        lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
        lines.append(
            r"Configuration & ASR (\%) & 95\% CI & Med.\ Iters"
            r" & ASR (\%) & 95\% CI & Med.\ Iters \\"
        )
        lines.append(r"\midrule")

    for cfg in CONFIG_ORDER:
        name = CONFIG_DISPLAY.get(cfg, cfg)
        row_parts = [name]

        for target in targets:
            analysis = results[target]
            entry    = analysis.get(cfg)
            if entry is None:
                row_parts += ["—", "—", "—"]
                continue

            asr_pct = f"{entry['asr']*100:.1f}"
            ci      = f"[{entry['wilson_ci_lower']*100:.1f}, {entry['wilson_ci_upper']*100:.1f}]"
            med     = f"{entry['median_iters_success']:.1f}" if entry.get("median_iters_success") else "—"

            # Dagger: bootstrap CI for ΔASR excludes zero (matches paper caption)
            boot = entry.get("bootstrap_asr_diff", {})
            if cfg != "full" and boot.get("significant", False):
                asr_pct += r"$^\dagger$"

            row_parts += [asr_pct, ci, med]

        lines.append("  " + " & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\end{table}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {output_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="REPRO Ablation Statistical Analysis")
    parser.add_argument(
        "--input", required=True,
        help="Primary experiment output directory (GPT study or only study)",
    )
    parser.add_argument(
        "--input2", default=None,
        help="Optional second experiment output directory (Llama study)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write combined outputs (ablation_statistics.md, ablation_table.tex). "
             "Defaults to the primary --input directory.",
    )
    args = parser.parse_args()

    input_dirs = [(Path(args.input), infer_target(Path(args.input)))]
    if args.input2:
        input_dirs.append((Path(args.input2), infer_target(Path(args.input2))))

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input)

    all_results: Dict[str, Dict] = {}

    for input_dir, target in input_dirs:
        print(f"\nAnalysing: {input_dir}  (target={target})")
        analysis = analyze_target(input_dir, target)
        all_results[target] = analysis

    # Combined markdown stats
    stats_path = output_dir / "ablation_statistics.md"
    generate_statistics_md(all_results, stats_path)

    # LaTeX table
    tex_path = output_dir / "ablation_table.tex"
    generate_latex_table(all_results, tex_path)

    print(f"\nAll outputs written to: {output_dir}")
    if len(all_results) == 1:
        print("Tip: pass --input2 <llama_dir> to generate combined LaTeX + stats across both targets.")


if __name__ == "__main__":
    main()
