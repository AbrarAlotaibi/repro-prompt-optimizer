# REPRO: Reflective Evolutionary Prompt Optimization

REPRO is a modular jailbreak-attack optimizer for evaluating the safety of large language models.
It combines an evolutionary algorithm with a reflective language model to discover successful
adversarial prompts, and ships a full ablation study over its four core components.

---

## Overview

REPRO optimizes prompts through an evolutionary loop driven by four tightly integrated components:

| Component | Role |
|---|---|
| **Failure Analysis** | Classifies refusals, routes counter-strategies, tracks per-behavior failure history |
| **Semantic Crossover** | Recombines candidates from multiple parents to escape local optima |
| **Bayesian Strategy Selection** | Thompson-sampling over strategy priors; concentrates budget on high-yield strategies |
| **Elitism** | Forwards top-*E* candidates across generations; prevents regression on partially solved behaviors |

Additional infrastructure: reflective LLM mutation, Pareto-based selection, population diversity
preservation, and multi-provider support (OpenAI, Anthropic, Gemini, vLLM, ExLlamaV3, Requesty).

---

## Ablation Study

### Setup

We evaluate all five configurations (Full REPRO + one-component-removed ablations) on a
stratified 50-behavior subset of [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)
(JBB-50, seed 42) against two targets of differing resistance:

- **GPT-4o-mini** (OpenAI) — easily jailbroken; exposes efficiency differences when ASR saturates
- **Llama-3.3-70B-Instruct** (Meta / DeepInfra) — harder target; shows ASR differentiation

**Infrastructure:** all models local or self-hosted.

| Role | Model | Provider | Port |
|---|---|---|---|
| Reflective LM | Mistral-7B-Instruct-v0.3 (EXL3 4-bit) | TabbyAPI exllamav3 | 5012 |
| Refiner / Prefilter | Mistral-7B-Instruct-v0.3 (EXL3 4-bit) | TabbyAPI exllamav3 | 5013 |
| Judge | Qwen3-8B (EXL3 4-bit) | TabbyAPI exllamav3 | 5017 |

**Hyperparameters:** exploit mode, 100 iterations/behavior, population 24, elite size 3, seed 42,
`--num-rewordings 0` (binary per-candidate ASR).

---

### Results

#### Attack Success Rate (ASR) — JBB-50

| Configuration | GPT-4o-mini ASR | 95% CI | Med. Iters | Llama-3.3-70B ASR | 95% CI | Med. Iters |
|---|---|---|---|---|---|---|
| **REPRO (Full)** | **100.0%** (50/50) | [92.9, 100.0] | 3 | **98.0%** (49/50) | [89.5, 99.6] | 3 |
| − Failure Analysis | 100.0% (50/50) | [92.9, 100.0] | 2 | 94.0% (47/50) | [83.8, 97.9] | 1 |
| − Semantic Crossover | 90.0%† (45/50) | [78.6, 95.7] | 1 | 90.0%† (45/50) | [78.6, 95.7] | 2 |
| − Bayesian Strategy | 88.0%† (44/50) | [76.2, 94.4] | 1 | 90.0%† (45/50) | [78.6, 95.7] | 1 |
| − Elitism | 90.0%† (45/50) | [78.6, 95.7] | 3 | 94.0% (47/50) | [83.8, 97.9] | 1 |

† bootstrap 95% CI for ΔASR excludes zero (Bonferroni-corrected α′ = 0.0125).

---

#### Bootstrap ΔASR — Full REPRO vs. Ablated (paired, 10 000 resamples)

| Ablation | GPT-4o-mini ΔASR | 95% CI | Llama-3.3-70B ΔASR | 95% CI |
|---|---|---|---|---|
| − Failure Analysis | +0% | [0%, 0%] | +4% | [0%, +10%] |
| − Semantic Crossover | **+10%** | [+2%, +18%] | **+8%** | [+2%, +16%] |
| − Bayesian Strategy | **+12%** | [+4%, +22%] | **+8%** | [+2%, +16%] |
| − Elitism | **+10%** | [+2%, +20%] | +4% | [−4%, +12%] |

Semantic crossover and Bayesian strategy selection show the most consistent gains across both
targets. Elitism is clearly beneficial on the easier target; the evidence is weaker on the harder
one. Failure analysis contributes only on the harder target where refusal patterns are structured.

---

#### Omnibus Tests

| Test | GPT-4o-mini | Llama-3.3-70B |
|---|---|---|
| **Cochran's Q** (ASR, 5 configs) | Q = 16.57, df = 4, **p = 0.002** | Q = 4.87, df = 4, p = 0.301 |
| **Kruskal-Wallis** (iterations) | H = 4.12, K = 5, p = 0.389 | H = 10.24, K = 5, **p = 0.037** |

Cochran's Q confirms a significant omnibus ASR difference across configurations on GPT-4o-mini.
No single component dominates convergence speed; the efficiency story is in final ASR ceiling,
not iterations-to-solution.

---

#### Power Analysis Note

McNemar's exact binomial test at α′ = 0.0125 requires ≥ 8 discordant pairs to reject; with
≤ 6 pairs in every comparison the test has zero power. Bootstrap and Cochran's Q carry the
inferential weight. Approximately **93–140 behaviors** would be needed for 80% power at the
observed 8–12% ΔASR effect sizes.

---

### Experiment Artifacts

All raw data, per-behavior logs, and analysis outputs are included in this repository:

```
experiments/
├── gpt4omini_ablation_20260417_003149/   # GPT-4o-mini study
│   ├── config.json                        # Experiment configuration
│   ├── per_behavior_{cfg}.jsonl           # Per-behavior results per config
│   ├── ablation_statistics.md             # Full statistical report
│   ├── ablation_table.tex                 # LaTeX table (paper-ready)
│   ├── statistical_analysis.json          # Raw test results (all tests)
│   ├── ablation_summary_gpt4omini.csv     # Per-behavior CSV
│   ├── convergence_gpt4omini.{pdf,png}    # Convergence trajectory plot
│   └── runs/                              # Per-behavior run directories
│       └── {config}/{behavior_id}/
│           ├── run_metadata.json
│           ├── best_attack.json
│           └── mastered_attacks.jsonl
│
├── llama70b_ablation_20260422_070304/     # Llama-3.3-70B study
│   └── (same structure as above)
│
├── run_ablation.py                        # GPT ablation runner
├── run_ablation_llama.py                  # Llama ablation runner
├── analyze_ablation.py                    # Statistical analysis script
└── ablation_writeup.tex                   # Full paper section (LaTeX)
```

To regenerate all statistics and plots:

```bash
python experiments/analyze_ablation.py \
    --input  experiments/gpt4omini_ablation_20260417_003149 \
    --input2 experiments/llama70b_ablation_20260422_070304
```

---

## Setup

```bash
git clone https://github.com/AbrarAlotaibi/repro-prompt-optimizer.git
cd repro-prompt-optimizer

pip install -r requirements.txt

cp .env.example .env
# Add API keys for target / Requesty (judge + reflective can be fully local)
```

### Local model servers (TabbyAPI)

| Instance | Config | Port | Model |
|---|---|---|---|
| Reflective LM | `config_mistral.yml` | 5012 | Mistral-7B-Instruct-v0.3 EXL3 4-bit |
| Refiner | `config_refiner.yml` | 5013 | Mistral-7B-Instruct-v0.3 EXL3 4-bit |
| Judge | `config_qwen3_8b.yml` | 5017 | Qwen3-8B EXL3 4-bit |

---

## Usage

```bash
# Run via CLI
python -m repro.cli --help

# Minimal exploit-mode example
python -m repro.cli \
  --optimize-mode exploit \
  --target-model gpt-4o-mini \
  --target-provider openai \
  --judge-model Qwen3-8B-exl3-4.0bpw \
  --judge-provider exllamav3 \
  --judge-url http://127.0.0.1:5017/v1 \
  --reflective-llm Mistral-7B-Instruct-v0.3-exl3-4.0bpw \
  --reflective-provider exllamav3 \
  --dataset jbb \
  --iters 100 \
  --population 24 \
  --report-dir ./output
```

```python
# Library usage
from repro import REPROAttackOptimizer, Task, Evaluator
from repro.target import OpenAIModel
from repro.judge import OpenAIJudge
```

---

## Repository Structure

```
repro/
├── algorithm/          # Core evolutionary optimizer
│   ├── optimizer.py
│   └── optimizer_evaluation.py
├── clients/            # LLM provider clients
├── core/               # Configuration and monitoring
├── monitoring/         # Observability infrastructure
├── cli.py              # CLI entry point
├── target.py           # Target model wrappers
├── judge.py            # Judging logic
└── seed_strategies.py  # Initial seed generation

experiments/
├── run_ablation.py         # GPT-4o-mini ablation runner
├── run_ablation_llama.py   # Llama-3.3-70B ablation runner
├── analyze_ablation.py     # Statistical analysis (9 tests)
├── ablation_writeup.tex    # Paper section
└── gpt4omini_ablation_*/   # Experiment outputs (GPT)
└── llama70b_ablation_*/    # Experiment outputs (Llama)
```
