# Signal Credibility Score: A Multi-Phase Validation Framework for Machine Learning Trading Signals

This repository contains the full implementation and data for the paper:

> **Khan, H. & Danoffre, A. (2026).** *Signal Credibility Score: A Multi-Phase Validation Framework for Machine Learning Trading Signals.* Submitted to Journal of Financial Data Science.

## Overview

The Signal Credibility Score (SCS) is a three-phase validation framework for machine-learning trading signals, designed to separate genuine predictive ability from overfitting artifacts:

1. **Phase A (Discovery, 2010-2013):** Exhaustive search over 6 horizons × 2 label strategies × 3 models × 5 seeds × 3 sub-periods (540 runs per ticker, 14,580 total). Scores each of 12 signal groups with SCS-A, a composite of accuracy, Sharpe ratio, stability, drawdown, and model consensus (performance-weighted Spearman with Kish effective-N guard).
2. **Phase B (Walk-Forward, 2014-2022):** Expanding-window walk-forward validation on held-out data with purged temporal splits. Scores with SCS-B (five sub-scores, stricter thresholds).
3. **Phase C (Out-of-Sample, 2023-2025):** Frozen models applied to three non-overlapping OOS windows (2023, 2024, 2025) with bootstrap, Deflated Sharpe Ratio, and Ledoit-Wolf tests.

Applied to 12 signal groups across 27 US equities (14,580 runs):

- **Phase A:** 5 of 12 groups pass SCS-A >= 0.70
- **Phase B:** All 5 pass SCS-B >= 0.60
- **Phase C:** OOS Sharpe ratios averaging +0.76 (2023), +1.32 (2024), +0.68 (2025); no signal significantly outperforms buy-and-hold
- **FDR:** 0 observed false positives at tau >= 0.70 (12,000 null trials; Clopper-Pearson 95% CI [0, 0.03%])
- **Power:** 96-100% observed detection across tested oracle noise levels

## Repository Structure

```
config.yaml              # Single source of truth for all pipeline parameters
src/                     # Core modules
  data/                  #   Config loader and Yahoo Finance data downloader
  features/              #   11 deterministic technical features from OHLCV
  labeling/              #   Binary directional and ternary volatility-adaptive labels
  models/                #   LightGBM, XGBoost, Logistic Regression classifiers
  validation/            #   SCS-A and SCS-B scoring, purged temporal split
  backtest/              #   Multi-position long/short portfolio engine
  statistics/            #   Bootstrap, DSR, permutation test, Ledoit-Wolf
pipelines/               # Phase A, B, C, multi-window, FDR, power analysis pipelines
tests/                   # Automated tests (pytest)
results/                 # All experiment outputs (JSON)
figures/                 # Publication figures (PNG)
paper_scs_framework.tex  # LaTeX manuscript
cover_letter_jfds.tex    # Cover letter for JFDS
```

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v              # Run all tests
python -m pipelines.run_all             # Run full pipeline (Phase A + B + C)
python -m pipelines.run_fdr_simulation  # Run FDR calibration (slow)
python -m pipelines.run_power_analysis  # Run power analysis (slow)
```

## Citation

If you use this framework, please cite:

```bibtex
@article{khan2026scs,
  title={Signal Credibility Score: A Multi-Phase Validation Framework for Machine Learning Trading Signals},
  author={Khan, Hamza and Danoffre, Alexandre},
  year={2026},
  note={Submitted to Journal of Financial Data Science}
}
```
