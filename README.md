# Signal Credibility Score: Multi-Window Validation, FDR Calibration, and Power Analysis for Machine Learning Trading Signals

This repository contains the full implementation and data for the paper:

> **Khan, H. (2026).** *Signal Credibility Score: Multi-Window Validation, FDR Calibration, and Power Analysis for Machine Learning Trading Signals.* Submitted to Journal of Financial Data Science.

## Overview

This paper extends the SCS framework introduced in [Khan & Danoffre (2026)](https://github.com/HamzaKhan2002/scs-framework) with three methodological innovations:

1. **S_model v2:** Performance-weighted Spearman correlation with a Kish effective-N guard, replacing the significance-counting approach of v1. This fixes the consensus-on-failure problem where models agreeing on poor predictions inflate S_model.
2. **Multi-window out-of-sample testing:** Three regime-distinct OOS windows (2021 bull, 2022 bear, 2023 recovery) replacing a single OOS year.
3. **FDR calibration and power analysis:** Monte Carlo false discovery rate estimation (500 seeds, 6,000 null trials) and oracle-based power analysis.

Applied to 12 signal groups across 27 US equities (discovery 2010-2013, walk-forward 2014-2022, OOS 2021-2023):

- **Phase A:** 4 of 12 groups pass SCS-A >= 0.70
- **Phase B:** All 4 pass SCS-B >= 0.60
- **Phase C:** OOS Sharpe ratios 0.47-0.92; no signal significantly outperforms buy-and-hold
- **FDR:** 0% false positives at tau >= 0.75 (6,000 null trials)
- **Power:** >= 90.5% detection across all oracle noise levels at tau = 0.75

## Repository Structure

```
config.yaml              # Single source of truth for all pipeline parameters
src/                     # Core modules
  data/                  #   Config loader and Yahoo Finance data downloader
  features/              #   11 deterministic technical features from OHLCV
  labeling/              #   Binary directional and ternary volatility-adaptive labels
  models/                #   LightGBM, XGBoost, Logistic Regression classifiers
  validation/            #   SCS-A (v2) and SCS-B scoring, purged temporal split
  backtest/              #   Multi-position long/short portfolio engine
  statistics/            #   Bootstrap, DSR, permutation test, Ledoit-Wolf
pipelines/               # Phase A, B, C, multi-window, FDR, power analysis pipelines
tests/                   # 42 automated tests (pytest)
data/cache/              # Cached Yahoo Finance OHLCV data (parquet)
results/                 # All experiment outputs (JSON)
figures/                 # Publication figures (PDF + PNG)
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
@article{khan2026scs_extended,
  title={Signal Credibility Score: Multi-Window Validation, FDR Calibration, and Power Analysis for Machine Learning Trading Signals},
  author={Khan, Hamza},
  year={2026},
  note={Submitted to Journal of Financial Data Science}
}
```

## Related Work

This paper extends the SCS framework introduced in:

> Khan, H. & Danoffre, A. (2026). *Signal Credibility Score: A Multi-Phase Validation Framework for Machine Learning Trading Signals.* [Code](https://github.com/HamzaKhan2002/scs-framework).
