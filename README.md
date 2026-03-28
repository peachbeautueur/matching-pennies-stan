# matching-pennies-stan

Single agent belief learning model with softmax choice for matching pennies, implemented in Stan.

## Overview

This repository contains a coursework style analysis pipeline for a single agent matching pennies model. The project includes:

- a Stan implementation of the belief learning model
- scripts for simulation, model fitting, model checks, and parameter recovery
- six generated figures used in the report
- a report draft in Markdown

## Main Files

- `stan/belief_model.stan`: main Stan model used for fitting
- `stan/belief_model_commented.stan`: commented reference version of the Stan model
- `scripts/simulate_data.py`: simulate example matching pennies data
- `scripts/fit_model.py`: fit the Stan model to a dataset
- `scripts/model_checks.py`: prior predictive, posterior predictive, and prior posterior checks
- `scripts/parameter_recovery.py`: parameter recovery across trial counts
- `scripts/run_analysis.py`: end to end workflow that regenerates the analysis outputs
- `report/assignment_report.md`: coursework report draft

## Basic Workflow

Run the full pipeline with:

```bash
python scripts/run_analysis.py
```

This regenerates the main figures in `figures/`:

- `figure_1_prior_predictive_checks.png`
- `figure_2_posterior_predictive_checks.png`
- `figure_3_prior_posterior_update.png`
- `figure_4_alpha_recovery.png`
- `figure_5_beta_recovery.png`
- `figure_6_recovery_error_by_trials.png`

It also updates:

- `data/simulated_data.json`
- `data/recovery/recovery_results.csv`
- `data/recovery/recovery_summary.csv`

## Notes

- The current model estimates `alpha` and `beta`.
- The initial belief is fixed at `p0 = 0.5`.
- A free `p0` extension is possible, but it is not included in the current coursework version.
- On Windows, the helper logic in `scripts/stan_utils.py` attempts to automatically detect common RTools and CmdStan runtime paths, so manual `set PATH=...` commands should not be necessary in the intended environment.
