# matching-pennies-stan

Single-agent belief learning model with softmax choice for matching pennies, implemented in Stan.

## Files

- `stan/belief_model.stan`: main Stan model
- `stan/belief_model_commented.stan`: line-by-line commented Stan version
- `scripts/simulate_data.py`: simulate example matching pennies data
- `scripts/fit_model.py`: fit the Stan model to a dataset
- `scripts/model_checks.py`: prior predictive, posterior predictive, and prior-posterior update checks
- `scripts/parameter_recovery.py`: parameter recovery across trial counts
- `scripts/run_analysis.py`: one-command workflow that generates all assignment figures
- `report/assignment_report.md`: assignment-ready report draft

## Basic workflow

```bash
python scripts/run_analysis.py
```

This generates six main figures in `figures/`:

- `figure_1_prior_predictive_checks.png`
- `figure_2_posterior_predictive_checks.png`
- `figure_3_prior_posterior_update.png`
- `figure_4_alpha_recovery.png`
- `figure_5_beta_recovery.png`
- `figure_6_recovery_error_by_trials.png`

## Notes

- The current model estimates `alpha` and `beta`.
- The initial belief is fixed at `p0 = 0.5`.
- A free `p0` extension is feasible, but it should be justified because it can be weakly identified in short tasks.
