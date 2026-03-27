import json
from pathlib import Path

import numpy as np

from model_checks import (
    fit_posterior,
    load_data,
    posterior_predictive_summary,
    prior_posterior_update_summary,
    prior_predictive_check,
    save_check_plots,
)
from parameter_recovery import run_recovery, save_recovery_plots, summarize_recovery
from simulate_data import simulate_matching_pennies


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def save_simulated_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = simulate_matching_pennies(T=200, alpha=0.3, beta=3.0, seed=42)
    with open(DATA_DIR / "simulated_data.json", "w", encoding="utf-8") as f:
        json.dump({"T": data["T"], "y": data["y"], "opp": data["opp"]}, f, indent=2)


def main():
    save_simulated_data()

    data = load_data()
    observed_y = np.asarray(data["y"], dtype=int)
    prior_choice, prior_switch = prior_predictive_check(data)
    fit = fit_posterior()
    post_pred = posterior_predictive_summary(fit, observed_y)
    prior_post = prior_posterior_update_summary(fit)
    save_check_plots(prior_choice, prior_switch, post_pred, prior_post)

    results = run_recovery()
    summary = summarize_recovery(results)
    save_recovery_plots(results, summary)

    print("Saved assignment figures 1-6 to figures/.")


if __name__ == "__main__":
    main()
