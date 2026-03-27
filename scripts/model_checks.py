import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stan_utils import build_model


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "simulated_data.json"
FIG_DIR = ROOT / "figures"


def load_data(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simulate_from_parameters(opp, alpha, beta, rng):
    t_max = len(opp)
    y = np.zeros(t_max, dtype=int)
    p = 0.5

    for t in range(t_max):
        v_h = 2 * p - 1
        v_t = 1 - 2 * p
        p_choose_h = 1.0 / (1.0 + np.exp(-beta * (v_h - v_t)))
        y[t] = rng.binomial(1, p_choose_h)
        p = p + alpha * (opp[t] - p)

    return y


def prior_predictive_check(data, draws=200, seed=123):
    rng = np.random.default_rng(seed)
    opp = np.asarray(data["opp"], dtype=int)

    choice_rates = []
    switch_rates = []

    for _ in range(draws):
        alpha = rng.beta(2, 2)
        beta = rng.lognormal(0, 1)
        y = simulate_from_parameters(opp, alpha, beta, rng)
        choice_rates.append(y.mean())
        switch_rates.append(np.mean(np.diff(y) != 0))

    return np.asarray(choice_rates), np.asarray(switch_rates)


def fit_posterior(data_file=DATA_PATH):
    model = build_model()
    return model.sample(
        data=str(data_file),
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        seed=123,
    )


def posterior_predictive_summary(fit, observed_y):
    draws = fit.stan_variable("y_rep")
    mean_choice_rate = draws.mean(axis=1)
    mean_switch_rate = np.mean(np.diff(draws, axis=1) != 0, axis=1)

    obs_choice_rate = np.mean(observed_y)
    obs_switch_rate = np.mean(np.diff(observed_y) != 0)

    return {
        "pp_choice_rate": mean_choice_rate,
        "pp_switch_rate": mean_switch_rate,
        "obs_choice_rate": obs_choice_rate,
        "obs_switch_rate": obs_switch_rate,
    }


def prior_posterior_update_summary(fit):
    alpha_post = fit.stan_variable("alpha")
    beta_post = fit.stan_variable("beta")

    rng = np.random.default_rng(123)
    alpha_prior = rng.beta(2, 2, size=len(alpha_post))
    beta_prior = rng.lognormal(0, 1, size=len(beta_post))

    return {
        "alpha_prior": alpha_prior,
        "alpha_post": alpha_post,
        "beta_prior": beta_prior,
        "beta_post": beta_post,
    }


def save_check_plots(prior_choice, prior_switch, post_pred, prior_post):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(prior_choice, bins=25, color="#6baed6", alpha=0.8)
    axes[0].set_title("Prior Predictive Choice Rate")
    axes[0].set_xlabel("Mean P(choose Heads)")
    axes[0].set_ylabel("Count")

    axes[1].hist(prior_switch, bins=25, color="#9ecae1", alpha=0.8)
    axes[1].set_title("Prior Predictive Switch Rate")
    axes[1].set_xlabel("Switch rate")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_1_prior_predictive_checks.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(post_pred["pp_choice_rate"], bins=25, color="#74c476", alpha=0.8)
    axes[0].axvline(post_pred["obs_choice_rate"], color="black", linestyle="--", label="Observed")
    axes[0].set_title("Posterior Predictive Choice Rate")
    axes[0].set_xlabel("Replicated mean choice rate")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(post_pred["pp_switch_rate"], bins=25, color="#31a354", alpha=0.8)
    axes[1].axvline(post_pred["obs_switch_rate"], color="black", linestyle="--", label="Observed")
    axes[1].set_title("Posterior Predictive Switch Rate")
    axes[1].set_xlabel("Replicated switch rate")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_2_posterior_predictive_checks.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(prior_post["alpha_prior"], bins=30, alpha=0.6, label="Prior", color="#9ecae1")
    axes[0].hist(prior_post["alpha_post"], bins=30, alpha=0.6, label="Posterior", color="#08519c")
    axes[0].set_title("Alpha Prior vs Posterior")
    axes[0].set_xlabel("alpha")
    axes[0].legend()

    axes[1].hist(prior_post["beta_prior"], bins=30, alpha=0.6, label="Prior", color="#a1d99b")
    axes[1].hist(prior_post["beta_post"], bins=30, alpha=0.6, label="Posterior", color="#006d2c")
    axes[1].set_title("Beta Prior vs Posterior")
    axes[1].set_xlabel("beta")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_3_prior_posterior_update.png", dpi=150)
    plt.close(fig)


def main():
    data = load_data()
    observed_y = np.asarray(data["y"], dtype=int)

    prior_choice, prior_switch = prior_predictive_check(data)
    fit = fit_posterior()
    post_pred = posterior_predictive_summary(fit, observed_y)
    prior_post = prior_posterior_update_summary(fit)

    save_check_plots(prior_choice, prior_switch, post_pred, prior_post)
    print("Saved figures 1-3 to figures/.")


if __name__ == "__main__":
    main()
