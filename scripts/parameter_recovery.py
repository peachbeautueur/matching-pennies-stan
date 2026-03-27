import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stan_utils import build_model


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "recovery"
FIG_DIR = ROOT / "figures"


def simulate_dataset(t_max, alpha, beta, seed):
    rng = np.random.default_rng(seed)

    opp = np.zeros(t_max, dtype=int)
    opp[0] = rng.integers(0, 2)
    for t in range(1, t_max):
        opp[t] = opp[t - 1] if rng.random() < 0.7 else 1 - opp[t - 1]

    y = np.zeros(t_max, dtype=int)
    p = 0.5
    for t in range(t_max):
        v_h = 2 * p - 1
        v_t = 1 - 2 * p
        p_choose_h = 1.0 / (1.0 + np.exp(-beta * (v_h - v_t)))
        y[t] = rng.binomial(1, p_choose_h)
        p = p + alpha * (opp[t] - p)

    return {"T": int(t_max), "y": y.tolist(), "opp": opp.tolist()}


def fit_once(model, dataset, seed, iter_warmup=500, iter_sampling=500):
    fit = model.sample(
        data=dataset,
        chains=4,
        parallel_chains=4,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=seed,
        show_progress=False,
    )
    return {
        "alpha_hat": float(np.mean(fit.stan_variable("alpha"))),
        "beta_hat": float(np.mean(fit.stan_variable("beta"))),
    }


def run_recovery(
    trial_counts=(50, 100, 200, 400),
    datasets_per_condition=3,
    alpha_grid=(0.2, 0.5, 0.8),
    beta_grid=(1.0, 3.0, 6.0),
    iter_warmup=500,
    iter_sampling=500,
):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model()
    rows = []
    seed = 100

    for t_max in trial_counts:
        for alpha in alpha_grid:
            for beta in beta_grid:
                for rep in range(datasets_per_condition):
                    dataset = simulate_dataset(t_max, alpha, beta, seed)
                    with open(
                        DATA_DIR / f"sim_T{t_max}_a{alpha}_b{beta}_r{rep}.json",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(dataset, f, indent=2)
                    estimates = fit_once(
                        model,
                        dataset,
                        seed,
                        iter_warmup=iter_warmup,
                        iter_sampling=iter_sampling,
                    )
                    rows.append(
                        {
                            "T": t_max,
                            "rep": rep,
                            "alpha_true": alpha,
                            "beta_true": beta,
                            **estimates,
                        }
                    )
                    seed += 1

    with open(DATA_DIR / "recovery_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["T", "rep", "alpha_true", "beta_true", "alpha_hat", "beta_hat"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def summarize_recovery(results):
    grouped = {}
    for row in results:
        grouped.setdefault(row["T"], {"alpha_abs_error": [], "beta_abs_error": []})
        grouped[row["T"]]["alpha_abs_error"].append(abs(row["alpha_hat"] - row["alpha_true"]))
        grouped[row["T"]]["beta_abs_error"].append(abs(row["beta_hat"] - row["beta_true"]))

    summary = []
    for t_max in sorted(grouped):
        summary.append(
            {
                "T": t_max,
                "alpha_abs_error": float(np.mean(grouped[t_max]["alpha_abs_error"])),
                "beta_abs_error": float(np.mean(grouped[t_max]["beta_abs_error"])),
            }
        )

    with open(DATA_DIR / "recovery_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["T", "alpha_abs_error", "beta_abs_error"],
        )
        writer.writeheader()
        writer.writerows(summary)

    return summary


def save_recovery_plots(results, summary):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    alpha_true = np.asarray([row["alpha_true"] for row in results], dtype=float)
    alpha_hat = np.asarray([row["alpha_hat"] for row in results], dtype=float)
    beta_true = np.asarray([row["beta_true"] for row in results], dtype=float)
    beta_hat = np.asarray([row["beta_hat"] for row in results], dtype=float)
    trials = np.asarray([row["T"] for row in summary], dtype=int)
    alpha_error = np.asarray([row["alpha_abs_error"] for row in summary], dtype=float)
    beta_error = np.asarray([row["beta_abs_error"] for row in summary], dtype=float)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(alpha_true, alpha_hat, alpha=0.7, color="#3182bd")
    lims = [0, 1]
    ax.plot(lims, lims, linestyle="--", color="black")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Alpha Recovery")
    ax.set_xlabel("True alpha")
    ax.set_ylabel("Recovered alpha")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_4_alpha_recovery.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    max_beta = max(beta_true.max(), beta_hat.max())
    ax.scatter(beta_true, beta_hat, alpha=0.7, color="#31a354")
    ax.plot([0, max_beta], [0, max_beta], linestyle="--", color="black")
    ax.set_title("Beta Recovery")
    ax.set_xlabel("True beta")
    ax.set_ylabel("Recovered beta")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_5_beta_recovery.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(trials, alpha_error, marker="o", color="#08519c")
    axes[0].set_title("Alpha Error by Trial Count")
    axes[0].set_xlabel("Trials")
    axes[0].set_ylabel("Mean absolute error")

    axes[1].plot(trials, beta_error, marker="o", color="#006d2c")
    axes[1].set_title("Beta Error by Trial Count")
    axes[1].set_xlabel("Trials")
    axes[1].set_ylabel("Mean absolute error")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_6_recovery_error_by_trials.png", dpi=150)
    plt.close(fig)


def main():
    results = run_recovery()
    summary = summarize_recovery(results)
    save_recovery_plots(results, summary)
    print("Saved figures 4-6 and recovery tables to data/recovery and figures/.")


if __name__ == "__main__":
    main()
