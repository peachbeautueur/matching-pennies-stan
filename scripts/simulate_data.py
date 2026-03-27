import json
import numpy as np
from pathlib import Path

def simulate_matching_pennies(T=200, alpha=0.3, beta=3.0, seed=42):
    rng = np.random.default_rng(seed)

    opp = np.zeros(T, dtype=int)
    opp[0] = rng.integers(0, 2)

    for t in range(1, T):
        if rng.random() < 0.7:
            opp[t] = opp[t - 1]
        else:
            opp[t] = 1 - opp[t - 1]

    y = np.zeros(T, dtype=int)

    p = 0.5
    for t in range(T):
        vH = 2 * p - 1
        vT = 1 - 2 * p
        p_choose_H = 1 / (1 + np.exp(-beta * (vH - vT)))

        y[t] = rng.binomial(1, p_choose_H)

        p = p + alpha * (opp[t] - p)

    return {
        "T": int(T),
        "y": y.tolist(),
        "opp": opp.tolist(),
        "true_alpha": alpha,
        "true_beta": beta
    }

if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    data = simulate_matching_pennies(T=200, alpha=0.3, beta=3.0, seed=42)

    with open(out_dir / "simulated_data.json", "w") as f:
        json.dump(
            {"T": data["T"], "y": data["y"], "opp": data["opp"]},
            f,
            indent=2
        )

    print("Saved data/simulated_data.json")
    print(f"True alpha = {data['true_alpha']}, true beta = {data['true_beta']}")