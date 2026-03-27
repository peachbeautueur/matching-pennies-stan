from pathlib import Path

from stan_utils import build_model


def main():
    data_file = "data/simulated_data.json"

    print("Compiling Stan model...")
    model = build_model()

    print("Sampling from model...")
    fit = model.sample(
        data=data_file,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        seed=123,
    )

    print(fit.summary())

    out_dir = Path("data/stan_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    fit.save_csvfiles(dir=str(out_dir))
    print("Stan fit finished and CSV files saved.")


if __name__ == "__main__":
    main()
