import typer

from scpanova.simulations import simulated_production_function


def main(seed: int = 0, scenario: int = 6, n_iter: int = 100, n: int = 150):
    df = simulated_production_function(
        n_iter=n_iter,
        n=n,
        k=10,
        scenario=scenario,
        seed=seed,
        r_script="scpanova/production_function.R",
    ).T.sort_index()
    df.columns = [str(col) for col in df.columns]
    df.to_parquet(f"data/production_results_{scenario}_{n}_{seed}.parquet")


if __name__ == "__main__":
    typer.run(main)
