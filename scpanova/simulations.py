import logging
import time
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet
import rpy2.robjects as robjects
import typer
from cpsplines.fittings.fit_cpsplines import CPsplines, NumericalError
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scpanova.anova_model import ANOVAModel

logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO
)


def generate_data(
    x_1: Tuple[Union[int, float], Union[int, float]],
    x_2: Tuple[Union[int, float], Union[int, float]],
    n: int,
    grid: bool = True,
    scenario: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Given two intervals `x_1` and `x_2`, generates a sample of points in the
    generated rectangle with size `n * n` and evaluates them at the surface in the
    corresponding scenario.

    Parameters
    ----------
    x_1 : Tuple[Union[int, float], Union[int, float]]
        The interval in the X_1-axis.
    x_2 : Tuple[Union[int, float], Union[int, float]]
        The interval in the X_2-axis.
    n : int
        The square root of the sample size.
    grid : bool, optional
        If True, `n` points will be drawn from uniform distributions U(`x_1`)
        and U(`x_2`) and then a grid data structure is built combining these
        positions. Otherwise, `n` * `n` samples are directly drawn from a
        uniform distribution over the rectangle `x_1` x `x_ 2`. By default True.
    scenario : int, optional
        The number of scenario, by default 1.
    seed : int, optional
        The random seed for reproducibilty, by default 0.

    Returns
    -------
    pd.DataFrame
        A DataFrame with three columns and `n` * `n` rows:
        - "x_1": The X_1-coordinates of the points in the sample.
        - "x_2": The X_2-coordinates of the points in the sample.
        - "y_": The evaluations of the points (x_1, x_2) at the surface.
    """
    np.random.seed(seed)
    if grid:
        x_1 = np.repeat(np.sort(np.random.uniform(*x_1, size=n)), n)
        x_2 = np.tile(np.sort(np.random.uniform(*x_2, size=n)), n)
    else:
        x_1 = np.random.uniform(*x_1, size=n * n)
        x_2 = np.random.uniform(*x_2, size=n * n)

    if scenario == 1:
        # Check [1] for the expression fo the surface
        y = (
            (x_2 + np.sin(6 * np.pi * x_2) / (6 * np.pi))
            * (1 + np.power(2 * x_1 - 1, 3))
            / 2
        )
    elif scenario == 2:
        y = np.exp(2 * x_2) / np.sin(np.pi * x_1 / 4)
    elif scenario == 3:
        y = np.sin(np.pi * x_1) * np.exp(8 * x_2) / (2 + np.exp(8 * x_2))
    elif scenario == 4:
        y = (np.sin(x_2 * np.pi / 2) + 1.6 * x_2) * np.tanh(x_1) / 4 - 2
    elif scenario == 5:
        y = (np.power(x_2, 3) + 1) / (1 + np.exp(-10 * x_1))
    elif scenario == 6:
        y = 0.1 * x_1 + 0.1 * x_2 + 0.3 * np.sqrt(x_1 * x_2)
    elif scenario == 7:
        y = 0.1 * x_1 + 0.1 * x_2 + 0.3 * np.power(x_1 * x_2, 1 / 3)
    elif scenario == 8:
        y = np.log(x_1) * np.log(x_2)
    elif scenario == 9:
        y = 5 * np.sin(x_1 * np.pi / 20) * np.sin(x_2 * np.pi / 20)
    else:
        raise ValueError("Scenario not implemented.")
    return pd.DataFrame({"x1": x_1, "x2": x_2, "y_": y})


def simulated_example_results(
    n_iter: int,
    n: int,
    k: int,
    x_1: Tuple[Union[int, float], Union[int, float]],
    x_2: Tuple[Union[int, float], Union[int, float]],
    grid: bool,
    sigma: Union[int, float],
    graph: Optional[Callable] = None,
    scenario: int = 1,
    seed: int = 0,
    family: str = "gaussian",
    int_constraints: dict = None,
    r_script: str = "R_scripts.R",
    method_name: str = "simulated_example_results_R",
) -> pd.DataFrame:
    """Performs a simulation study in shape-constrained regression, comparing
    the packages cpsplines, cgam and scam, together with the unconstrained
    P-splines.

    Parameters
    ----------
    n_iter : int
        The number of iterations.
    n : int
        The square root of the sample size.
    k : int
        The number of inner knots to be used in the construction of the basis.
    x_1 : Tuple[Union[int, float], Union[int, float]]
        The interval in the X_1-axis.
    x_2 : Tuple[Union[int, float], Union[int, float]]
        The interval in the X_2-axis.
    grid : bool, optional
        If True, `n` points will be drawn from uniform distributions U(`x_1`)
        and U(`x_2`) and then a grid data structure is built combining these
        positions. Otherwise, `n` * `n` samples are directly drawn from a
        uniform distribution over the rectangle `x_1` x `x_ 2`. By default True.
    sigma : Union[int, float]
        The standard deviation of the noise error term.
    graph : Optional[Callable], optional
        The theoretical surface in which the constraints are to be imposed. If
        None, the surfaces in `generate_data` are used. By default None.
    scenario : int, optional
        The number of scenario, by default 1.
    seed : int, optional
        The random seed for reproducibilty, by default 0.
    family : str, optional
        The distribution from the exponential family, by default "gaussian".
    int_constraints : dict, optional
        The shape constraints to be enforced. If None, unconstrained P-splines are
        fitted. By default, None.
    r_script : str, optional
        The path with the R code to execute the simulations in the packages cgam
        and scam, by default "simulated_example_R.R".
    method_name : str, optional
        The name of the function in `r_script` used to execute the simulations,
        by default "simulated_example_results_R".

    Returns
    -------
    pd.DataFrame
        The results of the simulations. The DataFrame has `n_iter` rows and 12
        columns. For each method, the following metrics are stored:
        - Theo_MAE: The Mean Absolute Error between the estimated surface and
        the theoretical surface.
        - Theo_MAE: The Mean Squared Error between the estimated surface and the
        theoretical surface.
        - Times: The execution time.
    """

    # Define the columns of the output DataFrame with the results
    metrics = ["Theo_MAE", "Theo_MSE", "Times"]
    out_columns = [
        "Theo_MAE_Unconstrained",
        "Theo_MSE_Unconstrained",
        "Times_Unconstrained",
        "Theo_MAE_cpsplines",
        "Theo_MSE_cpsplines",
        "Times_cpsplines",
        "Theo_MAE_scam",
        "Theo_MSE_scam",
        "Times_scam",
        "Theo_MAE_cgam",
        "Theo_MSE_cgam",
        "Times_cgam",
    ]

    # Define the theoretical surface
    if graph is None:
        graph = generate_data

    # Initialize the output DataFrame
    results = np.zeros((n_iter, len(out_columns)))

    for w in range(n_iter):
        df = graph(x_1=x_1, x_2=x_2, n=n, grid=grid, scenario=scenario, seed=w + seed)
        logging.info(f"Iteration: {w + seed}")
        # Add noise to the theoretical curve
        if family == "gaussian":
            np.random.seed(w + seed)
            error = np.random.normal(0, sigma, len(df))
            df["y"] = df["y_"]
            df["y_error"] = df["y"] + error
        elif family == "poisson":
            df["y"] = np.exp(df["y_"])
            np.random.seed(w + seed)
            error = np.random.poisson(df["y"], size=len(df))
            df["y_error"] = error
            assert df["y_error"].dtypes == np.int64
        elif family == "binomial":
            df["y"] = np.exp(df["y_"]) / (1 + np.exp(df["y_"]))
            np.random.seed(w + seed)
            error = np.random.binomial(1, df["y"], size=len(df))
            df["y_error"] = error
            assert df["y_error"].dtypes == np.int64
        else:
            raise ValueError("Family not implemented.")
        try:
            # Fit the data with unconstrained P-splines and update the results
            start = time.time()
            unconstrained_fit = CPsplines(
                deg=(3, 3),
                ord_d=(2, 2),
                n_int=(k, k),
                sp_method="optimizer",
                sp_args={"method": "L-BFGS-B", "options": {"ftol": 1e-9}},
                family=family,
            )
            _ = unconstrained_fit.fit(
                data=df.drop(columns=["y", "y_"]), y_col="y_error"
            )
            y_unconstrained = unconstrained_fit.predict(data=df[["x1", "x2"]])
            if family == "poisson":
                y_unconstrained = np.log(y_unconstrained)
            elif family == "binomial":
                y_unconstrained = np.log(y_unconstrained / (1 - y_unconstrained))
            end = time.time()

            results[w, : len(metrics)] = np.array(
                [
                    mean_absolute_error(y_unconstrained, df["y_"]),
                    mean_squared_error(y_unconstrained, df["y_"]),
                    end - start,
                ]
            )
            # Fit the data with double non-decreasing P-splines and update the
            # results
            start = time.time()
            cpsplines_fit = CPsplines(
                deg=(3, 3),
                ord_d=(2, 2),
                n_int=(k, k),
                sp_method="optimizer",
                sp_args={"method": "L-BFGS-B", "options": {"ftol": 1e-9}},
                int_constraints=int_constraints,
                family=family,
            )
            _ = cpsplines_fit.fit(data=df.drop(columns=["y", "y_"]), y_col="y_error")
            y_cpsplines = cpsplines_fit.predict(data=df[["x1", "x2"]])
            if family == "poisson":
                y_cpsplines = np.log(y_cpsplines)
            elif family == "binomial":
                y_cpsplines = np.log(y_cpsplines / (1 - y_cpsplines))
            end = time.time()
            results[w, len(metrics) : 2 * len(metrics)] = np.array(
                [
                    mean_absolute_error(y_cpsplines, df["y_"]),
                    mean_squared_error(y_cpsplines, df["y_"]),
                    end - start,
                ]
            )
            # Defining the R script and loading the instance in Python
            r = robjects.r
            r["source"](r_script)
            # Loading the function we have defined in R
            scam_cgam_eval_r = robjects.globalenv[method_name]
            # converting it into r object for passing into r function
            with localconverter(robjects.default_converter + pandas2ri.converter):
                df_r = robjects.conversion.py2rpy(df)
            # Invoking the R function and getting the result
            results[w, 2 * len(metrics) :] = np.asarray(
                scam_cgam_eval_r(df_r, k, family, scenario)
            )
            df = df.drop(columns=["y", "y_error"])
        except RRuntimeError or NumericalError:
            logging.warning(f"Iteration {w+1} has not reached to the expected solution")
            results[w, :] = np.array([np.nan] * len(out_columns))
    results_df = pd.DataFrame(data=results, columns=out_columns)
    return results_df.reindex(sorted(results_df.columns), axis=1).dropna()


def simulated_production_function(
    n_iter: int,
    n: int,
    k: int,
    scenario: int = 1,
    seed: int = 0,
    r_script: str = "production_function.R",
    method_name: str = "simulated_production_function_R",
) -> pd.DataFrame:
    """Performs a simulation study in estimating production functions, comparing the
    packages cpsplines (with ANOVA or not), AAFS, DEA and C2NLS.

    Parameters
    ----------
    n_iter : int
        The number of iterations.
    n : int
        The square root of the sample size.
    k : int
        The number of inner knots to be used in the construction of the basis.
    scenario : int, optional
        The number of scenario, by default 1.
    seed : int, optional
        The random seed for reproducibilty, by default 0.
    r_script : str, optional
        The path with the R code to execute the simulations in the packages cgam
        and scam, by default "simulated_example_R.R".
    method_name : str, optional
        The name of the function in `r_script` used to execute the simulations,
        by default "simulated_example_results_R".

    Returns
    -------
    pd.DataFrame
        The results of the simulations. The DataFrame has `n_iter` rows and 15
        columns. For each method, the following metrics are stored:
        - Theo_MAE: The Mean Absolute Error between the estimated surface and
        the theoretical surface.
        - Theo_MAE: The Mean Squared Error between the estimated surface and the
        theoretical surface.
        - Times: The execution time.
    """
    results = np.zeros((n_iter, 15))
    for w in range(n_iter):
        logging.info(f"Iteration: {w + seed}")
        data = pd.DataFrame(index=range(n))
        np.random.seed(w + seed)
        u = np.abs(np.random.normal(loc=0, scale=0.4, size=n))
        x1 = np.random.uniform(low=1, high=10, size=n)
        x2 = np.random.uniform(low=1, high=10, size=n)
        if scenario == 6:
            y = 0.1 * x1 + 0.1 * x2 + 0.3 * np.sqrt(x1 * x2)
        elif scenario == 7:
            y = 0.1 * x1 + 0.1 * x2 + 0.3 * np.power(x1 * x2, 1 / 3)
        elif scenario == 8:
            y = np.log(x1) * np.log(x2)
        elif scenario == 9:
            y = 5 * np.sin(x1 * np.pi / 20) * np.sin(x2 * np.pi / 20)
        else:
            raise ValueError("Scenario not implemented.")
        data = data.assign(**{"x1": x1, "x2": x2, "y": y - u, "yD": y})
        data.loc[:, "y"] = np.clip(y - u, a_min=0.01, a_max=None)
        data.loc[:, "yD"] = y

        start = time.time()
        cpsplines_fit = CPsplines(
            deg=(3, 3),
            ord_d=(2, 2),
            n_int=(k, k),
            int_constraints={
                "x1": {1: {"+": 0}, 2: {"-": 0}},
                "x2": {1: {"+": 0}, 2: {"-": 0}},
            },
            sp_method="optimizer",
            sp_args={"method": "L-BFGS-B", "options": {"ftol": 1e-9}},
            pt_constraints={(0, 0): {"greaterThan": data.drop(columns=["yD"])}},
        )
        _ = cpsplines_fit.fit(data=data.drop(columns=["yD"]), y_col="y")
        y_cpsplines = cpsplines_fit.predict(data=data[["x1", "x2"]])
        end = time.time()
        cpsplines_results = np.array(
            [
                mean_absolute_error(y_cpsplines, data["yD"]),
                mean_squared_error(y_cpsplines, data["yD"]),
                end - start,
            ]
        )

        np.set_printoptions(suppress=True)
        start = time.time()
        anova_fit = ANOVAModel(
            deg=(3, 3),
            ord_d=(2, 2),
            n_int=(k, k),
            sp_method="optimizer",
            sp_args={"method": "L-BFGS-B", "options": {"ftol": 1e-9}, "verbose": True},
        )
        _ = anova_fit.fit(data=data.drop(columns=["yD"]), y_col="y")
        print(f"Best smoothing parameters for ANOVA: {anova_fit.best_sp}")
        y_anova = anova_fit.predict(data=data[["x1", "x2"]])
        end = time.time()
        anova_results = np.array(
            [
                mean_absolute_error(y_anova, data["yD"]),
                mean_squared_error(y_anova, data["yD"]),
                end - start,
            ]
        )

        # Defining the R script and loading the instance in Python
        r = robjects.r
        r["source"](r_script)
        # Loading the function we have defined in R
        MLfrontiers_r = robjects.globalenv[method_name]
        # converting it into r object for passing into r function
        with localconverter(robjects.default_converter + pandas2ri.converter):
            data_r = robjects.conversion.py2rpy(data)
        # Invoking the R function and getting the result
        mlfrontiers_results = np.asarray(MLfrontiers_r(data_r))
        results[w,] = np.r_[cpsplines_results, anova_results, mlfrontiers_results]
    results = pd.DataFrame(
        results,
        columns=[
            "Theo_MAE_cpsplines",
            "Theo_MSE_cpsplines",
            "Times_cpsplines",
            "Theo_MAE_ANOVA",
            "Theo_MSE_ANOVA",
            "Times_ANOVA",
            "Theo_MAE_DEA",
            "Theo_MSE_DEA",
            "Times_DEA",
            "Theo_MAE_C2NLS",
            "Theo_MSE_C2NLS",
            "Times_C2NLS",
            "Theo_MAE_AAFS",
            "Theo_MSE_AAFS",
            "Times_AAFS",
        ],
    )
    return results


def main(
    r_script: str = "scpanova/R_scripts.R",
    seed: int = 0,
    scenario: int = 1,
    n_iter: int = 100,
):
    if scenario == 1:
        x_1 = (0, 1)
        x_2 = (0, 1)
        int_constraints = {"x1": {1: {"+": 0}}, "x2": {1: {"+": 0}}}
        family = "gaussian"
    elif scenario == 2:
        x_1 = (1, 3)
        x_2 = (-2, 0)
        int_constraints = {"x1": {2: {"+": 0}}, "x2": {2: {"+": 0}}}
        family = "gaussian"
    elif scenario == 3:
        x_1 = (0, 1)
        x_2 = (-1, 1)
        int_constraints = {
            "x1": {0: {"+": 0}, 2: {"-": 0}},
            "x2": {0: {"+": 0}, 1: {"+": 0}},
        }
        family = "gaussian"
    elif scenario == 4:
        x_1 = (0, 4)
        x_2 = (0, 8)
        int_constraints = {"x1": {1: {"+": 0}}, "x2": {1: {"+": 0}}}
        family = "poisson"
    elif scenario == 5:
        x_1 = (-1, 1)
        x_2 = (-1, 1)
        int_constraints = {"x1": {1: {"+": 0}}, "x2": {1: {"+": 0}}}
        family = "binomial"
    else:
        raise ValueError("Scenario not implemented.")

    sigmas = (0.05, 0.10, 0.25, 0.50) if family == "gaussian" else (1,)
    gridded = (True, False)

    L = []
    for grid in gridded:
        ls = []
        for sigma in sigmas:
            logging.info(f"Sigma: {sigma}")
            df = simulated_example_results(
                n_iter=n_iter,
                n=50,
                k=30,
                x_1=x_1,
                x_2=x_2,
                grid=grid,
                seed=seed,
                scenario=scenario,
                int_constraints=int_constraints,
                sigma=sigma,
                r_script=r_script,
                family=family,
            )
            ls.append(df)
        # Concatenate the DataFrame for common data structure
        df_ = pd.concat([pd.concat([s], axis=1).T for s in ls], axis=0, keys=sigmas)
        L.append(df_)
    # Concatenate DataFrames with gridded and scattered data
    out = pd.concat([pd.concat([s], axis=1).T for s in L], axis=1, keys=gridded)
    table = pyarrow.Table.from_pandas(out.T)
    pyarrow.parquet.write_table(
        table,
        f"data/simulated_example_results_{scenario}_{seed}_{family}.parquet",
    )


if __name__ == "__main__":
    typer.run(main)
