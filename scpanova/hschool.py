import time

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet
import rpy2.robjects as robjects
import typer
from cpsplines.fittings.fit_cpsplines import CPsplines
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold


def hschool_results_by_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_col: str,
    k: int,
    r_script: str,
    method_name_scam: str,
    method_name_cgam: str,
) -> pd.DataFrame:
    """Performs an estimation procedure of a surface that fits `X_train` and
    computes the error metrics and the computation time on the test partition
    `X_test`.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training partition.
    X_test : pd.DataFrame
        The test partition.
    y_col : str
        The name of the response variable.
    k : int
        The number of inner knots to be used in the construction of the basis.
    r_script : str
        The path with the R code to execute the simulations in the packages cgam
        and scam.
    method_name_scam : str
        The name of the function in `r_script` used to execute the simulations
        with the scam package.
    method_name_cgam : str
        The name of the function in `r_script` used to execute the simulations
        with the cgam package.

    Returns
    -------
    pd.DataFrame
        The results for this fold. The DataFrame has the method names as indexes
        and the metrics MAE, MSE and execution times as columns.
    """
    metrics = ("MAE", "MSE", "Times")
    methods = ["Unconstrained", "cpsplines", "cgam", "scam"]
    # Initialize the output DataFrame
    results = pd.DataFrame(index=methods, columns=metrics)

    start = time.time()
    # Fit the data with unconstrained P-splines and update the results
    unconstrained_fit = CPsplines(
        deg=(3, 3),
        ord_d=(2, 2),
        n_int=(k, k),
        family="poisson",
        sp_args={"method": "L-BFGS-B"},
    )
    _ = unconstrained_fit.fit(data=X_train, y_col=y_col)
    y_pred = unconstrained_fit.predict(X_test.drop(columns=y_col))
    end = time.time()
    results.loc["Unconstrained", :] = np.array(
        [
            mean_absolute_error(X_test[y_col], y_pred),
            mean_squared_error(X_test[y_col], y_pred),
            end - start,
        ]
    )
    start = time.time()
    # Fit the data with double non-increasing P-splines and update the results
    cpsplines_fit = CPsplines(
        deg=(3, 3),
        ord_d=(2, 2),
        n_int=(k, k),
        family="poisson",
        int_constraints={"math": {1: {"-": 0.0}}, "langarts": {1: {"-": 0.0}}},
        sp_args={"method": "L-BFGS-B"},
    )
    _ = cpsplines_fit.fit(data=X_train, y_col="daysabs")
    y_pred = cpsplines_fit.predict(X_test.drop(columns=y_col))
    end = time.time()
    results.loc["cpsplines", :] = np.array(
        [
            mean_absolute_error(X_test[y_col], y_pred),
            mean_squared_error(X_test[y_col], y_pred),
            end - start,
        ]
    )
    r = robjects.r
    r["source"](r_script)
    # Loading the functions we have defined in R
    scam_eval_r = robjects.globalenv[method_name_scam]
    cgam_eval_r = robjects.globalenv[method_name_cgam]
    # Converting it into r object for passing into R function
    with localconverter(robjects.default_converter + pandas2ri.converter):
        X_train_r = robjects.conversion.py2rpy(X_train.astype(str))
        X_test_r = robjects.conversion.py2rpy(X_test.astype(str))
    # Invoking the R function and getting the result
    results.loc["scam", :] = np.asarray(scam_eval_r(X_train_r, X_test_r)).astype(
        np.float32
    )
    results.loc["cgam", :] = np.asarray(cgam_eval_r(X_train_r, X_test_r, k)).astype(
        np.float32
    )
    return results


def main(
    r_script: str = "scpanova/R_scripts.R",
    fdata: str = "data/hschool.csv",
    fout: str = "data/hschool_results.parquet",
    method_name_scam="scam_hschool",
    method_name_cgam="cgam_hschool",
):
    hschool = pd.read_csv(fdata)
    # Create a 10-fold partition on the original data set
    kf = KFold(n_splits=2, random_state=None, shuffle=False)
    feature_cols = ["math", "langarts", "daysabs"]
    y_col = "daysabs"

    # cgam struggles predicting on points at the boundaries of the convex hull
    # of the data, so all these points will be in the training
    extreme_indexes = np.flatnonzero(
        pd.concat(
            [
                hschool[col].isin(hschool[col].agg(["min", "max"]))
                for col in ["math", "langarts"]
            ],
            axis=1,
        ).sum(axis=1)
    )

    L = []
    for train_index, test_index in kf.split(hschool):
        # Update the train/test indexes based on the previous requirement
        must_be_index = np.array(
            [idx for idx in extreme_indexes if idx not in train_index]
        )
        train_index = np.sort(np.concatenate((must_be_index, train_index)))
        test_index = np.sort(np.setdiff1d(test_index, must_be_index))
        X_train = hschool.loc[train_index, feature_cols]
        X_test = hschool.loc[test_index, feature_cols]

        L.append(
            hschool_results_by_fold(
                X_train=X_train,
                X_test=X_test,
                y_col=y_col,
                k=10,
                r_script=r_script,
                method_name_scam=method_name_scam,
                method_name_cgam=method_name_cgam,
            )
        )
    table = pd.concat(L, axis=1, keys=range(kf.n_splits)).stack(1).T
    table = pyarrow.Table.from_pandas(table)
    pyarrow.parquet.write_table(table, fout)


if __name__ == "__main__":
    typer.run(main)
