import time

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet
import rpy2.robjects as robjects
import typer
from additive_model import AdditiveModel
from anova_model import ANOVAModel
from cpsplines.fittings.fit_cpsplines import CPsplines
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold


def pima_results_by_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_col: str,
    k: int,
    x_range: dict[str, tuple[float, float]],
    r_script: str,
    method_add_scam: str,
    method_inter_scam: str,
    cutoff: float = 0.5,
):
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
    x_range: dict[str, tuple[float, float]]
        The prediction horizon of the covariates.
    r_script : str
        The path with the R code to execute the simulations in the packages cgam
        and scam.
    method_add_scam : str
        The name of the function in `r_script` used to execute the simulations
        with the additive scam package.
    method_inter_scam : str
        The name of the function in `r_script` used to execute the simulations
        with the interaction scam package.
    cutoff : float, optional
        The cutoff used to classify a prediction in the positive class.

    Returns
    -------
    pd.DataFrame
        The results for this fold. The DataFrame has the method names as indexes
        and the metrics accuracy, F1-score and execution times as columns.
    """

    metrics = ("accuracy", "f1-score", "precision", "recall", "Times")
    methods = ("Unconstrained", "cpsplines", "scam")

    # Initialize the output DataFrames
    results_add = pd.DataFrame(index=methods, columns=metrics)
    results_inter = pd.DataFrame(index=methods, columns=metrics)
    results_anova = pd.DataFrame(index=methods, columns=metrics)

    # Compute the metrics for the unconstrained and constrained interaction
    # with `cpsplines`
    for int_constrains, method in zip(
        [None, {"bmi": {1: {"+": 0.0}}, "pregnant": {1: {"+": 0.0}}}],
        ["Unconstrained", "cpsplines"],
    ):
        start = time.time()
        psplines_inter_fit = CPsplines(
            deg=(3, 3),
            ord_d=(2, 2),
            n_int=(k, k),
            x_range=x_range,
            family="binomial",
            sp_args={"method": "L-BFGS-B", "options": {"ftol": 1e-9}, "verbose": True},
            int_constraints=int_constrains,
        )
        _ = psplines_inter_fit.fit(data=X_train, y_col=y_col)
        end = time.time()
        y_pred = pd.Series(
            np.where(
                psplines_inter_fit.predict(X_test.drop(columns=y_col)) < cutoff,
                0,
                1,
            ).flatten()
        )
        results_inter.loc[method, :] = [
            accuracy_score(X_test[y_col], y_pred),
            f1_score(X_test[y_col], y_pred),
            precision_score(X_test[y_col], y_pred),
            recall_score(X_test[y_col], y_pred),
            end - start,
        ]

    # Compute the metrics for the unconstrained and constrained additive
    # with `cpsplines`
    for constrained, method in zip([True, False], ["Unconstrained", "cpsplines"]):
        start = time.time()
        psplines_add_fit = AdditiveModel(
            deg=(3, 3),
            ord_d=(2, 2),
            n_int=(k, k),
            x_range={k: v for k, v in zip(X_test.columns, x_range.values())},
            constrained=constrained,
        )
        _ = psplines_add_fit.fit(data=X_train, y_col=y_col)
        end = time.time()
        y_pred = pd.Series(
            np.where(
                psplines_add_fit.predict(X_test.drop(columns=y_col)) < cutoff,
                0,
                1,
            ).flatten()
        )
        results_add.loc[method, :] = [
            accuracy_score(X_test[y_col], y_pred),
            f1_score(X_test[y_col], y_pred),
            precision_score(X_test[y_col], y_pred),
            recall_score(X_test[y_col], y_pred),
            end - start,
        ]
    # Compute the metrics for the unconstrained and constrained ANOVA
    # with `cpsplines`
    for constrained, method in zip([True, False], ["Unconstrained", "cpsplines"]):
        start = time.time()
        psplines_anova_fit = ANOVAModel(
            deg=(3, 3),
            ord_d=(2, 2),
            n_int=(k, k),
            x_range={k: v for k, v in zip(X_test.columns, x_range.values())},
            constraint_concave=False,
            constraint_envelope=False,
            constraint_monotone=constrained,
            family="binomial",
        )
        _ = psplines_anova_fit.fit(data=X_train, y_col=y_col)
        end = time.time()
        y_pred = pd.Series(
            np.where(
                psplines_anova_fit.predict(X_test.drop(columns=y_col)) < cutoff,
                0,
                1,
            ).flatten()
        )
        results_anova.loc[method, :] = [
            accuracy_score(X_test[y_col], y_pred),
            f1_score(X_test[y_col], y_pred),
            precision_score(X_test[y_col], y_pred),
            recall_score(X_test[y_col], y_pred),
            end - start,
        ]
    r = robjects.r
    r["source"](r_script)
    # Loading the functions we have defined in R
    scam_add_eval_r = robjects.globalenv[method_add_scam]
    scam_inter_eval_r = robjects.globalenv[method_inter_scam]
    # Converting it into r object for passing into R function
    with localconverter(robjects.default_converter + pandas2ri.converter):
        X_train_r = robjects.conversion.py2rpy(X_train.astype(str))
        X_test_r = robjects.conversion.py2rpy(X_test.astype(str))
    # Invoking the R function and getting the result
    scam_add_fit = scam_add_eval_r(X_train_r, X_test_r, k)
    scam_inter_fit = scam_inter_eval_r(X_train_r, X_test_r)
    # Compute the metrics for the constrained additive and interaction models
    # with scam
    for fitting, df in zip(
        [scam_add_fit, scam_inter_fit], [results_add, results_inter]
    ):
        y_pred = pd.Series(
            np.where(
                np.array(eval("[" + fitting[0] + "]"), dtype=np.float32) < cutoff, 0, 1
            ).flatten()
        )
        df.loc["scam", :] = [
            accuracy_score(X_test[y_col], y_pred),
            f1_score(X_test[y_col], y_pred),
            precision_score(X_test[y_col], y_pred),
            recall_score(X_test[y_col], y_pred),
            float(fitting[1]),
        ]
    return results_add, results_inter, results_anova


def main(
    r_script: str = "scpanova/R_scripts.R",
    fdata: str = "data/pima.csv",
    fout: str = "data/pima_results.parquet",
    method_add_scam="scam_additive_pima",
    method_inter_scam="scam_interaction_pima",
):
    pima = pd.read_csv(fdata)
    feature_cols = ["pregnant", "bmi", "test"]
    y_col = "test"
    # Drop rows with zero values in the column "bmi"
    pima_clean = pima[feature_cols].query("bmi != 0").reset_index(drop=True)
    # Create a 10-fold partition on the original data set
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    # Compute the prediction ranges for each covariate
    x_range = {
        col: (pima_clean[col].min(), pima_clean[col].max())
        for col in feature_cols
        if col != y_col
    }

    L_add, L_inter, L_anova = [], [], []
    for train_index, test_index in kf.split(pima_clean):
        X_train = pima_clean.loc[train_index, feature_cols]
        X_test = pima_clean.loc[test_index, feature_cols]

        output = pima_results_by_fold(
            X_train=X_train,
            X_test=X_test,
            y_col=y_col,
            k=30,
            x_range=x_range,
            r_script=r_script,
            method_add_scam=method_add_scam,
            method_inter_scam=method_inter_scam,
        )
        L_add.append(output[0])
        L_inter.append(output[1])
        L_anova.append(output[2])
    table = pd.concat(
        (
            pd.concat(L_add, axis=1, keys=range(kf.n_splits)).stack(1).T,
            pd.concat(L_inter, axis=1, keys=range(kf.n_splits)).stack(1).T,
            pd.concat(L_anova, axis=1, keys=range(kf.n_splits)).stack(1).T,
        ),
        axis=1,
        keys=("Additive", "Interaction", "ANOVA"),
    )
    table = pyarrow.Table.from_pandas(table)
    pyarrow.parquet.write_table(table, fout)


if __name__ == "__main__":
    typer.run(main)
