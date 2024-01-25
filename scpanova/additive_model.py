from typing import Any, Iterable, Optional, Union

import mosek.fusion
import numpy as np
import pandas as pd
from cpsplines.fittings.fit_cpsplines import CPsplines
from cpsplines.mosek_functions.interval_constraints import IntConstraints
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.families.family import Binomial
from statsmodels.tools.tools import add_constant


def fit_irls(
    y: np.ndarray,
    B: np.ndarray,
    penalty_term: np.ndarray,
    tol: Union[int, float] = 1e-8,
    maxiter: int = 100,
) -> np.ndarray:
    theta_old = np.zeros(penalty_term.shape[0])
    mu = Binomial().starting_mu(y)
    eta = Binomial().predict(mu)

    for _ in range(maxiter):
        W = Binomial().weights(mu)
        Z = eta + Binomial().link.deriv(mu) * (y - mu)
        theta = np.linalg.solve(
            B.T @ np.diag(W) @ B + penalty_term, B.T @ np.multiply(W, Z)
        )

        eta = np.dot(B, theta)
        mu = Binomial().fitted(eta)
        if np.linalg.norm(theta - theta_old) < tol:
            break
        theta_old = theta.copy()
    return mu


def GCV(
    sp: Iterable[Union[int, float]],
    obj_matrices: dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    B: np.ndarray,
) -> float:
    penalty_term = [
        np.multiply(s, M.T @ D.T @ D @ M)
        for D, M, s in zip(obj_matrices["D"], obj_matrices["M"], sp)
    ]
    penalty_term = block_diag(*[0] + penalty_term)

    y_hat = fit_irls(y=obj_matrices["y"], B=B, penalty_term=penalty_term)
    n = B.shape[0]
    bases_term = B.T @ np.diag(Binomial().weights(Binomial().starting_mu(y_hat))) @ B

    return (n * Binomial().deviance(obj_matrices["y"], y_hat)) / np.square(
        n - np.trace(np.linalg.solve(bases_term + penalty_term, bases_term))
    )


class AdditiveModel(CPsplines):
    def __init__(
        self,
        deg: Iterable[int] = (3, 3),
        ord_d: Iterable[int] = (2, 2),
        n_int: Iterable[int] = (20, 20),
        x_range: Optional[dict[str, tuple[float, float]]] = None,
        sp_method: str = "optimizer",
        sp_args: Optional[dict[str, Any]] = None,
        constrained: bool = False,
    ):
        super(AdditiveModel, self).__init__(
            deg=deg,
            ord_d=ord_d,
            n_int=n_int,
            x_range=x_range,
            sp_method=sp_method,
            sp_args=sp_args,
        )
        self.constrained = constrained

    def _get_obj_func_arrays(
        self, y: np.ndarray
    ) -> Union[np.ndarray, Iterable[np.ndarray], StandardScaler]:
        matrices = super(AdditiveModel, self)._get_obj_func_arrays(y=y)

        scalers, B_list, M_list = [], [], []
        for B in matrices["B"]:
            scaler = StandardScaler(with_std=False)
            scaler.fit(B)
            B_list.append(scaler.transform(B))
            scalers.append(scaler)
            M_list.append(
                np.vstack([np.eye(B.shape[1] - 1), np.repeat(-1, B.shape[1] - 1)])
            )
        matrices.update({"B": B_list, "M": M_list, "scaler": scalers})
        return matrices

    def _initialize_model(self) -> mosek.fusion.Model:
        B = add_constant(
            np.concatenate(
                [B @ M for B, M in zip(self.obj_matrices["B"], self.obj_matrices["M"])],
                axis=1,
            )
        )
        y = self.obj_matrices["y"]
        M = mosek.fusion.Model()
        # Define the regression coefficients decision variables
        theta = M.variable("theta", B.shape[1], mosek.fusion.Domain.unbounded())

        s = M.variable("s", len(y), mosek.fusion.Domain.greaterThan(0.0))
        u = M.variable("u", len(y), mosek.fusion.Domain.greaterThan(0.0))
        v = M.variable("v", len(y), mosek.fusion.Domain.greaterThan(0.0))

        t_D = M.variable("t_D", len(self.deg), mosek.fusion.Domain.greaterThan(0.0))
        sp = M.parameter("sp", len(self.deg))

        count = 1
        for g, (D, M_) in enumerate(
            zip(self.obj_matrices["D"], self.obj_matrices["M"])
        ):
            M.constraint(
                f"rot_cone_D_{g}",
                mosek.fusion.Expr.vstack(
                    t_D.slice(g, g + 1),
                    1 / 2,
                    mosek.fusion.Expr.mul(
                        mosek.fusion.Matrix.sparse(D @ M_),
                        theta.slice(count, count + M_.shape[1]),
                    ),
                ),
                mosek.fusion.Domain.inRotatedQCone(),
            )
            count += M_.shape[1]

        M.constraint(
            mosek.fusion.Expr.hstack(
                u,
                mosek.fusion.Expr.constTerm(len(y), 1.0),
                mosek.fusion.Expr.sub(mosek.fusion.Expr.mul(B, theta), s),
            ),
            mosek.fusion.Domain.inPExpCone(),
        )

        M.constraint(
            mosek.fusion.Expr.hstack(
                v,
                mosek.fusion.Expr.constTerm(len(y), 1.0),
                mosek.fusion.Expr.mul(-1, s),
            ),
            mosek.fusion.Domain.inPExpCone(),
        )

        M.constraint(mosek.fusion.Expr.add(u, v), mosek.fusion.Domain.lessThan(1.0))

        obj = mosek.fusion.Expr.add(
            mosek.fusion.Expr.sub(
                mosek.fusion.Expr.sum(s),
                mosek.fusion.Expr.dot(np.dot(y, B), theta),
            ),
            mosek.fusion.Expr.dot(sp, t_D),
        )
        M.objective("obj", mosek.fusion.ObjectiveSense.Minimize, obj)

        return M

    def _get_sp_optimizer(self) -> float:
        B = add_constant(
            np.concatenate(
                [B @ M for B, M in zip(self.obj_matrices["B"], self.obj_matrices["M"])],
                axis=1,
            )
        )
        best_sp = minimize(
            GCV,
            x0=np.ones((len(self.deg), 1)),
            args=(self.obj_matrices, B),
            method="L-BFGS-B",
            bounds=[(1e-10, 1e16) for _ in range(len(self.deg))],
            options={"ftol": 1e-9},
        ).x
        return best_sp

    def _set_constraints(self, model: mosek.fusion.Model) -> mosek.fusion.Model:
        count = 1
        for name, bsp, M, d in zip(
            self.feature_names, self.bspline_bases, self.obj_matrices["M"], self.deg
        ):
            _ = bsp.get_matrices_S()
            S = bsp.matrices_S
            S_list = []
            for i, S_ in enumerate(S):
                S_ = np.c_[
                    np.zeros((d + 1, i)), S_, np.zeros((d + 1, M.shape[0] - d - 1 - i))
                ]
                S_list.append((S_ @ M)[1:, :])

            int_cons = IntConstraints(
                bspline={name: bsp}, var_name=name, derivative=1, constraints={"+": 0.0}
            )
            theta_ = model.getVariable("theta").slice(count, count + M.shape[1])
            W = int_cons._get_matrices_W()
            C = [w @ s for w, s in zip(W, S_list)]
            H = int_cons._get_matrices_H()
            X = model.variable(mosek.fusion.Domain.inPSDCone(d, len(S)))

            for j, C_ in enumerate(C):
                X_ = X.slice([j, 0, 0], [j + 1, d, d]).reshape([d, d])
                for k in range(d - 1):
                    model.constraint(
                        mosek.fusion.Expr.dot(H[k], X_),
                        mosek.fusion.Domain.equalsTo(0.0),
                    )

                for k in range(d - 1, 2 * d - 1):
                    model.constraint(
                        mosek.fusion.Expr.sub(
                            mosek.fusion.Expr.mul(C_, theta_).slice(
                                k + 1 - d, k + 2 - d
                            ),
                            mosek.fusion.Expr.dot(H[k], X_),
                        ),
                        mosek.fusion.Domain.equalsTo(0.0),
                    )
            count += M.shape[1]
        return model

    def fit(self, data: pd.DataFrame, y_col: str):
        self.feature_names = data.drop(columns=y_col).columns
        self.data_hull = Delaunay(data.loc[:, self.feature_names])

        x = [data[col].values for col in data.drop(columns=y_col)]
        y = data[y_col].values

        self.bspline_bases = self._get_bspline_bases(x=x)
        self.obj_matrices = self._get_obj_func_arrays(y=y)

        M = self._initialize_model()

        if self.constrained:
            M = self._set_constraints(model=M)

        self.best_sp = self._get_sp_optimizer()
        _ = M.getParameter("sp").setValue(self.best_sp)

        try:
            M.solve()
            self.sol = M.getVariable("theta").level()
        except mosek.fusion.SolutionError as e:
            raise ValueError(f"The original error was {e}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        x = [row for row in data.values.T]
        B_predict = add_constant(
            np.concatenate(
                [
                    scaler.transform(bsp.bspline_basis(x=x_)) @ M
                    for x_, bsp, scaler, M in zip(
                        x,
                        self.bspline_bases,
                        self.obj_matrices["scaler"],
                        self.obj_matrices["M"],
                    )
                ],
                axis=1,
            ),
            has_constant="add",
        )
        return Binomial().fitted(B_predict @ self.sol)
