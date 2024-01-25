from typing import Any, Iterable, Optional, Union

import mosek.fusion
import numpy as np
import pandas as pd
from cpsplines.fittings.fit_cpsplines import CPsplines, IntConstraints
from cpsplines.utils.box_product import box_product
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.families.family import Binomial, Gaussian
from statsmodels.tools.tools import add_constant


def fit_irls(
    y: np.ndarray,
    B: np.ndarray,
    penalty_term: np.ndarray,
    family: Union[Gaussian, Binomial],
    tol: Union[int, float] = 1e-8,
    maxiter: int = 100,
) -> np.ndarray:
    theta_old = np.zeros(penalty_term.shape[0])
    mu = family.starting_mu(y)
    eta = family.predict(mu)

    for _ in range(maxiter):
        W = family.weights(mu)
        Z = eta + family.link.deriv(mu) * (y - mu)
        theta = np.linalg.solve(
            B.T @ np.diag(W) @ B + penalty_term, B.T @ np.multiply(W, Z)
        )

        eta = np.dot(B, theta)
        mu = family.fitted(eta)
        if np.linalg.norm(theta - theta_old) < tol:
            break
        theta_old = theta.copy()
    return mu


def GCV(
    sp: Iterable[Union[int, float]],
    obj_matrices: dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    family: Union[Gaussian, Binomial],
) -> float:
    C = block_diag(
        *[
            1,
            obj_matrices["M"][0],
            obj_matrices["M"][1],
            np.kron(obj_matrices["M"][0], obj_matrices["M"][1]),
        ]
    )

    B = (
        add_constant(
            np.concatenate(
                [
                    obj_matrices["B"][0],
                    obj_matrices["B"][1],
                    box_product(obj_matrices["B"][0], obj_matrices["B"][1]),
                ],
                axis=1,
            )
        )
        @ C
    )

    penalty_term = (
        C.T
        @ block_diag(
            *[
                0,
                sp[0] * obj_matrices["D_mul"][0],
                sp[1] * obj_matrices["D_mul"][1],
                sp[2]
                * np.kron(
                    obj_matrices["D_mul"][0], np.eye(obj_matrices["M"][1].shape[0])
                )
                + sp[3]
                * np.kron(
                    np.eye(obj_matrices["M"][0].shape[0]), obj_matrices["D_mul"][1]
                ),
            ]
        )
        @ C
    )

    y_hat = fit_irls(y=obj_matrices["y"], B=B, penalty_term=penalty_term, family=family)
    n = B.shape[0]
    bases_term = B.T @ np.diag(family.weights(family.starting_mu(y_hat))) @ B

    return (n * family.deviance(obj_matrices["y"], y_hat)) / np.square(
        n - np.trace(np.linalg.solve(bases_term + penalty_term, bases_term))
    )


class ANOVAModel(CPsplines):
    def __init__(
        self,
        deg: Iterable[int] = (3, 3),
        ord_d: Iterable[int] = (2, 2),
        n_int: Iterable[int] = (20, 20),
        x_range: Optional[dict[str, tuple[float, float]]] = None,
        sp_method: str = "optimizer",
        sp_args: Optional[dict[str, Any]] = None,
        family: str = "gaussian",
        constraint_envelope: bool = True,
        constraint_monotone: bool = True,
        constraint_concave: bool = True,
    ):
        super(ANOVAModel, self).__init__(
            deg=deg,
            ord_d=ord_d,
            n_int=n_int,
            x_range=x_range,
            sp_method=sp_method,
            sp_args=sp_args,
            family=family,
        )
        self.constraint_envelope = constraint_envelope
        self.constraint_monotone = constraint_monotone
        self.constraint_concave = constraint_concave

    def _get_obj_func_arrays(
        self, y: np.ndarray
    ) -> Union[np.ndarray, Iterable[np.ndarray], StandardScaler]:
        matrices = super(ANOVAModel, self)._get_obj_func_arrays(y=y)

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
                [
                    self.obj_matrices["B"][0] @ self.obj_matrices["M"][0],
                    self.obj_matrices["B"][1] @ self.obj_matrices["M"][1],
                    box_product(
                        self.obj_matrices["B"][0] @ self.obj_matrices["M"][0],
                        self.obj_matrices["B"][1] @ self.obj_matrices["M"][1],
                    ),
                ],
                axis=1,
            )
        )
        y = self.obj_matrices["y"]
        M = mosek.fusion.Model()
        # Define the regression coefficients decision variables
        theta = M.variable("theta", B.shape[1], mosek.fusion.Domain.unbounded())

        t_D = M.variable("t_D", 4, mosek.fusion.Domain.greaterThan(0.0))
        sp = M.parameter("sp", 4)

        M.constraint(
            f"rot_cone_D_{1}",
            mosek.fusion.Expr.vstack(
                t_D.slice(0, 1),
                1 / 2,
                mosek.fusion.Expr.mul(
                    mosek.fusion.Matrix.sparse(
                        self.obj_matrices["D"][0] @ self.obj_matrices["M"][0]
                    ),
                    theta.slice(1, self.obj_matrices["M"][0].shape[1] + 1),
                ),
            ),
            mosek.fusion.Domain.inRotatedQCone(),
        )

        M.constraint(
            f"rot_cone_D_{2}",
            mosek.fusion.Expr.vstack(
                t_D.slice(1, 2),
                1 / 2,
                mosek.fusion.Expr.mul(
                    mosek.fusion.Matrix.sparse(
                        self.obj_matrices["D"][1] @ self.obj_matrices["M"][1]
                    ),
                    theta.slice(
                        self.obj_matrices["M"][0].shape[1] + 1,
                        self.obj_matrices["M"][0].shape[1]
                        + self.obj_matrices["M"][1].shape[1]
                        + 1,
                    ),
                ),
            ),
            mosek.fusion.Domain.inRotatedQCone(),
        )

        M.constraint(
            f"rot_cone_D_{3}",
            mosek.fusion.Expr.vstack(
                t_D.slice(2, 3),
                1 / 2,
                mosek.fusion.Expr.mul(
                    mosek.fusion.Matrix.sparse(
                        np.kron(
                            self.obj_matrices["D"][0] @ self.obj_matrices["M"][0],
                            self.obj_matrices["M"][1],
                        )
                    ),
                    theta.slice(
                        self.obj_matrices["M"][0].shape[1]
                        + self.obj_matrices["M"][1].shape[1]
                        + 1,
                        B.shape[1],
                    ),
                ),
            ),
            mosek.fusion.Domain.inRotatedQCone(),
        )

        M.constraint(
            f"rot_cone_D_{4}",
            mosek.fusion.Expr.vstack(
                t_D.slice(3, 4),
                1 / 2,
                mosek.fusion.Expr.mul(
                    mosek.fusion.Matrix.sparse(
                        np.kron(
                            self.obj_matrices["M"][0],
                            self.obj_matrices["D"][1] @ self.obj_matrices["M"][1],
                        )
                    ),
                    theta.slice(
                        self.obj_matrices["M"][0].shape[1]
                        + self.obj_matrices["M"][1].shape[1]
                        + 1,
                        B.shape[1],
                    ),
                ),
            ),
            mosek.fusion.Domain.inRotatedQCone(),
        )

        if isinstance(self.family, Gaussian):
            t = M.variable("t", 1, mosek.fusion.Domain.greaterThan(0.0))
            M.constraint(
                f"rot_cone_B",
                mosek.fusion.Expr.vstack(
                    t,
                    1 / 2,
                    mosek.fusion.Expr.sub(
                        y,
                        mosek.fusion.Expr.mul(
                            mosek.fusion.Matrix.sparse(B),
                            theta,
                        ),
                    ),
                ),
                mosek.fusion.Domain.inRotatedQCone(),
            )
            obj = mosek.fusion.Expr.add(t, mosek.fusion.Expr.dot(sp, t_D))
        elif isinstance(self.family, Binomial):
            s = M.variable("s", len(y), mosek.fusion.Domain.greaterThan(0.0))
            u = M.variable("u", len(y), mosek.fusion.Domain.greaterThan(0.0))
            v = M.variable("v", len(y), mosek.fusion.Domain.greaterThan(0.0))

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
        else:
            raise ValueError("Not implemented.")

        M.objective("obj", mosek.fusion.ObjectiveSense.Minimize, obj)

        return M

    def _set_additive_constraints(
        self,
        model: mosek.fusion.Model,
        concave: bool = True,
        monotone: bool = True,
    ) -> mosek.fusion.Model:
        count = 1
        for v, (name, bsp, M, d) in enumerate(
            zip(
                self.feature_names, self.bspline_bases, self.obj_matrices["M"], self.deg
            )
        ):
            _ = bsp.get_matrices_S()
            S = bsp.matrices_S
            S_list = []
            for i, S_ in enumerate(S):
                S_ = np.c_[
                    np.zeros((d + 1, i)), S_, np.zeros((d + 1, M.shape[0] - d - 1 - i))
                ]
                S_list.append(S_ @ M)
            self.bspline_bases[v].matrices_S = S_list.copy()
            theta_ = model.getVariable("theta").slice(count, count + M.shape[1])

            if monotone:
                int_cons_incr = IntConstraints(
                    bspline={name: bsp},
                    var_name=name,
                    derivative=1,
                    constraints={"+": 0.0},
                )
                W_incr = int_cons_incr._get_matrices_W()
                C_incr = [w @ s[1:, :] for w, s in zip(W_incr, S_list)]
                H_incr = int_cons_incr._get_matrices_H()
                X_incr = model.variable(mosek.fusion.Domain.inPSDCone(d, len(S)))

                for j, C_ in enumerate(C_incr):
                    X_ = X_incr.slice([j, 0, 0], [j + 1, d, d]).reshape([d, d])
                    for k in range(d - 1):
                        model.constraint(
                            mosek.fusion.Expr.dot(H_incr[k], X_),
                            mosek.fusion.Domain.equalsTo(0.0),
                        )

                    for k in range(d - 1, 2 * d - 1):
                        model.constraint(
                            mosek.fusion.Expr.sub(
                                mosek.fusion.Expr.mul(C_, theta_).slice(
                                    k + 1 - d, k + 2 - d
                                ),
                                mosek.fusion.Expr.dot(H_incr[k], X_),
                            ),
                            mosek.fusion.Domain.equalsTo(0.0),
                        )
            if concave:
                int_cons_conc = IntConstraints(
                    bspline={name: bsp},
                    var_name=name,
                    derivative=2,
                    constraints={"-": 0.0},
                )
                W_conc = int_cons_conc._get_matrices_W()
                C_conc = [w @ s[2:, :] for w, s in zip(W_conc, S_list)]
                H_conc = int_cons_conc._get_matrices_H()
                X_conc = model.variable(mosek.fusion.Domain.inPSDCone(d, len(S)))

                for j, C_ in enumerate(C_conc):
                    X_ = X_conc.slice([j, 0, 0], [j + 1, d - 1, d - 1]).reshape(
                        [d - 1, d - 1]
                    )
                    for k in range(d - 2):
                        model.constraint(
                            mosek.fusion.Expr.dot(H_conc[k], X_),
                            mosek.fusion.Domain.equalsTo(0.0),
                        )

                    for k in range(d - 2, 2 * d - 3):
                        model.constraint(
                            mosek.fusion.Expr.sub(
                                mosek.fusion.Expr.mul(-C_, theta_).slice(k + 2 - d, k),
                                mosek.fusion.Expr.dot(H_conc[k], X_),
                            ),
                            mosek.fusion.Domain.equalsTo(0.0),
                        )
            count += M.shape[1]
        return model

    def _set_interaction_constraints(
        self,
        model: mosek.fusion.Model,
        concave: bool = True,
        monotone: bool = True,
    ) -> mosek.fusion.Model:
        theta_ = model.getVariable("theta").slice(
            self.obj_matrices["M"][0].shape[1] + self.obj_matrices["M"][1].shape[1] + 1,
            model.getVariable("theta").getShape()[0],
        )
        for i, kt2 in enumerate(
            self.bspline_bases[1].knots[self.deg[1] : -self.deg[1] - 1]
        ):
            if monotone:
                int_cons_incr = IntConstraints(
                    bspline={"x1": self.bspline_bases[0]},
                    var_name="x1",
                    derivative=1,
                    constraints={"+": 0.0},
                )
                W_incr = int_cons_incr._get_matrices_W()
                C_incr = [
                    np.kron(
                        w @ s[1:, :],
                        np.array([1, kt2, kt2**2, kt2**3])
                        @ self.bspline_bases[1].matrices_S[i],
                    )
                    for w, s in zip(W_incr, self.bspline_bases[0].matrices_S)
                ]
                H_incr = int_cons_incr._get_matrices_H()
                X_incr = model.variable(
                    mosek.fusion.Domain.inPSDCone(self.deg[0], len(W_incr))
                )

                for j, C_ in enumerate(C_incr):
                    X_ = X_incr.slice(
                        [j, 0, 0], [j + 1, self.deg[0], self.deg[0]]
                    ).reshape([self.deg[0], self.deg[0]])
                    for k in range(self.deg[0] - 1):
                        model.constraint(
                            mosek.fusion.Expr.dot(H_incr[k], X_),
                            mosek.fusion.Domain.equalsTo(0.0),
                        )

                    for k in range(self.deg[0] - 1, 2 * self.deg[0] - 1):
                        model.constraint(
                            mosek.fusion.Expr.sub(
                                mosek.fusion.Expr.mul(C_, theta_).slice(
                                    k + 1 - self.deg[0], k + 2 - self.deg[0]
                                ),
                                mosek.fusion.Expr.dot(H_incr[k], X_),
                            ),
                            mosek.fusion.Domain.equalsTo(0.0),
                        )
            if concave:
                int_cons_conc = IntConstraints(
                    bspline={"x1": self.bspline_bases[0]},
                    var_name="x1",
                    derivative=2,
                    constraints={"-": 0.0},
                )
                W_conc = int_cons_conc._get_matrices_W()
                C_conc = [
                    np.kron(
                        w @ s[2:, :],
                        np.array([1, kt2, kt2**2, kt2**3])
                        @ self.bspline_bases[1].matrices_S[i],
                    )
                    for w, s in zip(W_conc, self.bspline_bases[0].matrices_S)
                ]
                H_conc = int_cons_conc._get_matrices_H()
                X_conc = model.variable(
                    mosek.fusion.Domain.inPSDCone(self.deg[0], len(W_conc))
                )

                for j, C_ in enumerate(C_conc):
                    X_ = X_conc.slice(
                        [j, 0, 0], [j + 1, self.deg[0] - 1, self.deg[0] - 1]
                    ).reshape([self.deg[0] - 1, self.deg[0] - 1])
                    for k in range(self.deg[0] - 2):
                        model.constraint(
                            mosek.fusion.Expr.dot(H_conc[k], X_),
                            mosek.fusion.Domain.equalsTo(0.0),
                        )

                    for k in range(self.deg[0] - 2, 2 * self.deg[0] - 3):
                        model.constraint(
                            mosek.fusion.Expr.sub(
                                mosek.fusion.Expr.mul(-C_, theta_).slice(
                                    k + 2 - self.deg[0], k
                                ),
                                mosek.fusion.Expr.dot(H_conc[k], X_),
                            ),
                            mosek.fusion.Domain.equalsTo(0.0),
                        )
        for i, kt1 in enumerate(
            self.bspline_bases[0].knots[self.deg[0] : -self.deg[0] - 1]
        ):
            if monotone:
                int_cons_incr = IntConstraints(
                    bspline={"x2": self.bspline_bases[1]},
                    var_name="x2",
                    derivative=1,
                    constraints={"+": 0.0},
                )
                W_incr = int_cons_incr._get_matrices_W()
                C_incr = [
                    np.kron(
                        np.array([1, kt1, kt1**2, kt1**3])
                        @ self.bspline_bases[0].matrices_S[i],
                        w @ s[1:, :],
                    )
                    for w, s in zip(W_incr, self.bspline_bases[1].matrices_S)
                ]
                H_incr = int_cons_incr._get_matrices_H()
                X_incr = model.variable(
                    mosek.fusion.Domain.inPSDCone(self.deg[1], len(W_incr))
                )

                for j, C_ in enumerate(C_incr):
                    X_ = X_incr.slice(
                        [j, 0, 0], [j + 1, self.deg[1], self.deg[1]]
                    ).reshape([self.deg[1], self.deg[1]])
                    for k in range(self.deg[1] - 1):
                        model.constraint(
                            mosek.fusion.Expr.dot(H_incr[k], X_),
                            mosek.fusion.Domain.equalsTo(0.0),
                        )

                    for k in range(self.deg[1] - 1, 2 * self.deg[1] - 1):
                        model.constraint(
                            mosek.fusion.Expr.sub(
                                mosek.fusion.Expr.mul(C_, theta_).slice(
                                    k + 1 - self.deg[1], k + 2 - self.deg[1]
                                ),
                                mosek.fusion.Expr.dot(H_incr[k], X_),
                            ),
                            mosek.fusion.Domain.equalsTo(0.0),
                        )
            if concave:
                int_cons_conc = IntConstraints(
                    bspline={"x2": self.bspline_bases[1]},
                    var_name="x2",
                    derivative=2,
                    constraints={"-": 0.0},
                )
                W_conc = int_cons_conc._get_matrices_W()
                C_conc = [
                    np.kron(
                        np.array([1, kt1, kt1**2, kt1**3])
                        @ self.bspline_bases[0].matrices_S[i],
                        w @ s[2:, :],
                    )
                    for w, s in zip(W_conc, self.bspline_bases[1].matrices_S)
                ]
                H_conc = int_cons_conc._get_matrices_H()
                X_conc = model.variable(
                    mosek.fusion.Domain.inPSDCone(self.deg[1], len(W_conc))
                )

                for j, C_ in enumerate(C_conc):
                    X_ = X_conc.slice(
                        [j, 0, 0], [j + 1, self.deg[1] - 1, self.deg[1] - 1]
                    ).reshape([self.deg[1] - 1, self.deg[1] - 1])
                    for k in range(self.deg[1] - 2):
                        model.constraint(
                            mosek.fusion.Expr.dot(H_conc[k], X_),
                            mosek.fusion.Domain.equalsTo(0.0),
                        )

                    for k in range(self.deg[1] - 2, 2 * self.deg[1] - 3):
                        model.constraint(
                            mosek.fusion.Expr.sub(
                                mosek.fusion.Expr.mul(-C_, theta_).slice(
                                    k + 2 - self.deg[1], k
                                ),
                                mosek.fusion.Expr.dot(H_conc[k], X_),
                            ),
                            mosek.fusion.Domain.equalsTo(0.0),
                        )
        return model

    def _set_point_constraints(
        self, model: mosek.fusion.Model, y: np.ndarray
    ) -> mosek.fusion.Model:
        C = block_diag(
            *[
                1,
                self.obj_matrices["M"][0],
                self.obj_matrices["M"][1],
                np.kron(self.obj_matrices["M"][0], self.obj_matrices["M"][1]),
            ]
        )

        B = (
            add_constant(
                np.concatenate(
                    [
                        self.obj_matrices["B"][0],
                        self.obj_matrices["B"][1],
                        box_product(
                            self.obj_matrices["B"][0], self.obj_matrices["B"][1]
                        ),
                    ],
                    axis=1,
                )
            )
            @ C
        )
        model.constraint(
            mosek.fusion.Expr.mul(B, model.getVariable("theta")),
            mosek.fusion.Domain.greaterThan(y.astype(float)),
        )
        return model

    def _get_sp_optimizer(self) -> float:
        best_sp = minimize(
            GCV,
            x0=np.ones((4, 1)),
            args=(self.obj_matrices, self.family),
            method="L-BFGS-B",
            bounds=[(1e-10, 1e16) for _ in range(4)],
            options={"ftol": 1e-9},
        ).x
        return best_sp

    def fit(self, data: pd.DataFrame, y_col: str):
        self.feature_names = data.drop(columns=y_col).columns
        self.data_hull = Delaunay(data.loc[:, self.feature_names])

        x = [data[col].values for col in data.drop(columns=y_col)]
        y = data[y_col].values

        self.bspline_bases = self._get_bspline_bases(x=x)
        self.obj_matrices = self._get_obj_func_arrays(y=y)

        M = self._initialize_model()
        if self.constraint_monotone or self.constraint_concave:
            _ = self._set_additive_constraints(
                model=M,
                monotone=self.constraint_monotone,
                concave=self.constraint_concave,
            )
            _ = self._set_interaction_constraints(
                model=M,
                monotone=self.constraint_monotone,
                concave=self.constraint_concave,
            )
        if self.constraint_envelope:
            _ = self._set_point_constraints(model=M, y=y)

        self.best_sp = self._get_sp_optimizer()
        _ = M.getParameter("sp").setValue(self.best_sp)

        try:
            M.solve()
            self.sol = M.getVariable("theta").level()
        except mosek.fusion.SolutionError as e:
            raise ValueError(f"The original error was {e}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        x = [row for row in data.values.T]
        BM0 = (
            self.obj_matrices["scaler"][0].transform(
                self.bspline_bases[0].bspline_basis(x=x[0])
            )
            @ self.obj_matrices["M"][0]
        )
        BM1 = (
            self.obj_matrices["scaler"][1].transform(
                self.bspline_bases[1].bspline_basis(x=x[1])
            )
            @ self.obj_matrices["M"][1]
        )
        B_predict = add_constant(
            np.concatenate([BM0, BM1, box_product(BM0, BM1)], axis=1),
            has_constant="add",
        )
        return self.family.fitted(B_predict @ self.sol)
