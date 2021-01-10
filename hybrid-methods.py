# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import logging
import os
import socket
from datetime import datetime

import numpy as np
import pandas as pd
import pickle
import scipy.optimize as opt
from fairlearn.reductions import DemographicParity, ExponentiatedGradient, ErrorRate, GridSearch
from sklearn.dummy import DummyClassifier
from time import time

from sklearn.linear_model import LogisticRegression

from utils import load_data

_PRECISION = 1e-8
_LINE = "_" * 9
_INDENTATION = " " * 9

logger = logging.getLogger(__name__)


def _pmf_predict(X, predictors, weights):
    pred = pd.DataFrame()
    for t in range(len(predictors)):
        pred[t] = predictors[t].predict(X)
    positive_probs = pred[weights.index].dot(weights).to_frame()
    return np.concatenate((1-positive_probs, positive_probs), axis=1)


class _Lagrangian:
    """Operations related to the Lagrangian.

    :param X: the training features
    :type X: Array
    :param sensitive_features: the sensitive features to use for constraints
    :type sensitive_features: Array
    :param y: the training labels
    :type y: Array
    :param estimator: the estimator to fit in every iteration of best_h
    :type estimator: an estimator that has a `fit` method with arguments X, y, and sample_weight
    :param constraints: Object describing the parity constraints. This provides the reweighting
        and relabelling
    :type constraints: `fairlearn.reductions.Moment`
    :param eps: allowed constraint violation
    :type eps: float
    :param B:
    :type B:
    :param opt_lambda: indicates whether to optimize lambda during the calculation of the
        Lagrangian; optional with default value True
    :type opt_lambda: bool
    """

    def __init__(self, X, sensitive_features, y, estimator, constraints, eps, B, opt_lambda=True):
        self.X = X
        self.constraints = constraints
        self.constraints.load_data(X, y, sensitive_features=sensitive_features)
        self.obj = self.constraints.default_objective()
        self.obj.load_data(X, y, sensitive_features=sensitive_features)
        self.pickled_estimator = pickle.dumps(estimator)
        self.eps = eps
        self.B = B
        self.opt_lambda = opt_lambda
        self.hs = pd.Series(dtype="float64")
        self.classifiers = pd.Series(dtype="float64")
        self.errors = pd.Series(dtype="float64")
        self.gammas = pd.DataFrame()
        self.lambdas = pd.DataFrame()
        self.n = self.X.shape[0]
        self.n_oracle_calls = 0
        self.n_oracle_calls_dummy_returned = 0
        self.oracle_execution_times = []
        self.last_linprog_n_hs = 0
        self.last_linprog_result = None

    def _eval(self, Q, lambda_vec):
        """Return the value of the Lagrangian.

        :param Q: `Q` is either a series of weights summing up to 1 that indicate the weight of
            each `h` in contributing to the randomized classifier, or a callable corresponding to
            a deterministic predict function.
        :type Q: pandas.Series or callable
        :param lambda_vec: lambda vector
        :type lambda_vec: pandas.Series

        :return: tuple `(L, L_high, gamma, error)` where `L` is the value of the Lagrangian,
            `L_high` is the value of the Lagrangian under the best response of the lambda player,
            `gamma` is the vector of constraint violations, and `error` is the empirical error
        """
        if callable(Q):
            error = self.obj.gamma(Q)[0]
            gamma = self.constraints.gamma(Q)
        else:
            error = self.errors[Q.index].dot(Q)
            gamma = self.gammas[Q.index].dot(Q)

        if self.opt_lambda:
            lambda_projected = self.constraints.project_lambda(lambda_vec)
            L = error + np.sum(lambda_projected * gamma) - self.eps * np.sum(lambda_projected)
        else:
            L = error + np.sum(lambda_vec * gamma) - self.eps * np.sum(lambda_vec)

        max_gamma = gamma.max()
        if max_gamma < self.eps:
            L_high = error
        else:
            L_high = error + self.B * (max_gamma - self.eps)
        return L, L_high, gamma, error

    def eval_gap(self, Q, lambda_hat, nu):
        r"""Return the duality gap object for the given :math:`Q` and :math:`\hat{\lambda}`."""
        L, L_high, gamma, error = self._eval(Q, lambda_hat)
        result = _GapResult(L, L, L_high, gamma, error)
        for mul in [1.0, 2.0, 5.0, 10.0]:
            h_hat, h_hat_idx = self.best_h(mul * lambda_hat)
            logger.debug("%smul=%.0f", _INDENTATION, mul)
            L_low_mul, _, _, _ = self._eval(pd.Series({h_hat_idx: 1.0}), lambda_hat)
            if L_low_mul < result.L_low:
                result.L_low = L_low_mul
            if result.gap() > nu + _PRECISION:
                break
        return result

    def solve_linprog(self, errors=None, gammas=None, nu=1e-6):
        if errors is None:
          errors = self.errors
        if gammas is None:
          gammas = self.gammas
        n_hs = len(self.hs)
        n_constraints = len(self.constraints.index)
        if self.last_linprog_n_hs == n_hs:
            return self.last_linprog_result
        c = np.concatenate((errors, [self.B]))
        A_ub = np.concatenate((gammas - self.eps, -np.ones((n_constraints, 1))), axis=1)
        b_ub = np.zeros(n_constraints)
        A_eq = np.concatenate((np.ones((1, n_hs)), np.zeros((1, 1))), axis=1)
        b_eq = np.ones(1)
        result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='simplex')
        Q = pd.Series(result.x[:-1], self.hs.index)
        dual_c = np.concatenate((b_ub, -b_eq))
        dual_A_ub = np.concatenate((-A_ub.transpose(), A_eq.transpose()), axis=1)
        dual_b_ub = c
        dual_bounds = [(None, None) if i == n_constraints else (0, None) for i in range(n_constraints + 1)]  # noqa: E501
        result_dual = opt.linprog(dual_c,
                                  A_ub=dual_A_ub,
                                  b_ub=dual_b_ub,
                                  bounds=dual_bounds,
                                  method='simplex')
        lambda_vec = pd.Series(result_dual.x[:-1], self.constraints.index)
        self.last_linprog_n_hs = n_hs
        self.last_linprog_result = (Q, lambda_vec, self.eval_gap(Q, lambda_vec, nu))
        return self.last_linprog_result

    def _call_oracle(self, lambda_vec):
        signed_weights = self.obj.signed_weights() + self.constraints.signed_weights(lambda_vec)
        redY = 1 * (signed_weights > 0)
        redW = signed_weights.abs()
        redW = self.n * redW / redW.sum()

        redY_unique = np.unique(redY)

        classifier = None
        if len(redY_unique) == 1:
            logger.debug("redY had single value. Using DummyClassifier")
            classifier = DummyClassifier(strategy='constant',
                                         constant=redY_unique[0])
            self.n_oracle_calls_dummy_returned += 1
        else:
            classifier = pickle.loads(self.pickled_estimator)

        oracle_call_start_time = time()
        classifier.fit(self.X, redY, sample_weight=redW)
        self.oracle_execution_times.append(time() - oracle_call_start_time)
        self.n_oracle_calls += 1

        return classifier

    def best_h(self, lambda_vec):
        """Solve the best-response problem.

        Returns the classifier that solves the best-response problem for
        the vector of Lagrange multipliers `lambda_vec`.
        """
        classifier = self._call_oracle(lambda_vec)
        def h(X): return classifier.predict(X)
        h_error = self.obj.gamma(h)[0]
        h_gamma = self.constraints.gamma(h)
        h_value = h_error + h_gamma.dot(lambda_vec)

        if not self.hs.empty:
            values = self.errors + self.gammas.transpose().dot(lambda_vec)
            best_idx = values.idxmin()
            best_value = values[best_idx]
        else:
            best_idx = -1
            best_value = np.PINF

        if h_value < best_value - _PRECISION:
            logger.debug("%sbest_h: val improvement %f", _LINE, best_value - h_value)
            h_idx = len(self.hs)
            self.hs.at[h_idx] = h
            self.classifiers.at[h_idx] = classifier
            self.errors.at[h_idx] = h_error
            self.gammas[h_idx] = h_gamma
            self.lambdas[h_idx] = lambda_vec.copy()
            best_idx = h_idx

        return self.hs[best_idx], best_idx


class _GapResult:
    """The result of a duality gap computation."""

    def __init__(self, L, L_low, L_high, gamma, error):
        self.L = L
        self.L_low = L_low
        self.L_high = L_high
        self.gamma = gamma
        self.error = error

    def gap(self):
        return max(self.L - self.L_low, self.L_high - self.L)


def solve_linprog(errors=None, gammas=None, eps=0.05, nu=1e-6, pred=None):
    B = 1/eps
    n_hs = len(pred)
    n_constraints = 4 #len()

    c = np.concatenate((errors, [B]))
    A_ub = np.concatenate((gammas - eps, -np.ones((n_constraints, 1))), axis=1)
    b_ub = np.zeros(n_constraints)
    A_eq = np.concatenate((np.ones((1, n_hs)), np.zeros((1, 1))), axis=1)
    b_eq = np.ones(1)
    result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='simplex')
    Q = pd.Series(result.x[:-1])
    return Q


def run_methods(X_train_all, y_train_all, A_train_all, eps):
    # RUN hybrid models

    # Combine all training data into a single data frame and glance at a few rows
    train_all = pd.concat([X_train_all, y_train_all, A_train_all], axis=1)

    # # Read past results
    # if not os.path.isfile(hybrid_results_file_name):
    #     all_results = []
    # else:
    #     with open(hybrid_results_file_name, 'r') as _file:
    #         all_results = json.load(_file)

    results = []

    # Subsampling process
    num_samples = 10  # 20
    # fractions = [0.1, 0.5]
    num_fractions = 6  # 20
    fractions = np.logspace(-3, 0, num=num_fractions)

    # Iterations on difference fractions
    for i, f in enumerate(fractions):
        # if i > 0: break

        _time_expgrad_fracs = []
        _time_grid_fracs = []
        _time_hybrids = []

        _error_expgrad_fracs = []
        _error_hybrids = []
        _error_grid_pmf_fracs = []
        _error_rewts = []

        _vio_expgrad_fracs = []
        _vio_hybrids = []
        _vio_grid_pmf_fracs = []
        _vio_rewts = []

        _vio_rewts_partial = []
        _error_rewts_partial = []

        _vio_no_grid_rewts = []
        _error_no_grid_rewts = []

        _best_lambda = pd.DataFrame()

        for n in range(num_samples):
            # if n > 1: break

            print(f"Processing: fraction {f}, sample {n}")

            # get samples
            subsampling = train_all.sample(frac=f, random_state=n + 20)
            subsampling = subsampling.reset_index()
            subsampling = subsampling.drop(columns=['index'])
            tmp = subsampling.iloc[:, :-1]
            A_train = subsampling.iloc[:, -1]
            X_train = tmp.iloc[:, :-1]
            y_train = tmp.iloc[:, -1]

            # Expgrad on sub dataset
            expgrad_X_logistic_frac = ExponentiatedGradient(
                LogisticRegression(solver='liblinear', fit_intercept=True),
                constraints=DemographicParity(), eps=eps, nu=1e-6)

            print("Fitting ExponentiatedGradient on subset...")
            a = datetime.now()
            expgrad_X_logistic_frac.fit(X_train, y_train, sensitive_features=A_train)
            b = datetime.now()
            time_expgrad_frac = (b - a).total_seconds()
            _time_expgrad_fracs.append(time_expgrad_frac)
            print(f"ExponentiatedGradient on subset done in {b - a}")

            def Qexp(X):
                return expgrad_X_logistic_frac._pmf_predict(X)[:, 1]

            # violation of log res
            disparity_moment = DemographicParity()
            disparity_moment.load_data(X_train_all, y_train_all,
                                       sensitive_features=A_train_all)
            violation_expgrad_logistic_frac = disparity_moment.gamma(Qexp).max()

            # error of log res
            error = ErrorRate()
            error.load_data(X_train_all, y_train_all,
                            sensitive_features=A_train_all)
            error_expgrad_logistic_frac = error.gamma(Qexp)[0]

            _error_expgrad_fracs.append(error_expgrad_logistic_frac)
            _vio_expgrad_fracs.append(violation_expgrad_logistic_frac)

            # Hybrid 5: run LP with full dataset on predictors trained on partial dataset only
            no_grid_errors = []
            no_grid_vio = pd.DataFrame()
            expgrad_predictors = expgrad_X_logistic_frac._predictors
            for x in range(len(expgrad_predictors)):
                def Q_preds_no_grid(X): return expgrad_predictors[x].predict(X)

                # violation of log res
                disparity_moment = DemographicParity()
                disparity_moment.load_data(X_train_all, y_train_all,
                                           sensitive_features=A_train_all)
                violation_no_grid_frac = disparity_moment.gamma(Q_preds_no_grid)

                # error of log res
                error = ErrorRate()
                error.load_data(X_train_all, y_train_all,
                                sensitive_features=A_train_all)
                error_no_grid_frac = error.gamma(Q_preds_no_grid)['all']

                no_grid_vio[x] = violation_no_grid_frac
                no_grid_errors.append(error_no_grid_frac)

            no_grid_errors = pd.Series(no_grid_errors)
            new_weights_no_grid = solve_linprog(errors=no_grid_errors, gammas=no_grid_vio, eps=0.05, nu=1e-6,
                                                pred=expgrad_predictors)

            def Q_rewts_no_grid(X):
                return _pmf_predict(X, expgrad_predictors, new_weights_no_grid)[:, 1]

            # violation of log res
            disparity_moment = DemographicParity()
            disparity_moment.load_data(X_train_all, y_train_all, sensitive_features=A_train_all)
            vio_rewts_no_grid_ = disparity_moment.gamma(Q_rewts_no_grid).max()

            # error of log res
            error = ErrorRate()
            error.load_data(X_train_all, y_train_all, sensitive_features=A_train_all)
            error_rewts_no_grid_ = error.gamma(Q_rewts_no_grid)[0]

            _vio_no_grid_rewts.append(vio_rewts_no_grid_)
            _error_no_grid_rewts.append(error_rewts_no_grid_)

            # Grid Search part
            print("Running GridSearch...")
            lambda_vecs_logistic = expgrad_X_logistic_frac._lambda_vecs_lagrangian
            grid_search_logistic_frac = GridSearch(
                LogisticRegression(solver='liblinear', fit_intercept=True),
                constraints=DemographicParity(), grid=lambda_vecs_logistic)
            a = datetime.now()
            grid_search_logistic_frac.fit(X_train_all, y_train_all,
                                          sensitive_features=A_train_all)
            b = datetime.now()
            time_grid_frac = (b - a).total_seconds()
            _time_grid_fracs.append(time_grid_frac)
            _time_hybrids.append(time_grid_frac + time_expgrad_frac)
            print(f"GridSearch done in {b - a}")

            def Qgrid(X):
                return grid_search_logistic_frac.predict(X)

            # violation of log res
            disparity_moment = DemographicParity()
            disparity_moment.load_data(X_train_all, y_train_all,
                                       sensitive_features=A_train_all)
            violation_grid_frac = disparity_moment.gamma(Qgrid).max()

            # error of log res
            error = ErrorRate()
            error.load_data(X_train_all, y_train_all,
                            sensitive_features=A_train_all)
            error_grid_frac = error.gamma(Qgrid)['all']

            _vio_hybrids.append(violation_grid_frac)
            _error_hybrids.append(error_grid_frac)

            # Hybrid 2: pmf_predict with exp grad weights in grid search
            print("Running Hybrid 2...")
            _weights_logistic = expgrad_X_logistic_frac._weights
            _predictors = grid_search_logistic_frac._predictors

            def Qlog(X):
                return _pmf_predict(X, _predictors, _weights_logistic)[:, 1]

            # violation of log res
            disparity_moment = DemographicParity()
            disparity_moment.load_data(X_train_all, y_train_all, sensitive_features=A_train_all)
            violation_grid_search_pmf = disparity_moment.gamma(Qlog).max()

            # error of log res
            error = ErrorRate()
            error.load_data(X_train_all, y_train_all, sensitive_features=A_train_all)
            error_grid_search_pmf = error.gamma(Qlog)[0]

            _vio_grid_pmf_fracs.append(violation_grid_search_pmf)
            _error_grid_pmf_fracs.append(error_grid_search_pmf)
            print("Hybrid 2 done")

            # Hybrid 0: grid of best lambdas of each subset
            # lambda_vecs = expgrad_X_logistic_frac._lambda_vecs_LP
            # keys = lambda_vecs.keys()
            # _best_lambda[n] = lambda_vecs[keys[-1]]

            # Hybrid 3: re-weight using LP
            print("Running Hybrid 3...")
            grid_errors = []
            grid_vio = pd.DataFrame()
            for x in range(len(_predictors)):
                def Q_preds(X): return _predictors[x].predict(X)

                # violation of log res
                disparity_moment = DemographicParity()
                disparity_moment.load_data(X_train_all, y_train_all,
                                           sensitive_features=A_train_all)
                violation_grid_frac = disparity_moment.gamma(Q_preds)

                # error of log res
                error = ErrorRate()
                error.load_data(X_train_all, y_train_all,
                                sensitive_features=A_train_all)
                error_grid_frac = error.gamma(Q_preds)['all']

                grid_vio[x] = violation_grid_frac
                grid_errors.append(error_grid_frac)

            grid_errors = pd.Series(grid_errors)
            new_weights = solve_linprog(errors=grid_errors, gammas=grid_vio, eps=0.05, nu=1e-6, pred=_predictors)

            def Q_rewts(X):
                return _pmf_predict(X, _predictors, new_weights)[:, 1]

            # violation of log res
            disparity_moment = DemographicParity()
            disparity_moment.load_data(X_train_all, y_train_all, sensitive_features=A_train_all)
            vio_rewts_pmf_ = disparity_moment.gamma(Q_rewts).max()

            # error of log res
            error = ErrorRate()
            error.load_data(X_train_all, y_train_all, sensitive_features=A_train_all)
            error_rewts_pmf_ = error.gamma(Q_rewts)[0]

            _vio_rewts.append(vio_rewts_pmf_)
            _error_rewts.append(error_rewts_pmf_)
            print("Hybrid 3 done")

            # Hybrid 4: re-weight only the non-zero weight predictors using LP
            print("Running Hybrid 4...")
            re_wts_predictors = []
            for x in range(len(_weights_logistic)):
                if _weights_logistic[x] != 0:
                    re_wts_predictors.append(_predictors[x])
            grid_errors_partial = []
            grid_vio_partial = pd.DataFrame()
            for x in range(len(re_wts_predictors)):
                def Q_preds_partial(X): return re_wts_predictors[x].predict(X)

                # violation of log res
                disparity_moment = DemographicParity()
                disparity_moment.load_data(X_train_all, y_train_all, sensitive_features=A_train_all)
                violation_grid_frac_partial = disparity_moment.gamma(Q_preds_partial)

                # error of log res
                error = ErrorRate()
                error.load_data(X_train_all, y_train_all, sensitive_features=A_train_all)
                error_grid_frac_partial = error.gamma(Q_preds_partial)['all']

                grid_vio_partial[x] = violation_grid_frac_partial
                grid_errors_partial.append(error_grid_frac_partial)

            grid_errors_partial = pd.Series(grid_errors_partial)
            new_weights_partial = solve_linprog(errors=grid_errors_partial, gammas=grid_vio_partial, eps=0.05, nu=1e-6,
                                                pred=re_wts_predictors)

            def Q_rewts_partial(X):
                return _pmf_predict(X, re_wts_predictors, new_weights_partial)[:, 1]

            # violation of log res
            disparity_moment = DemographicParity()
            disparity_moment.load_data(X_train_all, y_train_all, sensitive_features=A_train_all)
            vio_rewts_partial_ = disparity_moment.gamma(Q_rewts_partial).max()

            # error of log res
            error = ErrorRate()
            error.load_data(X_train_all, y_train_all, sensitive_features=A_train_all)
            error_rewts_partial_ = error.gamma(Q_rewts_partial)[0]

            _vio_rewts_partial.append(vio_rewts_partial_)
            _error_rewts_partial.append(error_rewts_partial_)
            print("Hybrid 4 done")

            print("Sample processing complete.")
            print()

        print(f"Done {len(_error_expgrad_fracs)} samples for fraction {f}")

        results.append({
            "frac": f,
            "_time_expgrad_fracs": _time_expgrad_fracs,
            "_time_hybrids": _time_hybrids,
            "_error_expgrad_fracs": _error_expgrad_fracs,
            "_vio_expgrad_fracs": _vio_expgrad_fracs,
            "_error_hybrids": _error_hybrids,
            "_vio_hybrids": _vio_hybrids,
            "_error_grid_pmf_fracs": _error_grid_pmf_fracs,
            "_vio_grid_pmf_fracs": _vio_grid_pmf_fracs,
            "_error_rewts": _error_rewts,
            "_vio_rewts": _vio_rewts,
            "_error_rewts_partial": _error_rewts_partial,
            "_vio_rewts_partial": _vio_rewts_partial,
            "_error_no_grid_rewts": _error_no_grid_rewts,
            "_vio_no_grid_rewts": _vio_no_grid_rewts,
        })

    return results


def main():
    host_name = socket.gethostname()
    if "." in host_name:
        host_name = host_name.split(".")[-1]

    eps = 0.05

    hybrid_results_file_name = \
        f'results/{host_name}/{str(eps)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_hybrid.json'

    X_train_all, y_train_all, A_train_all = load_data()

    results = run_methods(X_train_all, y_train_all, A_train_all, eps)

    # Store results
    base_dir = os.path.dirname(hybrid_results_file_name)
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    with open(hybrid_results_file_name, 'w') as _file:
        json.dump(results, _file, indent=2)


if __name__ == "__main__":
    main()
