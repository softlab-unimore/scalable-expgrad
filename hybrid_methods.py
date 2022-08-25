# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import logging
import os
import socket
from argparse import ArgumentParser
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
from synthetic_data import get_data, data_split


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


def run_hybrids(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all, eps,
                sample_indices, fractions, grid_fraction):
    # RUN hybrid models

    # Combine all training data into a single data frame
    train_all = pd.concat([X_train_all, y_train_all, A_train_all], axis=1)

    results = []

    # Iterations on difference fractions
    for i, f in enumerate(fractions):
        _time_expgrad_fracs = []
        _time_grid_fracs = []
        _time_hybrid1 = []
        _time_hybrid2 = []
        _time_hybrid3 = []
        _time_hybrid4 = []
        _time_hybrid5 = []
        _time_combined = []

        # Training arrays
        _train_error_expgrad_fracs = []
        _train_error_hybrids = []
        _train_error_grid_pmf_fracs = []
        _train_error_rewts = []

        _train_vio_expgrad_fracs = []
        _train_vio_hybrids = []
        _train_vio_grid_pmf_fracs = []
        _train_vio_rewts = []

        _train_vio_rewts_partial = []
        _train_error_rewts_partial = []

        _train_vio_no_grid_rewts = []
        _train_error_no_grid_rewts = []

        _train_vio_combined = []
        _train_error_combined = []

        # Testing arrays
        _test_error_expgrad_fracs = []
        _test_error_hybrids = []
        _test_error_grid_pmf_fracs = []
        _test_error_rewts = []

        _test_vio_expgrad_fracs = []
        _test_vio_hybrids = []
        _test_vio_grid_pmf_fracs = []
        _test_vio_rewts = []

        _test_vio_rewts_partial = []
        _test_error_rewts_partial = []

        _test_vio_no_grid_rewts = []
        _test_error_no_grid_rewts = []

        _test_vio_combined = []
        _test_error_combined = []

        for n in sample_indices:
            print(f"Processing: fraction {f}, sample {n}")



            # 2 algorithms:
            # - Expgrad + Grid
            # - Expgrad + Grid + LP (on all predictors, or top -k)
            # The actual LP is actually very expensive (can't run it). So we're using a "heuristic LP"



            # GridSearch data fraction
            grid_subsampling = train_all.sample(frac=grid_fraction, random_state=n + 60)
            grid_subsampling = grid_subsampling.reset_index()
            grid_subsampling = grid_subsampling.drop(columns=['index'])
            grid_tmp = grid_subsampling.iloc[:, :-1]
            grid_A_train = grid_subsampling.iloc[:, -1]
            grid_X_train = grid_tmp.iloc[:, :-1]
            grid_y_train = grid_tmp.iloc[:, -1]

            # Get a sample of the training data
            subsampling = train_all.sample(frac=f, random_state=n + 20)
            subsampling = subsampling.reset_index()
            subsampling = subsampling.drop(columns=['index'])
            tmp = subsampling.iloc[:, :-1]
            A_train = subsampling.iloc[:, -1]
            X_train = tmp.iloc[:, :-1]
            y_train = tmp.iloc[:, -1]

            # Expgrad on sample
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

            def getViolation(X, Y, A):
                disparity_moment = DemographicParity()
                disparity_moment.load_data(X, Y, sensitive_features=A)
                return disparity_moment.gamma(Qexp).max()

            def getError(X, Y, A):
                error = ErrorRate()
                error.load_data(X, Y, sensitive_features=A)
                return error.gamma(Qexp)[0]

            # Training Violation & Error of expgrad frac
            _train_error_expgrad_fracs.append(getError(X_train_all, y_train_all, A_train_all))
            _train_vio_expgrad_fracs.append(getViolation(X_train_all, y_train_all, A_train_all))

            # Testing Violation & Error of expgrad frac
            _test_error_expgrad_fracs.append(getError(X_test_all, y_test_all, A_test_all))
            _test_vio_expgrad_fracs.append(getViolation(X_test_all, y_test_all, A_test_all))

            #################################################################################################
            # Hybrid 5: Run LP with full dataset on predictors trained on partial dataset only
            # Get rid
            #################################################################################################
            no_grid_errors = []
            no_grid_vio = pd.DataFrame()
            expgrad_predictors = expgrad_X_logistic_frac.predictors_  # fairlearn==0.5.0
            # expgrad_predictors = expgrad_X_logistic_frac._predictors  # fairlearn==0.4.6

            a = datetime.now()
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

            # In hybrid 5, lin program is done on top of expgrad partial.
            new_weights_no_grid = solve_linprog(errors=no_grid_errors, gammas=no_grid_vio, eps=eps, nu=1e-6,
                                                pred=expgrad_predictors)
            b = datetime.now()
            time_lin_prog5 = (b - a).total_seconds()
            _time_hybrid5.append(time_expgrad_frac + time_lin_prog5)

            def Q_rewts_no_grid(X):
                return _pmf_predict(X, expgrad_predictors, new_weights_no_grid)[:, 1]

            def getViolation(X, Y, A):
                disparity_moment = DemographicParity()
                disparity_moment.load_data(X, Y, sensitive_features=A)
                return disparity_moment.gamma(Q_rewts_no_grid).max()

            def getError(X, Y, A):
                error = ErrorRate()
                error.load_data(X, Y, sensitive_features=A)
                return error.gamma(Q_rewts_no_grid)[0]

            # Training violation & error of hybrid 5
            train_vio_hybrid5 = getViolation(X_train_all, y_train_all, A_train_all)
            train_error_hybrid5 = getError(X_train_all, y_train_all, A_train_all)
            _train_vio_no_grid_rewts.append(train_vio_hybrid5)
            _train_error_no_grid_rewts.append(train_error_hybrid5)

            # Testing violation & error of hybrid 5
            test_vio_hybrid5 = getViolation(X_test_all, y_test_all, A_test_all)
            test_error_hybrid5 = getError(X_test_all, y_test_all, A_test_all)
            _test_vio_no_grid_rewts.append(test_vio_hybrid5)
            _test_error_no_grid_rewts.append(test_error_hybrid5)

            #################################################################################################
            # Hybrid 1: Just Grid Search -> expgrad partial + grid search
            #################################################################################################
            # Grid Search part
            print(f"Running GridSearch (fraction={grid_fraction})...")
            # TODO: Change constraint_weight according to eps
            lambda_vecs_logistic = expgrad_X_logistic_frac.lambda_vecs_  # 0.5.0
            # lambda_vecs_logistic = expgrad_X_logistic_frac._lambda_vecs_lagrangian  # 0.4.6
            grid_search_logistic_frac = GridSearch(
                LogisticRegression(solver='liblinear', fit_intercept=True),
                constraints=DemographicParity(), grid=lambda_vecs_logistic)
            a = datetime.now()
            # grid_search_logistic_frac.fit(X_train_all, y_train_all,
            #                               sensitive_features=A_train_all)
            grid_search_logistic_frac.fit(grid_X_train, grid_y_train, sensitive_features=grid_A_train)
            b = datetime.now()
            time_grid_frac = (b - a).total_seconds()
            _time_grid_fracs.append(time_grid_frac)
            print(f"GridSearch (fraction={grid_fraction}) done in {b - a}")


            # Remove this part
            def Qgrid(X):
                # Biased coin toss every predict
                return grid_search_logistic_frac.predict(X)

            def getViolation(X, Y, A):
                disparity_moment = DemographicParity()
                disparity_moment.load_data(X, Y, sensitive_features=A)
                return disparity_moment.gamma(Qgrid).max()

            def getError(X, Y, A):
                error = ErrorRate()
                error.load_data(X, Y, sensitive_features=A)
                return error.gamma(Qgrid)[0]

            # Training violation & error of hybrid 1
            a = datetime.now()
            train_vio_hybrid1 = getViolation(X_train_all, y_train_all, A_train_all)
            train_error_hybrid1 = getError(X_train_all, y_train_all, A_train_all)
            b = datetime.now()
            time_h1 = (b - a).total_seconds()
            _time_hybrid1.append(time_grid_frac + time_expgrad_frac + time_h1)
            _train_vio_hybrids.append(train_vio_hybrid1)
            _train_error_hybrids.append(train_error_hybrid1)

            # Testing violation & error of hybrid 1
            test_vio_hybrid1 = getViolation(X_test_all, y_test_all, A_test_all)
            test_error_hybrid1 = getError(X_test_all, y_test_all, A_test_all)
            _test_vio_hybrids.append(test_vio_hybrid1)
            _test_error_hybrids.append(test_error_hybrid1)

            #################################################################################################
            # Hybrid 2: pmf_predict with exp grad weights in grid search
            # Keep this, remove Hybrid 1.
            #################################################################################################
            print("Running Hybrid 2...")
            _weights_logistic = expgrad_X_logistic_frac.weights_  # 0.5.0
            _predictors = grid_search_logistic_frac.predictors_  # 0.5.0
            # Weights from ExpGrad
            # _weights_logistic = expgrad_X_logistic_frac.weights_  # 0.4.6
            # _predictors = grid_search_logistic_frac.predictors_  # 0.4.6

            # Time taken by hybrid 2 to fit a model is same as hybrid 1. The only change is while predicting
            _time_hybrid2.append(time_grid_frac + time_expgrad_frac + time_h1)

            def Qlog(X):
                return _pmf_predict(X, _predictors, _weights_logistic)[:, 1]

            def getViolation(X, Y, A):
                disparity_moment = DemographicParity()
                disparity_moment.load_data(X, Y, sensitive_features=A)
                return disparity_moment.gamma(Qlog).max()

            def getError(X, Y, A):
                error = ErrorRate()
                error.load_data(X, Y, sensitive_features=A)
                return error.gamma(Qlog)[0]

            # Training violation & error of hybrid 2
            a = datetime.now()
            train_vio_hybrid2 = getViolation(X_train_all, y_train_all, A_train_all)
            train_error_hybrid2 = getError(X_train_all, y_train_all, A_train_all)
            b = datetime.now()
            time_h2 = (b - a).total_seconds()
            _train_vio_grid_pmf_fracs.append(train_vio_hybrid2)
            _train_error_grid_pmf_fracs.append(train_error_hybrid2)

            # Testing violation & error of hybrid 2
            test_vio_hybrid2 = getViolation(X_test_all, y_test_all, A_test_all)
            test_error_hybrid2 = getError(X_test_all, y_test_all, A_test_all)
            _test_vio_grid_pmf_fracs.append(test_vio_hybrid2)
            _test_error_grid_pmf_fracs.append(test_error_hybrid2)
            print("Hybrid 2 done")

            #################################################################################################
            # Hybrid 3: re-weight using LP
            #################################################################################################
            print("Running Hybrid 3...")
            grid_errors = []
            grid_vio = pd.DataFrame()
            a = datetime.now()
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

                # error_grid_frac is a small vector (e.g. 4)

                grid_vio[x] = violation_grid_frac
                grid_errors.append(error_grid_frac)

            grid_errors = pd.Series(grid_errors)

            # In hybrid 3, time for linprogramming is added on top of hybrid 1 time.
            new_weights = solve_linprog(errors=grid_errors, gammas=grid_vio, eps=eps, nu=1e-6, pred=_predictors)
            b = datetime.now()
            time_lin_program = (b - a).total_seconds()
            _time_hybrid3.append(time_grid_frac + time_expgrad_frac + time_lin_program)

            def Q_rewts(X):
                return _pmf_predict(X, _predictors, new_weights)[:, 1]

            def getViolation(X, Y, A):
                disparity_moment = DemographicParity()
                disparity_moment.load_data(X, Y, sensitive_features=A)
                return disparity_moment.gamma(Q_rewts).max()

            def getError(X, Y, A):
                error = ErrorRate()
                error.load_data(X, Y, sensitive_features=A)
                return error.gamma(Q_rewts)[0]

            # Training violation & error of hybrid 3
            a = datetime.now()
            train_vio_hybrid3 = getViolation(X_train_all, y_train_all, A_train_all)
            train_error_hybrid3 = getError(X_train_all, y_train_all, A_train_all)
            b = datetime.now()
            time_h3 = (b - a).total_seconds()
            _train_vio_rewts.append(train_vio_hybrid3)
            _train_error_rewts.append(train_error_hybrid3)

            # Testing violation & error of hybrid 3
            test_vio_hybrid3 = getViolation(X_test_all, y_test_all, A_test_all)
            test_error_hybrid3 = getError(X_test_all, y_test_all, A_test_all)
            _test_vio_rewts.append(test_vio_hybrid3)
            _test_error_rewts.append(test_error_hybrid3)
            print("Hybrid 3 done")

            #################################################################################################
            # Hybrid 4: re-weight only the non-zero weight predictors using LP
            #################################################################################################
            print("Running Hybrid 4...")
            re_wts_predictors = []
            for x in range(len(_weights_logistic)):
                if _weights_logistic[x] != 0:
                    re_wts_predictors.append(_predictors[x])
            grid_errors_partial = []
            grid_vio_partial = pd.DataFrame()
            a = datetime.now()
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

            # Should we count this time? -> Yes
            # In hybrid 4, time taken to do perform lin programming is added on top of hybrid 1.
            new_weights_partial = solve_linprog(errors=grid_errors_partial, gammas=grid_vio_partial, eps=eps, nu=1e-6,
                                                pred=re_wts_predictors)
            b = datetime.now()
            time_new_lin_program = (b - a).total_seconds()
            _time_hybrid4.append(time_grid_frac + time_expgrad_frac + time_new_lin_program)

            def Q_rewts_partial(X):
                return _pmf_predict(X, re_wts_predictors, new_weights_partial)[:, 1]

            def getViolation(X, Y, A):
                disparity_moment = DemographicParity()
                disparity_moment.load_data(X, Y, sensitive_features=A)
                return disparity_moment.gamma(Q_rewts_partial).max()

            def getError(X, Y, A):
                error = ErrorRate()
                error.load_data(X, Y, sensitive_features=A)
                return error.gamma(Q_rewts_partial)[0]

            # Training violation & error of hybrid 4
            a = datetime.now()
            train_vio_hybrid4 = getViolation(X_train_all, y_train_all, A_train_all)
            train_error_hybrid4 = getError(X_train_all, y_train_all, A_train_all)
            b = datetime.now()
            time_h4 = (b - a).total_seconds()
            _train_vio_rewts_partial.append(train_vio_hybrid4)
            _train_error_rewts_partial.append(train_error_hybrid4)

            # Testing violation & error of hybrid 4
            test_vio_hybrid4 = getViolation(X_test_all, y_test_all, A_test_all)
            test_error_hybrid4 = getError(X_test_all, y_test_all, A_test_all)
            _test_vio_rewts_partial.append(test_vio_hybrid4)
            _test_error_rewts_partial.append(test_error_hybrid4)
            print("Hybrid 4 done")

            # combined time
            def get_combined_hybrid(alpha):
                hybrid1 = ((1 - alpha) * train_vio_hybrid1) + (alpha * train_error_hybrid1)
                hybrid2 = ((1 - alpha) * train_vio_hybrid2) + (alpha * train_error_hybrid2)
                hybrid3 = ((1 - alpha) * train_vio_hybrid3) + (alpha * train_error_hybrid3)
                hybrid4 = ((1 - alpha) * train_vio_hybrid4) + (alpha * train_error_hybrid4)
                hybrid5 = ((1 - alpha) * train_vio_hybrid5) + (alpha * train_error_hybrid5)

                max_hybrid = min(hybrid1, hybrid2, hybrid3, hybrid4, hybrid5)

                if max_hybrid == hybrid1:
                    return train_vio_hybrid1, train_error_hybrid1, test_vio_hybrid1, test_error_hybrid1
                if max_hybrid == hybrid2:
                    return train_vio_hybrid2, train_error_hybrid2, test_vio_hybrid2, test_error_hybrid2
                if max_hybrid == hybrid3:
                    return train_vio_hybrid3, train_error_hybrid3, test_vio_hybrid3, test_error_hybrid3
                if max_hybrid == hybrid4:
                    return train_vio_hybrid4, train_error_hybrid4, test_vio_hybrid4, test_error_hybrid4
                if max_hybrid == hybrid5:
                    return train_vio_hybrid5, train_error_hybrid5, test_vio_hybrid5, test_error_hybrid5

            alpha = 0.5
            train_vio_combined, train_error_combined, test_vio_combined, test_error_combined = get_combined_hybrid(
                alpha)
            _train_vio_combined.append(train_vio_combined)
            _train_error_combined.append(train_error_combined)
            _test_vio_combined.append(test_vio_combined)
            _test_error_combined.append(test_error_combined)
            _time_combined.append(time_grid_frac + time_expgrad_frac +
                                  time_new_lin_program + time_lin_prog5 + time_lin_program +
                                  time_h1 + time_h2 + time_h3 + time_h4)

            print("Fraction processing complete.")
            print()

        print(f"Done {len(_train_error_expgrad_fracs)} samples for fraction {f}")

        results.append({
            "eps": eps,
            "frac": f,
            "grid_frac": grid_fraction,

            "_time_expgrad_fracs": _time_expgrad_fracs,
            "_time_hybrid1": _time_hybrid1,
            "_time_hybrid2": _time_hybrid2,
            "_time_hybrid3": _time_hybrid3,
            "_time_hybrid4": _time_hybrid4,
            "_time_hybrid5": _time_hybrid5,
            "_time_combined": _time_combined,

            "_train_error_expgrad_fracs": _train_error_expgrad_fracs,
            "_train_vio_expgrad_fracs": _train_vio_expgrad_fracs,
            "_train_error_hybrids": _train_error_hybrids,
            "_train_vio_hybrids": _train_vio_hybrids,
            "_train_error_grid_pmf_fracs": _train_error_grid_pmf_fracs,
            "_train_vio_grid_pmf_fracs": _train_vio_grid_pmf_fracs,
            "_train_error_rewts": _train_error_rewts,
            "_train_vio_rewts": _train_vio_rewts,
            "_train_error_rewts_partial": _train_error_rewts_partial,
            "_train_vio_rewts_partial": _train_vio_rewts_partial,
            "_train_error_no_grid_rewts": _train_error_no_grid_rewts,
            "_train_vio_no_grid_rewts": _train_vio_no_grid_rewts,
            "_train_error_combined": _train_error_combined,
            "_train_vio_combined": _train_vio_combined,

            "_test_error_expgrad_fracs": _test_error_expgrad_fracs,
            "_test_vio_expgrad_fracs": _test_vio_expgrad_fracs,
            "_test_error_hybrids": _test_error_hybrids,
            "_test_vio_hybrids": _test_vio_hybrids,
            "_test_error_grid_pmf_fracs": _test_error_grid_pmf_fracs,
            "_test_vio_grid_pmf_fracs": _test_vio_grid_pmf_fracs,
            "_test_error_rewts": _test_error_rewts,
            "_test_vio_rewts": _test_vio_rewts,
            "_test_error_rewts_partial": _test_error_rewts_partial,
            "_test_vio_rewts_partial": _test_vio_rewts_partial,
            "_test_error_no_grid_rewts": _test_error_no_grid_rewts,
            "_test_vio_no_grid_rewts": _test_vio_no_grid_rewts,
            "_test_error_combined": _test_error_combined,
            "_test_vio_combined": _test_vio_combined,
        })

    return results


# def main():
#     host_name = socket.gethostname()
#     if "." in host_name:
#         host_name = host_name.split(".")[-1]
#
#     eps = 0.05
#     num_data_pts = 10000000
#     num_features = 4
#     type_ratio = 0.5
#     t0_ratio = 0.3
#     t1_ratio = 0.6
#     random_variation = 1
#     dataset_str = f"synth_n{num_data_pts}_f{num_features}_r{type_ratio}_{t0_ratio}_{t1_ratio}_v{random_variation}"
#
#     hybrid_results_file_name = \
#         f'results/{host_name}/{dataset_str}/{str(eps)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_hybrid.json'
#
#     #X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all = load_data()
#
#     print("Generating synth data...")
#     All = get_data(
#         num_data_pts=num_data_pts,
#         num_features=num_features,
#         type_ratio=type_ratio,
#         t0_ratio=t0_ratio,
#         t1_ratio=t1_ratio,
#         random_seed=random_variation + 40)
#     X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all = data_split(All, 0.3)
#
#     print(hybrid_results_file_name)
#
#     results = run_hybrids(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all, eps)
#
#     # Store results
#     base_dir = os.path.dirname(hybrid_results_file_name)
#     if not os.path.isdir(base_dir):
#         os.makedirs(base_dir, exist_ok=True)
#     with open(hybrid_results_file_name, 'w') as _file:
#         json.dump(results, _file, indent=2)
#
#
# if __name__ == "__main__":
#     main()
