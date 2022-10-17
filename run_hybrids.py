# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import inspect
import itertools
import logging
import pickle
from copy import deepcopy
from datetime import datetime
from time import time
from warnings import simplefilter

import numpy as np
import pandas as pd
import scipy.optimize as opt
from fairlearn.reductions import DemographicParity, ExponentiatedGradient, ErrorRate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from hybrid_models import Hybrid5, Hybrid1, Hybrid2, Hybrid3, Hybrid4, ExponentiatedGradientPmf
from utils import aggregate_phase_time
from metrics import metrics_dict

_PRECISION = 1e-8
_LINE = "_" * 9
_INDENTATION = " " * 9

logger = logging.getLogger(__name__)


class _Lagrangian:
    """ Operations related to the Lagrangian.

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
        dual_bounds = [(None, None) if i == n_constraints else (0, None) for i in
                       range(n_constraints + 1)]  # noqa: E501
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

        def h(X):
            return classifier.predict(X)

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


def get_metrics(dataset_dict: dict, predict_method, metrics_methods=metrics_dict):
    metrics_res = {}
    for phase, dataset_list in dataset_dict.items():
        X, Y, S = dataset_list
        y_pred = predict_method(X)
        for name, eval_method in metrics_methods.items():
            params = inspect.signature(eval_method).parameters.keys()
            if 'predict_method' in params:
                turn_res = eval_method(*dataset_list, predict_method=predict_method)
            elif 'y_pred' in params:
                turn_res = eval_method(*dataset_list, y_pred=y_pred)
            metrics_res[f'{phase}_{name}'] = turn_res
    return metrics_res


def run_hybrids(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all, eps,
                sample_indices, fractions, grid_fractions, train_test_fold=None):
    assert train_test_fold is not None
    simplefilter(action='ignore', category=FutureWarning)

    n_data = X_test_all.shape[0]

    # Combine all training data into a single data frame
    train_all = pd.concat([X_train_all, y_train_all, A_train_all], axis=1)

    results = []
    data_dict = {"eps": eps, "frac": 0, 'model_name': 'model', 'time': 0, 'phase': 'model name', 'random_seed': 0,
                 'train_test_fold': train_test_fold}
    eval_dataset_dict = {'train': [X_train_all, y_train_all, A_train_all],
                         'test': [X_test_all, y_test_all, A_test_all]}
    all_params = dict(X=X_train_all, y=y_train_all, sensitive_features=A_train_all)

    constraint = DemographicParity(difference_bound=eps)
    to_iter = list(itertools.product(fractions, grid_fractions, sample_indices))
    # Iterations on difference fractions
    for i, (exp_f, grid_f, n) in tqdm(list(enumerate(to_iter))):
        turn_results = []
        data_dict['exp_frac'] = exp_f
        data_dict["grid_frac"] = grid_f
        data_dict["random_seed"] = n
        data_dict['exp_size'] = int(n_data * exp_f)
        data_dict['grid_size'] = int(n_data * grid_f)
        print(f"Processing: fraction {exp_f:0<5}, sample {n:10} GridSearch fraction={grid_f:0<5}")

        base_model = LogisticRegression(solver='liblinear', fit_intercept=True, random_state=n)
        unconstrained_model = deepcopy(base_model)
        a = datetime.now()
        unconstrained_model.fit(X_train_all, y_train_all)
        b = datetime.now()
        time = (b - a).total_seconds()
        time_unconstrained = {'time': time, 'phase': 'unconstrained'}

        # 2 algorithms:
        # - Expgrad + Grid
        # - Expgrad + Grid + LP (on all predictors, or top -k)
        # The actual LP is actually very expensive (can't run it). So we're using a "heuristic LP"

        # GridSearch data fraction
        grid_sample = train_all.sample(frac=grid_f, random_state=n + 60)
        grid_sample = grid_sample.reset_index(drop=True)
        # todo check split
        grid_params = dict(X=grid_sample.iloc[:, :-2],
                           y=grid_sample.iloc[:, -2],
                           sensitive_features=grid_sample.iloc[:, -1])

        # Get a sample of the training data
        exp_sample = train_all.sample(frac=exp_f, random_state=n + 20)  # todo --> sampling with or w/out overlap
        exp_sample = exp_sample.reset_index(drop=True)
        exp_params = dict(X=exp_sample.iloc[:, :-2],
                          y=exp_sample.iloc[:, -2],
                          sensitive_features=exp_sample.iloc[:, -1])

        # Expgrad on sample
        data_dict['model_name'] = 'expgrad_fracs'
        expgrad_frac = ExponentiatedGradientPmf(estimator=deepcopy(base_model),
                                                constraints=deepcopy(constraint), eps=eps, nu=1e-6)
        metrics_res, time_exp_dict, time_eval_dict = fit_evaluate_model(expgrad_frac, exp_params, eval_dataset_dict)
        time_exp_dict['phase'] = 'expgrad_fracs'
        print(f"ExponentiatedGradient on subset done in {time_exp_dict['time']}")
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_exp_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # Hybrid 5: Run LP with full dataset on predictors trained on partial dataset only
        # Get rid
        #################################################################################################
        data_dict['model_name'] = 'hybrid_5'
        print(f"Running {data_dict['model_name']}")
        model1 = Hybrid5(expgrad_frac, eps=eps, constraint=deepcopy(constraint))
        metrics_res, time_lp_dict, time_eval_dict = fit_evaluate_model(model1, all_params, eval_dataset_dict)
        time_lp_dict['phase'] = 'lin_prog'
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_exp_dict, time_lp_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # H5 + unconstrained
        #################################################################################################
        data_dict['model_name'] = 'hybrid_5_U'
        print(f"Running {data_dict['model_name']}")
        model1.unconstrained_model = unconstrained_model
        metrics_res, time_lp_dict, time_eval_dict = fit_evaluate_model(model1, all_params, eval_dataset_dict)
        time_lp_dict['phase'] = 'lin_prog'
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_exp_dict, time_unconstrained, time_lp_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # Hybrid 1: Just Grid Search -> expgrad partial + grid search
        #################################################################################################
        data_dict['model_name'] = 'hybrid_1'
        print(f"Running {data_dict['model_name']}")
        model = Hybrid1(expgrad_frac, eps=eps, constraint=deepcopy(constraint))
        metrics_res, time_grid_dict, time_eval_dict = fit_evaluate_model(model, grid_params, eval_dataset_dict)
        time_grid_dict['phase'] = 'grid_frac'
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_exp_dict, time_grid_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        grid_search_frac = model.grid_search_frac

        #################################################################################################
        # Hybrid 2: pmf_predict with exp grad weights in grid search
        # Keep this, remove Hybrid 1.
        #################################################################################################
        data_dict['model_name'] = 'hybrid_2'
        print(f"Running {data_dict['model_name']}")
        model = Hybrid2(expgrad_frac, grid_search_frac, eps=eps, constraint=deepcopy(constraint))
        metrics_res, _, time_eval_dict = fit_evaluate_model(model, grid_params, eval_dataset_dict)
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_exp_dict, time_grid_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # Hybrid 3: re-weight using LP
        #################################################################################################
        data_dict['model_name'] = 'hybrid_3'
        print(f"Running {data_dict['model_name']}")
        model = Hybrid3(grid_search_frac=grid_search_frac, eps=eps, constraint=deepcopy(constraint))
        metrics_res, time_lp3_dict, time_eval_dict = fit_evaluate_model(model, all_params, eval_dataset_dict)
        time_lp3_dict['phase'] = 'lin_prog'
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_exp_dict, time_grid_dict, time_lp3_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # Hybrid 3 +U: re-weight using LP
        #################################################################################################
        data_dict['model_name'] = 'hybrid_3_U'
        print(f"Running {data_dict['model_name']}")
        model = Hybrid3(grid_search_frac=grid_search_frac, eps=eps, constraint=deepcopy(constraint),
                        unconstrained_model=unconstrained_model)
        metrics_res, time_lp3_dict, time_eval_dict = fit_evaluate_model(model, all_params, eval_dataset_dict)
        time_lp3_dict['phase'] = 'lin_prog'
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_exp_dict, time_grid_dict, time_lp3_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # Hybrid 4: re-weight only the non-zero weight predictors using LP
        #################################################################################################
        data_dict['model_name'] = 'hybrid_4'
        print(f"Running {data_dict['model_name']}")
        model = Hybrid4(expgrad_frac, grid_search_frac, eps=eps, constraint=deepcopy(constraint))
        metrics_res, time_lp4_dict, time_eval_dict = fit_evaluate_model(model, all_params, eval_dataset_dict)
        time_lp4_dict['phase'] = 'lin_prog'
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_exp_dict, time_grid_dict, time_lp4_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # Hybrid 6: exp + grid predictors
        #################################################################################################
        data_dict['model_name'] = 'hybrid_6'
        print(f"Running {data_dict['model_name']}")
        model = Hybrid3(add_exp_predictors=True, grid_search_frac=grid_search_frac, expgrad_frac=expgrad_frac,
                        eps=eps, constraint=deepcopy(constraint))
        metrics_res, time_lp3_dict, time_eval_dict = fit_evaluate_model(model, all_params, eval_dataset_dict)
        time_lp3_dict['phase'] = 'lin_prog'
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_exp_dict, time_grid_dict, time_lp3_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # Hybrid 6 + U: exp + grid predictors + unconstrained
        #################################################################################################
        data_dict['model_name'] = 'hybrid_6_U'
        print(f"Running {data_dict['model_name']}")
        model = Hybrid3(add_exp_predictors=True, grid_search_frac=grid_search_frac, expgrad_frac=expgrad_frac,
                        eps=eps, constraint=deepcopy(constraint), unconstrained_model=unconstrained_model)
        metrics_res, time_lp3_dict, time_eval_dict = fit_evaluate_model(model, all_params, eval_dataset_dict)
        time_lp3_dict['phase'] = 'lin_prog'
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_exp_dict, time_grid_dict, time_lp3_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        results += turn_results
        print("Fraction processing complete.\n")

    return results


def fit_evaluate_model(model, train_dataset, evaluate_dataset_dict):
    a = datetime.now()
    model.fit(**train_dataset)
    b = datetime.now()
    time_fit_dict = {'time': (b - a).total_seconds(), 'phase': 'train'}

    # Training violation & error of hybrid 4
    a = datetime.now()
    metrics_res = get_metrics(evaluate_dataset_dict, model.predict)
    b = datetime.now()
    time_eval_dict = {'time': (b - a).total_seconds(), 'phase': 'evaluation'}
    return metrics_res, time_fit_dict, time_eval_dict


if __name__ == "__main__":
    from test.test_hybrid_methods import test_run_hybrids

    test_run_hybrids()
    print('end')
