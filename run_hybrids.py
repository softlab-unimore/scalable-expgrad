# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
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

from hybrid_models import Hybrid5, Hybrid1, Hybrid2, Hybrid3, Hybrid4
from utils import aggregate_phase_time
from metrics import metrics_dict

_PRECISION = 1e-8
_LINE = "_" * 9
_INDENTATION = " " * 9

logger = logging.getLogger(__name__)


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
    for name, eval_method in metrics_methods.items():
        for phase, dataset_list in dataset_dict.items():
            metrics_res[f'{phase}_{name}'] = eval_method(*dataset_list, predict_method=predict_method)
    return metrics_res


def run_hybrids(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all, eps,
                sample_indices, fractions, grid_fraction):
    simplefilter(action='ignore', category=FutureWarning)
    # RUN hybrid models

    # Combine all training data into a single data frame
    train_all = pd.concat([X_train_all, y_train_all, A_train_all], axis=1)

    results = []
    data_dict = {"eps": eps, "frac": 0, 'model_name': 'model', 'time': 0, 'phase': 'model name', 'random_seed': 0}
    datasets_dict = {'train': [X_train_all, y_train_all, A_train_all],
                     'test': [X_test_all, y_test_all, A_test_all]}
    all_params = dict(X=X_train_all, y=y_train_all, sensitive_features=A_train_all)
    to_iter = list(itertools.product(fractions, sample_indices))
    constraint = DemographicParity(difference_bound=eps)
    # Iterations on difference fractions
    for i, (f, n) in tqdm(list(enumerate(to_iter))):
        turn_results = []
        data_dict['frac'] = f
        data_dict["grid_frac"] = grid_fraction
        data_dict["random_seed"] = n
        print(f"Processing: fraction {f}, sample {n}")

        # 2 algorithms:
        # - Expgrad + Grid
        # - Expgrad + Grid + LP (on all predictors, or top -k)
        # The actual LP is actually very expensive (can't run it). So we're using a "heuristic LP"

        # GridSearch data fraction
        grid_subsampling = train_all.sample(frac=grid_fraction, random_state=n + 60)
        grid_subsampling = grid_subsampling.reset_index(drop=True)
        grid_A_train = grid_subsampling.iloc[:, -1]
        grid_X_train = grid_subsampling.iloc[:, :-2]
        grid_y_train = grid_subsampling.iloc[:, -2]
        # todo check split
        grid_params = dict(X=grid_X_train, y=grid_y_train, sensitive_features=grid_A_train)

        # Get a sample of the training data
        exp_subsampling = train_all.sample(frac=f, random_state=n + 20)  # todo --> sampling with or w/out overlap
        exp_subsampling = exp_subsampling.reset_index(drop=True)
        # todo check & remove
        # subsampling = subsampling.drop(columns=['index'])
        X_train = exp_subsampling.iloc[:, :-2]
        A_train = exp_subsampling.iloc[:, -1]
        y_train = exp_subsampling.iloc[:, -2]
        exp_params = dict(X=X_train, y=y_train, sensitive_features=A_train)

        # Expgrad on sample
        data_dict['model_name'] = 'expgrad_fracs'
        expgrad_X_logistic_frac = ExponentiatedGradient(
            LogisticRegression(solver='liblinear', fit_intercept=True, random_state=n),
            constraints=deepcopy(constraint), eps=eps, nu=1e-6)
        print("Fitting ExponentiatedGradient on subset...")
        a = datetime.now()
        expgrad_X_logistic_frac.fit(**exp_params)
        b = datetime.now()
        time_expgrad_frac = (b - a).total_seconds()
        time_expgrad_frac_dict = {'time': time_expgrad_frac, 'phase': 'expgrad_fracs'}

        print(f"ExponentiatedGradient on subset done in {b - a}")

        def Qexp(X):
            return expgrad_X_logistic_frac._pmf_predict(X)[:, 1]

        a = datetime.now()
        metrics_res = get_metrics(datasets_dict, Qexp)
        b = datetime.now()
        time_eval_dict = {'time': (b - a).total_seconds(), 'phase': 'evaluation'}
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_expgrad_frac_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # Hybrid 5: Run LP with full dataset on predictors trained on partial dataset only
        # Get rid
        #################################################################################################
        data_dict['model_name'] = 'hybrid_5'
        model1 = Hybrid5(expgrad_X_logistic_frac, eps=eps, constraint=deepcopy(constraint))
        a = datetime.now()
        model1.fit(**all_params)
        b = datetime.now()
        time_lin_prog_h5 = (b - a).total_seconds()

        a = datetime.now()
        metrics_res = get_metrics(datasets_dict, model1.predict)
        b = datetime.now()
        time_eval_dict = {'time': (b - a).total_seconds(), 'phase': 'evaluation'}
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_expgrad_frac_dict,
                            {'time': time_lin_prog_h5, 'phase': 'lin_prog'}
                            ]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # Hybrid 1: Just Grid Search -> expgrad partial + grid search
        #################################################################################################
        # Grid Search part
        data_dict['model_name'] = 'hybrid_1'
        print(f"Running GridSearch (fraction={grid_fraction})...")
        model = Hybrid1(expgrad_X_logistic_frac, eps=eps, constraint=deepcopy(constraint))
        a = datetime.now()
        model.fit(**grid_params)
        b = datetime.now()
        time_grid_frac = (b - a).total_seconds()
        time_grid_frac_dict = {'time': time_grid_frac, 'phase': 'grid_frac'}

        print(f"GridSearch (fraction={grid_fraction}) done in {b - a}")
        a = datetime.now()
        metrics_res = get_metrics(datasets_dict, model.predict)
        b = datetime.now()
        time_eval_dict = {'time': (b - a).total_seconds(), 'phase': 'evaluation'}
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_expgrad_frac_dict, time_grid_frac_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        grid_search_logistic_frac = model.grid_search_logistic_frac

        #################################################################################################
        # Hybrid 2: pmf_predict with exp grad weights in grid search
        # Keep this, remove Hybrid 1.
        #################################################################################################

        print("Running Hybrid 2...")
        data_dict['model_name'] = 'hybrid_2'
        model = Hybrid2(expgrad_X_logistic_frac, grid_search_logistic_frac, eps=eps, constraint=deepcopy(constraint))
        model.fit(**grid_params)
        a = datetime.now()
        metrics_res = get_metrics(datasets_dict, model.predict)
        b = datetime.now()
        time_eval_dict = {'time': (b - a).total_seconds(), 'phase': 'evaluation'}
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_expgrad_frac_dict, time_grid_frac_dict]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # Hybrid 3: re-weight using LP
        #################################################################################################
        print("Running Hybrid 3...")
        data_dict['model_name'] = 'hybrid_3'
        # Hybrid 3 is hybrid 5 with the predictors of grid_search
        model = Hybrid3(grid_search_logistic_frac=grid_search_logistic_frac, eps=eps, constraint=deepcopy(constraint))

        a = datetime.now()
        model.fit(**all_params)
        b = datetime.now()
        time_lin_prog_h3 = (b - a).total_seconds()

        a = datetime.now()
        metrics_res = get_metrics(datasets_dict, model.predict)
        b = datetime.now()
        time_eval_dict = {'time': (b - a).total_seconds(), 'phase': 'evaluation'}
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_expgrad_frac_dict, time_grid_frac_dict,
                            {'time': time_lin_prog_h3, 'phase': 'lin_prog'}]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # Hybrid 4: re-weight only the non-zero weight predictors using LP
        #################################################################################################
        print("Running Hybrid 4...")
        data_dict['model_name'] = 'hybrid_4'

        model = Hybrid4(expgrad_X_logistic_frac, grid_search_logistic_frac, eps=eps, constraint=deepcopy(constraint))
        a = datetime.now()
        model.fit(**all_params)
        b = datetime.now()
        time_lin_prog_h4 = (b - a).total_seconds()

        # Training violation & error of hybrid 4
        a = datetime.now()
        metrics_res = get_metrics(datasets_dict, model.predict)
        b = datetime.now()
        time_eval_dict = {'time': (b - a).total_seconds(), 'phase': 'evaluation'}
        data_dict.update(**metrics_res)
        for t_time_dict in [time_eval_dict, time_expgrad_frac_dict, time_grid_frac_dict,
                            {'time': time_lin_prog_h4, 'phase': 'lin_prog'}]:
            data_dict.update(**t_time_dict)
            turn_results.append(deepcopy(data_dict))

        #################################################################################################
        # COMBINED
        #################################################################################################
        alpha = 0.5
        time_expanded_df = pd.DataFrame(turn_results)
        turn_res_df = aggregate_phase_time(time_expanded_df)
        hybrid_res = turn_res_df[turn_res_df['model_name'].str.startswith('hybrid')]
        composed_metric = hybrid_res['train_violation'] * (1 - alpha) + hybrid_res['train_error'] * alpha
        combo_res = hybrid_res.loc[composed_metric.idxmin()]
        data_dict['model_name'] = 'combined'
        data_dict['alpha'] = alpha
        for df_name in datasets_dict.keys():
            for metric_name in metrics_dict.keys():
                turn_key = f'{df_name}_{metric_name}'
                data_dict[turn_key] = combo_res[turn_key]

        eval_time_dict_list = [
            {'time': turn_res['time'], 'phase': f'evaluation_{turn_res["model_name"].split("_")[-1]}'}
            for turn_res in turn_results if
            turn_res['phase'] == 'evaluation' and turn_res['model_name'].startswith('hybrid_')]
        for time_dict in [time_expgrad_frac_dict,
                          time_grid_frac_dict,
                          {'time': time_lin_prog_h3, 'phase': 'lin_prog_3'},
                          {'time': time_lin_prog_h4, 'phase': 'lin_prog_4'},
                          {'time': time_lin_prog_h5, 'phase': 'lin_prog_5'}
                          ] + eval_time_dict_list:
            data_dict.update(**time_dict)
            turn_results.append(deepcopy(data_dict))

        results += turn_results
        print("Fraction processing complete.\n")

    return results


if __name__ == "__main__":
    from test.test_hybrid_methods import test_run_hybrids

    test_run_hybrids()
    print('end')
