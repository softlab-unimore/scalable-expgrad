import numpy as np
import pandas as pd
import scipy.optimize as opt
from fairlearn.reductions import ErrorRate, GridSearch
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


def solve_linprog_strict(errors=None, gammas=None, eps=0.05, nu=1e-6, pred=None):
    B = 1 / eps
    n_hs = len(pred)
    n_constraints = 4  # len()

    c = np.concatenate(errors)
    A_ub = gammas  # gammas @ weights <= eps
    b_ub = np.repeat(eps, n_constraints)
    A_eq = np.ones(1, n_hs)
    b_eq = np.ones(1)
    result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs-ds')
    Q = pd.Series(result.x[:-1])
    return Q


def solve_linprog(errors=None, gammas=None, eps=0.05, nu=1e-6, pred=None, eps_plus=0):
    B = 1 / eps
    n_hs = len(pred)
    n_constraints = 4  # len()

    c = np.concatenate((errors, [B**2])) # min err @ weights + B^2 * x5 # for feasibility
    A_ub = np.concatenate((gammas, -np.ones((n_constraints, 1))), axis=1) # vio @ weights <= eps + x5
    b_ub = np.repeat(eps, n_constraints)
    A_eq = np.array([[1] * n_hs + [0]])
    b_eq = np.ones((1,1))
    result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs-ds')
    Q = pd.Series(result.x[:-1])
    return Q


def _pmf_predict(X, predictors, weights):
    pred = pd.DataFrame()
    for t in range(len(predictors)):
        pred[t] = predictors[t].predict(X)
    positive_probs = pred[weights.index].dot(weights).to_frame()
    return np.concatenate((1 - positive_probs, positive_probs), axis=1)


class Hybrid5(BaseEstimator):

    def __init__(self, expgrad_X_logistic_frac=None, eps=None, constraint=None):
        if constraint is None:
            constraint = DemographicParity(difference_bound=eps)
        self.constraint = constraint
        self.expgrad_logistic_frac = expgrad_X_logistic_frac
        self.eps = eps

    def fit_expgrad(self, X, y, sensitive_features):
        expgrad_X_logistic_frac = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                                                        constraints=self.constraint, eps=self.eps, nu=1e-6)
        print("Fitting ExponentiatedGradient on subset...")
        expgrad_X_logistic_frac.fit(X, y, sensitive_features=sensitive_features)
        self.expgrad_logistic_frac = expgrad_X_logistic_frac

    def get_erro_vio(self, X, y, sensitive_features, predictors):
        error_list = []
        violation_df = pd.DataFrame()
        for x, turn_predictor in enumerate(predictors):
            def Q_preds(X): return turn_predictor.predict(X)

            # violation of log res
            disparity_moment = DemographicParity()
            disparity_moment.load_data(X, y, sensitive_features=sensitive_features)
            turn_violation = disparity_moment.gamma(Q_preds)

            # error of log res
            error = ErrorRate()
            error.load_data(X, y, sensitive_features=sensitive_features)
            turn_error = error.gamma(Q_preds)['all']

            violation_df[x] = turn_violation
            error_list.append(turn_error)

        error_list = pd.Series(error_list)
        return error_list, violation_df

    def fit(self, X, y, sensitive_features):
        if self.expgrad_logistic_frac is None:
            self.fit_expgrad(X, y, sensitive_features)
        self.predictors = self.expgrad_logistic_frac.predictors_  # fairlearn==0.5.0
        # self.expgrad_predictors = expgrad_X_logistic_frac._predictors  # fairlearn==0.4

        # In hybrid 5, lin program is done on top of expgrad partial.
        errors, violations = self.get_erro_vio(X, y, sensitive_features, self.predictors)
        self.weights = solve_linprog(errors=errors, gammas=violations, eps=self.eps, nu=1e-6,
                                     pred=self.predictors)
        return self

    def predict(self, X):
        return _pmf_predict(X, self.predictors, self.weights)[:, 1]


class Hybrid1(Hybrid5):

    def __init__(self, expgrad_X_logistic_frac=None, grid_search_logistic_frac=None, eps=None, **kwargs):
        super().__init__(eps=eps, **kwargs)
        self.expgrad_logistic_frac = expgrad_X_logistic_frac
        self.grid_search_logistic_frac = grid_search_logistic_frac
        self.eps = eps

    def fit_grid(self, X, y, sensitive_features):
        if self.expgrad_logistic_frac is None:
            self.fit_expgrad(X, y, sensitive_features)
        self.grid_search_logistic_frac = GridSearch(
            LogisticRegression(solver='liblinear', fit_intercept=True),
            constraints=self.constraint, grid=self.expgrad_logistic_frac.lambda_vecs_)  # TODO no eps
        # _lambda_vecs_lagrangian  # fairlearn==0.4
        self.grid_search_logistic_frac.fit(X, y, sensitive_features=sensitive_features)

    def fit(self, X, y, sensitive_features):
        if self.expgrad_logistic_frac is None:
            self.fit_expgrad(X, y, sensitive_features)
        if self.grid_search_logistic_frac is None:
            self.fit_grid(X, y, sensitive_features)
        return self

    def predict(self, X):
        return self.grid_search_logistic_frac.predict(X)


class Hybrid2(Hybrid1):

    def predict(self, X):
        self.weights = self.expgrad_logistic_frac.weights_
        self.predictors = self.grid_search_logistic_frac.predictors_
        return _pmf_predict(X, self.predictors, self.weights)[:, 1]


class Hybrid3(Hybrid1):

    def fit(self, X, y, sensitive_features):
        if self.grid_search_logistic_frac is None:
            self.fit_grid(X, y, sensitive_features)
        self.predictors = self.grid_search_logistic_frac.predictors_  # fairlearn==0.5.0
        # self.expgrad_predictors = expgrad_X_logistic_frac._predictors  # fairlearn==0.4
        errors, violations = self.get_erro_vio(X, y, sensitive_features, self.predictors)
        self.weights = solve_linprog(errors=errors, gammas=violations, eps=self.eps, nu=1e-6,
                                     pred=self.predictors)
        return self

    def predict(self, X):
        return _pmf_predict(X, self.predictors, self.weights)[:, 1]


class Hybrid4(Hybrid1):

    def fit(self, X, y, sensitive_features):
        super().fit(X, y, sensitive_features)
        weights_logistic = self.expgrad_logistic_frac.weights_
        predictors = self.grid_search_logistic_frac.predictors_
        re_wts_predictors = [predictors[idx] for idx, turn_weight in enumerate(weights_logistic) if turn_weight != 0]
        self.predictors = re_wts_predictors
        errors, violations = self.get_erro_vio(X, y, sensitive_features, self.predictors)
        self.weights = solve_linprog(errors=errors, gammas=violations, eps=self.eps, nu=1e-6,
                                     pred=self.predictors)

    def predict(self, X):
        return _pmf_predict(X, self.predictors, self.weights)[:, 1]
