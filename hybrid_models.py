import numpy as np
import pandas as pd
import scipy.optimize as opt
from fairlearn.reductions import ErrorRate, GridSearch
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


def solve_linprog(errors=None, gammas=None, eps=0.05, nu=1e-6, pred=None):
    B = 1 / eps
    n_hs = len(pred)
    n_constraints = 4  # len()

    c = np.concatenate((errors, [B]))
    A_ub = np.concatenate((gammas - eps, -np.ones((n_constraints, 1))), axis=1)
    b_ub = np.zeros(n_constraints)
    A_eq = np.concatenate((np.ones((1, n_hs)), np.zeros((1, 1))), axis=1)
    b_eq = np.ones(1)
    result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='simplex')
    Q = pd.Series(result.x[:-1])
    return Q


def _pmf_predict(X, predictors, weights):
    pred = pd.DataFrame()
    for t in range(len(predictors)):
        pred[t] = predictors[t].predict(X)
    positive_probs = pred[weights.index].dot(weights).to_frame()
    return np.concatenate((1 - positive_probs, positive_probs), axis=1)


class Hybrid5(BaseEstimator):

    def __init__(self, predictors=None):
        self.predictors = predictors

    def fit_expgrad(self, X, y, A):
        expgrad_X_logistic_frac = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                                                        constraints=DemographicParity(), eps=eps, nu=1e-6)
        print("Fitting ExponentiatedGradient on subset...")
        expgrad_X_logistic_frac.fit(X, y, sensitive_features=A)
        self.predictors = expgrad_X_logistic_frac.predictors_  # fairlearn==0.5.0
        # self.expgrad_predictors = expgrad_X_logistic_frac._predictors  # fairlearn==0.4

    def fit(self, X, y, A, eps):
        if self.predictors is None:
            self.fit_expgrad(X, y, A)

        grid_errors = []
        grid_vio = pd.DataFrame()
        for x in range(len(self.predictors)):
            def Q_preds(X): return self.predictors[x].predict(X)

            # violation of log res
            disparity_moment = DemographicParity()
            disparity_moment.load_data(X, y, sensitive_features=A)
            violation_no_grid_frac = disparity_moment.gamma(Q_preds)

            # error of log res
            error = ErrorRate()
            error.load_data(X, y, sensitive_features=A)
            error_no_grid_frac = error.gamma(Q_preds)['all']

            grid_vio[x] = violation_no_grid_frac
            grid_errors.append(error_no_grid_frac)

        grid_errors = pd.Series(grid_errors)

        # In hybrid 5, lin program is done on top of expgrad partial.
        self.weights = solve_linprog(errors=grid_errors, gammas=grid_vio, eps=eps, nu=1e-6,
                                     pred=self.predictors)
        return self

    def predict(self, X):
        return _pmf_predict(X, self.predictors, self.weights)[:, 1]


class Hybrid1(BaseEstimator):

    def __init__(self, lambda_vecs_=None):
        self.lambda_vecs_logistic = lambda_vecs_


    def fit_expgrad(self, X, y, A):
        """
        # TODO: Change constraint_weight according to eps
        """
        expgrad_X_logistic_frac = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                                                        constraints=DemographicParity(), eps=eps, nu=1e-6)
        print("Fitting ExponentiatedGradient on subset...")
        expgrad_X_logistic_frac.fit(X, y, sensitive_features=A)
        self.lambda_vecs_logistic = expgrad_X_logistic_frac.lambda_vecs_  # fairlearn==0.5.0
        # self.lambda_vecs_logistic = expgrad_X_logistic_frac._lambda_vecs_lagrangian  # fairlearn==0.4

    def fit(self, X, y, A):
        if self.lambda_vecs_logistic is None:
            self.fit_expgrad(X, y, A)
        self.grid_search_logistic_frac = GridSearch(
            LogisticRegression(solver='liblinear', fit_intercept=True),
            constraints=DemographicParity(), grid=self.lambda_vecs_logistic)
        self.grid_search_logistic_frac.fit(X, y, sensitive_features=A)
        return self

    def predict(self, X):
        # todo ?? Remove this part
        return self.grid_search_logistic_frac.predict(X)


class Hybrid2(BaseEstimator):

    def __init__(self, weights=None, predictors=None):
        self.weights_ = weights
        self.predictors_ = predictors

    def fit_expgrad(self, X, y, A):
        self.expgrad_X_logistic_frac = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                                                        constraints=DemographicParity(), eps=eps, nu=1e-6)
        print("Fitting ExponentiatedGradient on subset...")
        self.expgrad_X_logistic_frac.fit(X, y, sensitive_features=A)
        self.weights_ = self.expgrad_X_logistic_frac.weights_

    def fit_grid(self, X, y, A):
        self.grid_search_logistic_frac = GridSearch(
            LogisticRegression(solver='liblinear', fit_intercept=True),
            constraints=DemographicParity(), grid=self.expgrad_X_logistic_frac.lambda_vecs_)
        self.grid_search_logistic_frac.fit(X, y, sensitive_features=A)
        self.predictors_ = self.grid_search_logistic_frac.predictors_

    def fit(self, X, y, A):
        if self.weights_ is None:
            self.fit_expgrad(X, y, A)
        if self.predictors_ is None:
            self.fit_grid(X, y, A)
        return self

    def predict(self, X):
        return _pmf_predict(X, self.predictors_, self.weights_)[:, 1]