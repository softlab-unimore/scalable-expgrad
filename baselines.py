import json
import logging
import os
import socket
from datetime import datetime

from fairlearn.reductions import DemographicParity, ErrorRate, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression

from utils import load_data


def run_unmitigated(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all):
    # Unmitigated LogRes
    logistic_learner = LogisticRegression(solver='liblinear', fit_intercept=True)

    a = datetime.now()
    logistic_learner.fit(X_train_all, y_train_all)
    b = datetime.now()
    time_unmitigated = (b - a).total_seconds()


    def UnmitLogistic(X):
        return logistic_learner.predict(X)

    def getViolation(X, Y, A):
        disparity_moment = DemographicParity()
        disparity_moment.load_data(X, Y, sensitive_features=A)
        return disparity_moment.gamma(UnmitLogistic).max()

    def getError(X, Y, A):
        error = ErrorRate()
        error.load_data(X, Y, sensitive_features=A)
        return error.gamma(UnmitLogistic)[0]

    # Training Error & Violation of unmitigated estimator
    log_train_violation_unmitigated = getViolation(X_train_all, y_train_all, A_train_all)
    log_train_error_unmitigated = getError(X_train_all, y_train_all, A_train_all)
    print(f'Logistic Regression : '
          f'Training Violation: {log_train_violation_unmitigated:.6f}; '
          f'Training Error: {log_train_error_unmitigated:.6f}; '
          f'Time: {time_unmitigated:.2f} seconds')

    # Testing error and violation of unmitigated estimator
    log_test_violation_unmitigated = getViolation(X_test_all, y_test_all, A_test_all)
    log_test_error_unmitigated = getError(X_test_all, y_test_all, A_test_all)
    print(f'Logistic Regression : '
          f' Testing Violation: {log_test_violation_unmitigated:.6f}; '
          f' Testing Error: {log_test_error_unmitigated:.6f}')

    results = {
        "time_unmitigated": [time_unmitigated],
        "train_error_unmitigated": [log_train_error_unmitigated],
        "train_vio_unmitigated": [log_train_violation_unmitigated],
        "test_error_unmitigated": [log_test_error_unmitigated],
        "test_vio_unmitigated": [log_test_violation_unmitigated],
    }

    return results


# Fairlearn on full dataset
def run_fairlearn_full(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all, eps):
    _time_expgrad_all = []
    _train_error_expgrad_all = []
    _train_violation_expgrad_all = []
    _test_error_expgrad_all = []
    _test_violation_expgrad_all = []

    num_samples = 10

    for n in range(num_samples):
        expgrad_X_logistic = ExponentiatedGradient(
            LogisticRegression(solver='liblinear', fit_intercept=True),
            constraints=DemographicParity(), eps=eps, nu=1e-6)

        a = datetime.now()
        expgrad_X_logistic.fit(X_train_all, y_train_all,
                               sensitive_features=A_train_all)
        b = datetime.now()
        time_expgrad_all = (b - a).total_seconds()

        def Qexp_all(X):
            return expgrad_X_logistic._pmf_predict(X)[:, 1]

        def getViolation(X, Y, A):
            disparity_moment = DemographicParity()
            disparity_moment.load_data(X, Y, sensitive_features=A)
            return disparity_moment.gamma(Qexp_all).max()

        def getError(X, Y, A):
            error = ErrorRate()
            error.load_data(X, Y, sensitive_features=A)
            return error.gamma(Qexp_all)[0]

        # training violation & error of exp
        train_violation_expgrad_all = getViolation(X_train_all, y_train_all, A_train_all)
        train_error_expgrad_all = getError(X_train_all, y_train_all, A_train_all)
        print(
            f'Exponentiated Gradient on full dataset : '
            f'Training Violation: {train_violation_expgrad_all:.6f}; '
            f'Training Error: {train_error_expgrad_all:.6f}; '
            f'Time: {time_expgrad_all:.2f} seconds')

        # training violation & error of exp
        test_violation_expgrad_all = getViolation(X_test_all, y_test_all, A_test_all)
        test_error_expgrad_all = getError(X_test_all, y_test_all, A_test_all)
        print(
            f'Exponentiated Gradient on full dataset : '
            f' Testing Violation: {test_violation_expgrad_all:.6f}; '
            f' Testing Error: {test_error_expgrad_all:.6f}')

        _time_expgrad_all.append(time_expgrad_all)
        _train_error_expgrad_all.append(train_error_expgrad_all)
        _train_violation_expgrad_all.append(train_violation_expgrad_all)
        _test_error_expgrad_all.append(test_error_expgrad_all)
        _test_violation_expgrad_all.append(test_violation_expgrad_all)

    results = {
        "eps": eps,
        "time_expgrad_all": _time_expgrad_all,
        "train_error_expgrad_all": _train_error_expgrad_all,
        "train_vio_expgrad_all": _train_violation_expgrad_all,
        "test_error_expgrad_all": _test_error_expgrad_all,
        "test_vio_expgrad_all": _test_violation_expgrad_all,
    }

    return results


def main():
    # logging.basicConfig(level=logging.DEBUG)

    host_name = socket.gethostname()
    if "." in host_name:
        host_name = host_name.split(".")[-1]

    eps = 0.05

    unmitigated_results_file_name = \
        f'results/{host_name}/{str(eps)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_unmitigated.json'

    fairlearn_results_file_name = \
        f'results/{host_name}/{str(eps)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_fairlearn.json'

    X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all = load_data()

    unmitigated_results = run_unmitigated(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all)
    fairlearn_full_results = run_fairlearn_full(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all,
                                                A_test_all, eps)

    # Store results
    base_dir = os.path.dirname(unmitigated_results_file_name)
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    with open(unmitigated_results_file_name, 'w') as _file:
        json.dump(unmitigated_results, _file, indent=2)

    base_dir = os.path.dirname(fairlearn_results_file_name)
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    with open(fairlearn_results_file_name, 'w') as _file:
        json.dump(fairlearn_full_results, _file, indent=2)

    print(fairlearn_full_results)


if __name__ == "__main__":
    main()
