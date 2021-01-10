import json
import os
import socket
from datetime import datetime

from fairlearn.reductions import DemographicParity, ErrorRate, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression

from utils import load_data


def run_unmitigated(X_train_all, y_train_all, A_train_all):
    # Unmitigated LogRes

    logistic_learner = LogisticRegression(solver='liblinear', fit_intercept=True)

    a = datetime.now()
    logistic_learner.fit(X_train_all, y_train_all)
    b = datetime.now()
    time_unmitigated = (b - a).total_seconds()

    # Error & Violation of unmitigated estimator
    def UnmitLogistic(X):
        return logistic_learner.predict(X)

    # violation
    disparity_moment = DemographicParity()
    disparity_moment.load_data(X_train_all, y_train_all,
                               sensitive_features=A_train_all)
    log_violation_unmitigated = disparity_moment.gamma(UnmitLogistic).max()

    # error
    error = ErrorRate()
    error.load_data(X_train_all, y_train_all, sensitive_features=A_train_all)
    log_error_unmitigated = error.gamma(UnmitLogistic)[0]
    print('Logistic Regression - Time: {} seconds; Violation: {}; Error: {}'.format(
        time_unmitigated, log_violation_unmitigated, log_error_unmitigated))

    results = {
        "time_unmitigated": [time_unmitigated],
        "error_unmitigated": [log_error_unmitigated],
        "vio_unmitigated": [log_error_unmitigated],
    }

    return results


# Fairlearn on full dataset
def run_fairlearn_full(X_train_all, y_train_all, A_train_all, eps):
    _time_expgrad_all = []
    _error_expgrad_all = []
    _violation_expgrad_all = []

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

        # violation of log res
        disparity_moment = DemographicParity()
        disparity_moment.load_data(X_train_all, y_train_all,
                                   sensitive_features=A_train_all)
        violation_expgrad_all = disparity_moment.gamma(Qexp_all).max()

        # error of log res
        error = ErrorRate()
        error.load_data(X_train_all, y_train_all,
                        sensitive_features=A_train_all)
        error_expgrad_all = error.gamma(Qexp_all)[0]

        print('Exponentiated gradient on full dataset : Time: {} seconds; Violation: {}; Errror: {}'.format(
            time_expgrad_all, violation_expgrad_all, error_expgrad_all))

        _time_expgrad_all.append(time_expgrad_all)
        _error_expgrad_all.append(error_expgrad_all)
        _violation_expgrad_all.append(violation_expgrad_all)

    results = {
        "eps": eps,
        "time_expgrad_all": _time_expgrad_all,
        "error_expgrad_all": _error_expgrad_all,
        "vio_expgrad_all": _violation_expgrad_all,
    }

    return results


def main():
    host_name = socket.gethostname()
    if "." in host_name:
        host_name = host_name.split(".")[-1]

    eps = 0.05

    unmitigated_results_file_name = \
        f'results/{host_name}/{str(eps)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_unmitigated.json'

    fairlearn_results_file_name = \
        f'results/{host_name}/{str(eps)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_fairlearn.json'

    X_train_all, y_train_all, A_train_all = load_data()

    unmitigated_results = run_unmitigated(X_train_all, y_train_all, A_train_all)
    fairlearn_full_results = run_fairlearn_full(X_train_all, y_train_all, A_train_all, eps)

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


if __name__ == "__main__":
    main()
