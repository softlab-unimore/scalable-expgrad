from copy import deepcopy
from datetime import datetime

from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression

from hybrid_methods import get_metrics


def run_unmitigated(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all, random_seed=0):
    results = []
    data_dict = {'model_name': 'unmitigated', 'time': 0, 'phase': 'model name', 'random_seed': random_seed}
    datasets_dict = {'train': [X_train_all, y_train_all, A_train_all],
                     'test': [X_test_all, y_test_all, A_test_all]}
    # Unmitigated LogRes
    logistic_learner = LogisticRegression(solver='liblinear', fit_intercept=True, random_state=random_seed)

    a = datetime.now()
    logistic_learner.fit(X_train_all, y_train_all)
    b = datetime.now()
    time_unmitigated = (b - a).total_seconds()

    metrics_res = get_metrics(datasets_dict, logistic_learner.predict)
    data_dict.update(**metrics_res)
    time_expgrad_frac_dict = {'time': time_unmitigated, 'phase': 'train'}
    data_dict.update(**time_expgrad_frac_dict)
    results.append(deepcopy(data_dict))
    return results


# Fairlearn on full dataset
def run_fairlearn_full(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all, eps, random_seed=0):
    results = []
    data_dict = {'eps': eps, 'model_name': 'fairlearn_full', 'time': 0, 'phase': 'fairlearn_full',
                 'random_seed': random_seed}
    datasets_dict = {'train': [X_train_all, y_train_all, A_train_all],
                     'test': [X_test_all, y_test_all, A_test_all]}
    num_samples = 1
    for n in range(num_samples):
        expgrad_X_logistic = ExponentiatedGradient(
            LogisticRegression(solver='liblinear', fit_intercept=True, random_state=random_seed),
            constraints=DemographicParity(), eps=eps, nu=1e-6)
        print("Fitting Exponentiated Gradient on full dataset...")

        a = datetime.now()
        expgrad_X_logistic.fit(X_train_all, y_train_all, sensitive_features=A_train_all)
        b = datetime.now()
        time_expgrad_all = (b - a).total_seconds()

        Qexp_all = lambda X: expgrad_X_logistic._pmf_predict(X)[:, 1]
        metrics_res = get_metrics(datasets_dict, Qexp_all)
        data_dict.update(**metrics_res)
        time_expgrad_frac_dict = {'time': time_expgrad_all, 'phase': 'train'}
        data_dict.update(**time_expgrad_frac_dict)
        results.append(deepcopy(data_dict))

        print(f'Exponentiated Gradient on full dataset : ')
        for key, value in data_dict.items():
            if key not in ['model_name', 'phase']:
                print(f'{key} : {value:.6f}')
    return results

# def main():
# 
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
#     unmitigated_results_file_name = \
#         f'results/{host_name}/{dataset_str}/{str(eps)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_unmitigated.json'
# 
#     fairlearn_results_file_name = \
#         f'results/{host_name}/{dataset_str}/{str(eps)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_fairlearn.json'
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
#     print(unmitigated_results_file_name)
#     print(fairlearn_results_file_name)
# 
#     unmitigated_results = run_unmitigated(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all)
#     fairlearn_full_results = run_fairlearn_full(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all,
#                                                 A_test_all, eps)
# 
#     # Store results
#     base_dir = os.path.dirname(unmitigated_results_file_name)
#     if not os.path.isdir(base_dir):
#         os.makedirs(base_dir, exist_ok=True)
#     with open(unmitigated_results_file_name, 'w') as _file:
#         json.dump(unmitigated_results, _file, indent=2)
# 
#     base_dir = os.path.dirname(fairlearn_results_file_name)
#     if not os.path.isdir(base_dir):
#         os.makedirs(base_dir, exist_ok=True)
#     with open(fairlearn_results_file_name, 'w') as _file:
#         json.dump(fairlearn_full_results, _file, indent=2)
# 
#     print(fairlearn_full_results)
# 
# 
# if __name__ == "__main__":
#     main()
