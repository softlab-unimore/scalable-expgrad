import json
import os
from copy import deepcopy

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from utils import mean_confidence_interval


def get_combined_hybrid(train_err_hybrids: object, train_vio_hybrids: object, alpha: list) -> object:
    # alpha = importance of error vs. violation
    train_err_hybrids = np.array(train_err_hybrids)
    train_vio_hybrids = np.array(train_vio_hybrids)
    n = len(train_err_hybrids)
    if len(train_vio_hybrids) != n:
        raise Exception()

    scores = (alpha * train_err_hybrids + (1 - alpha) * train_vio_hybrids).tolist()

    best_index = scores.index(min(scores))

    return best_index


feature_names = ["time", "train-error", "train-vio", "test-error", "test-vio"]
null_confidence_interval = mean_confidence_interval(None)
empty_features_dict = {key: null_confidence_interval for key in feature_names}


def read_results(results_file_name, suffix: str):
    data_dict = deepcopy(empty_features_dict)
    if os.path.exists(results_file_name):
        with open(results_file_name, 'r') as _file:
            results = json.load(_file)

        for key in feature_names:
            converted_key = key.replace('-', '_') + '_' + suffix
            data_dict[key] = mean_confidence_interval(results[converted_key])

    return data_dict


def read_hybrids_results(hybrid_results_file_name, alphas=[0.5]):
    configuration_map = {'fairlearn_on_subsample': 'expgrad_fracs',
                         'hybrid_1': 'hybrids',
                         'hybrid_2': 'grid_pmf_fracs',
                         'hybrid_3': 'rewts',
                         'hybrid_4': 'rewts_partial',
                         'hybrid_5': 'no_grid_rewts',
                         'hybrid_combo': 'combo'}
    if os.path.exists(hybrid_results_file_name):
        hybrid_results = json.load(open(hybrid_results_file_name, 'r'))
    else:
        hybrid_results = [{}]

    fractions = np.full((1, 3), np.nan)
    grid_fractions = np.full((1, 3), np.nan)
    null_value = np.full((len(hybrid_results), 3), np.nan)
    time_suffix_list = list(configuration_map.values())

    data_dict = {'_'.join(['time', key, 'ci']): deepcopy(null_value) for key in time_suffix_list}

    # baselines
    combo_null_value = {alpha: deepcopy(null_value) for alpha in alphas}
    feature_list = ['error', 'vio']
    model_configuration = list(configuration_map.values())
    for prefix in ['train', 'test']:
        for feature in feature_list:
            for model_conf in model_configuration:
                if model_conf != 'combo':
                    data_dict['_'.join([prefix, feature, model_conf, 'ci'])] = deepcopy(null_value)
                elif model_conf == 'combo':
                    data_dict['_'.join([prefix, feature, model_conf, 'ci'])] = deepcopy(combo_null_value)

    for i, r in enumerate(hybrid_results):
        if r == {}:
            continue

        f = r["frac"]
        grid_f = r["grid_frac"]
        combined_dict = {}
        combined_null_value = {alpha: [None for i in range(len(r["_time_combined"]))] for alpha in alphas}
        for prefix in ['train', 'test']:
            for feature in feature_list:
                combined_dict[f'_{prefix}_{feature}_combined'] = deepcopy(combined_null_value)

        # CONSTRUCT COMBINED
        for j in range(len(r['_train_error_hybrids'])):
            __train_error_hybrids = [r['_train_error_no_grid_rewts'][j], r['_train_error_rewts'][j]]
            __train_vio_hybrids = [r['_train_vio_no_grid_rewts'][j], r['_train_vio_rewts'][j]]
            __test_error_hybrids = [r['_test_error_no_grid_rewts'][j], r['_test_error_rewts'][j]]
            __test_vio_hybrids = [r['_test_vio_no_grid_rewts'][j], r['_test_vio_rewts'][j]]
            for alpha in alphas:
                best_index = get_combined_hybrid(__train_error_hybrids, __train_vio_hybrids, alpha=alpha)
                # Set combined train
                combined_dict['_train_error_combined'][alpha][j] = __train_error_hybrids[best_index]
                combined_dict['_train_vio_combined'][alpha][j] = __train_vio_hybrids[best_index]
                # Set combined test
                combined_dict['_test_error_combined'][alpha][j] = __test_error_hybrids[best_index]
                combined_dict['_test_vio_combined'][alpha][j] = __test_vio_hybrids[best_index]

        fractions[i] = f
        grid_fractions[i] = grid_f

        for key, mapped_key in configuration_map.items():
            if mapped_key not in ['combo', 'expgrad_fracs']:
                data_dict[f'time_{mapped_key}_ci'][i] = mean_confidence_interval(
                    r[f'_time_{key.replace("_", "")}'])
        data_dict['time_expgrad_fracs_ci'][i] = mean_confidence_interval(r['_time_expgrad_fracs'])
        data_dict['time_combo_ci'][i] = mean_confidence_interval(r['_time_combined'])

        for prefix in ['train', 'test']:
            for feature in feature_list:
                for model_conf in model_configuration:
                    if model_conf != 'combo':
                        val = mean_confidence_interval(r[f'_{prefix}_{feature}_{model_conf}'])
                        data_dict[f'{prefix}_{feature}_{model_conf}_ci'][i] = val
                    elif model_conf == 'combo':
                        for alpha in alphas:
                            val = mean_confidence_interval(combined_dict[f'_{prefix}_{feature}_combined'][alpha])
                            data_dict[f'{prefix}_{feature}_{model_conf}_ci'][alpha][i] = val

        # Hybrid combined
        for prefix in ['train', 'test']:
            for feature in feature_list:
                for alpha in alphas:
                    val = mean_confidence_interval(combined_dict[f'_{prefix}_{feature}_combined'][alpha])
                    data_dict[f'{prefix}_{feature}_combo_ci'][alpha][i] = val
    print(grid_fractions)

    final_features_map = {"times": 'time',
                          "train_errors": 'train_error',
                          "train_violations": 'train_vio',
                          "test_errors": 'test_error',
                          "test_violations": 'test_vio'}
    ret_dict = {"fractions": fractions,
                "grid_fractions": grid_fractions}
    for key, mapped_key in configuration_map.items():
        tmp_dict = {}
        for feature, mapped_feature in final_features_map.items():
            tmp_dict[feature] = data_dict[f'{mapped_feature}_{mapped_key}_ci']
        ret_dict[key] = tmp_dict
    return ret_dict


def time_plot(results, num_hybrids=3):
    # Time Plots

    # grid_fractions = hybrid_results["grid_fractions"]
    grid_fraction = 0.5

    data = np.array([results[n]["unmitigated"].get("time") for n in results])
    plt.fill_between(data_sizes, data[:, 1], data[:, 2], color='b', alpha=0.3)
    plt.plot(data_sizes, data[:, 0], 'bo-', label="unmitigated")

    data = np.array([results[n]["fairlearn"].get("time") for n in results])
    # print(data)
    plt.fill_between(data_sizes, data[:, 1], data[:, 2], color='r', alpha=0.3)
    plt.plot(data_sizes, data[:, 0], 'ro-', label="expgrad full")

    # data = np.array([results[n]["hybrids"]["hybrid_3"].get("times") for n in results])[:,0,:]
    # print(data)
    # plt.fill_between(data_sizes, data[:,1], data[:,2], color='k', alpha=0.3)
    # plt.plot(data_sizes, data[:,0], 'ko-', label="hybrid 3 (GS + LP)")

    alpha = alphas[0]
    cols = list(mcolors.TABLEAU_COLORS.keys())
    for i in range(num_hybrids):
        c = cols[i]
        grid_f = results[10000]["hybrids"][i]["grid_fractions"][0][0]

        data = np.array([results[n]["hybrids"][i]["hybrid_combo"]["times"]
                         if i in results[n]["hybrids"] else np.array([[np.nan, np.nan, np.nan]])
                         for n in results])[:, 0, :]
        plt.fill_between(data_sizes, data[:, 1], data[:, 2], color=c, alpha=0.3)
        plt.plot(data_sizes, data[:, 0], c=c, marker="o", label=f"hybrid grid-f={grid_f}, combo alpha={alpha})")

        c = cols[5 + i]
        data = np.array([results[n]["hybrids"][i]["fairlearn_on_subsample"]["times"]
                         if i in results[n]["hybrids"] else np.array([[np.nan, np.nan, np.nan]])
                         for n in results])[:, 0, :]
        plt.fill_between(data_sizes, data[:, 1], data[:, 2], color=c, alpha=0.3)
        plt.plot(data_sizes, data[:, 0], c=c, marker="o", label=f"FL on subsample")

    plt.xlabel("Dataset Size (log scale)")
    plt.ylabel("Time (second)")
    plt.title("Fitting time v.s. Data Size")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 8)
    plt.show()


def error_plot(results, dataset_portion='train', num_hybrids=3):
    cols = list(mcolors.TABLEAU_COLORS.keys())

    empty = np.array([np.nan, np.nan, np.nan])

    errors = {"expgrad_alls": (
        "r", "expgrad on full", np.array([results[n]["fairlearn"][f"{dataset_portion}-error"] for n in data_sizes])),
    }

    violations = {
        "expgrad_alls": ("r", "expgrad on full",
                         np.array([results[n]["fairlearn"][f"{dataset_portion}-vio"] for n in data_sizes])),
    }

    for h in range(num_hybrids):
        grid_f = results[10000]["hybrids"][h]["grid_fractions"][0][0]

        errors[f"no_grid_rewts_{h}"] = ("y", f"hybrid-5 {grid_f} (LP)",
                                        np.array([results[n]["hybrids"][h]["hybrid_5"]["train_errors"][0]
                                                  if h in results[n]["hybrids"] else empty
                                                  for n in data_sizes]))

        violations[f"no_grid_rewts_{h}"] = ("y", f"hybrid-5 {grid_f} (LP)",
                                            np.array([results[n]["hybrids"][h]["hybrid_5"][
                                                          f"{dataset_portion}_violations"][0]
                                                      if h in results[n]["hybrids"] else empty
                                                      for n in data_sizes]))

        # Add alpha combos
        for i, alpha in enumerate(alphas):
            grid_f = results[10000]["hybrids"][h]["grid_fractions"][0][0]

            errors[f"combo_{h}[{alpha}]"] = (cols[h * (i + 1)], f"hybrid combo {grid_f} (alpha={alpha})",
                                             np.array(
                                                 [results[n]["hybrids"][h]["hybrid_combo"][f"{dataset_portion}_errors"][
                                                      alpha][0]
                                                  if h in results[n]["hybrids"] else empty
                                                  for n in data_sizes]))
            violations[f"combo_{h}[{alpha}]"] = (cols[h * (i + 1)], f"hybrid combo {grid_f} (alpha={alpha})",
                                                 np.array([results[n]["hybrids"][h]["hybrid_combo"][
                                                               f"{dataset_portion}_violations"][
                                                               alpha][0]
                                                           if h in results[n]["hybrids"] else empty
                                                           for n in data_sizes]))

    # Plot errors
    for k, (c1, label, means) in errors.items():
        plt.fill_between(data_sizes, means[:, 1], means[:, 2], color=c1, alpha=0.3)
        plt.plot(data_sizes, means[:, 0], c1, marker="o", label=label)

    ylabel = f'{"Training" if dataset_portion == "train" else dataset_portion.capitalize()} Error'
    plt.xlabel("Dataset Size (log scale)")
    plt.ylabel(ylabel)
    # plt.title('Training Error v.s. Fraction')
    plt.xscale("log")
    # plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot violations
    for k, (c1, label, means) in violations.items():
        plt.fill_between(data_sizes, means[:, 1], means[:, 2], color=c1, alpha=0.3)
        plt.plot(data_sizes, means[:, 0], c1, marker="o", label=label)

    plt.xlabel("Dataset Size (log scale)")
    plt.ylabel(ylabel.replace('Error', 'Violation'))
    # plt.title('Training Violation v.s. Fraction')
    plt.xscale("log")
    # plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":

    alphas = [0.5]  # , 0.5, 0.95]

    result_file_names = {
        10000: {
            "unmitigated": "2021-02-23_07-02-30_unmitigated.json",
            "fairlearn": "2021-02-23_07-03-19_fairlearn_e0.05.json",
            "hybrids": [
                "2021-02-23_07-05-29_hybrids_e0.05_g0.5.json",
                "2021-02-24_08-55-22_hybrids_e0.05_g0.2.json",
                "2021-02-24_09-10-41_hybrids_e0.05_g0.1.json",
            ],
        },
        100000: {
            "unmitigated": "2021-02-24_05-41-01_unmitigated.json",
            "fairlearn": "2021-02-23_09-37-14_fairlearn_e0.05.json",
            "hybrids": [
                "2021-02-23_07-19-34_hybrids_e0.05_g0.5.json",
                "2021-02-24_08-54-44_hybrids_e0.05_g0.2.json",
                "2021-02-24_09-12-38_hybrids_e0.05_g0.1.json",
            ],
        },
        1000000: {
            "unmitigated": "2021-02-24_05-41-18_unmitigated.json",
            "fairlearn": "2021-02-23_11-03-09_fairlearn_e0.05.json",
            "hybrids": [
                "2021-02-23_10-46-25_hybrids_e0.05_g0.5.json",
                "2021-02-24_06-04-24_hybrids_e0.05_g0.2.json",
                "2021-02-24_09-25-47_hybrids_e0.05_g0.1.json",
            ],
        },
        10000000: {
            "unmitigated": None,
            "fairlearn": None,
            "hybrids": [
                None,
                None,
                "2021-02-24_12-01-21_hybrids_e0.05_g0.1.json",
            ],
        },
    }

    data_sizes = result_file_names.keys()

    _results = {}

    for n in data_sizes:
        base_dir = f"results/yeeha/synth_n{n}_f3_t0.5_t00.3_t10.6_tr0.3_v1"
        unmitigated_results_file_name = f"{base_dir}/{result_file_names[n]['unmitigated']}"
        fairlearn_results_file_name = f"{base_dir}/{result_file_names[n]['fairlearn']}"

        unmitigated_results = read_results(unmitigated_results_file_name, 'unmitigated')
        fairlearn_results = read_results(fairlearn_results_file_name, 'expgrad_all')
        hybrid_results = {}
        for i, hybrid_results_file_name in enumerate(result_file_names[n]['hybrids']):
            hybrid_results[i] = read_hybrids_results(f"{base_dir}/{hybrid_results_file_name}")

        _results[n] = {
            "unmitigated": unmitigated_results,
            "fairlearn": fairlearn_results,
            "hybrids": hybrid_results,
        }
    results = _results

    time_plot(results)
    error_plot(results)
    error_plot(results, dataset_portion='test')


