import json
import os

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, sem


def mean_confidence_interval(data, confidence: float = 0.95):
    """
    Args:
        data:
        confidence:

    Returns:
        mean and confidence limit values for the given data and confidence
    """
    if data is None:
        return [np.nan, np.nan, np.nan]

    a = np.asarray(data).astype(float)
    n = len(a)
    m, se = np.nanmean(a), sem(a, nan_policy="omit", ddof=1)
    t_value = t.ppf((1.0 + confidence) / 2., n - 1)
    h1 = m - se * t_value
    h2 = m + se * t_value
    return np.array([m, h1, h2])


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
    data_dict = empty_features_dict.copy()
    if os.path.exists(results_file_name):
        with open(results_file_name, 'r') as _file:
            results = json.load(_file)

        for key in feature_names:
            converted_key = key.replace('-', '_') + '_' + suffix
            data_dict[key] = mean_confidence_interval(results[converted_key])
        data_dict[f'train-violation'] = data_dict.pop('train-vio')
    return data_dict



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
    num_hybrids = 3
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
