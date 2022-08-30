import json
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from graphic_synth import mean_confidence_interval, get_combined_hybrid


def load_data_adult_old(unmitigated_results, fairlearn_results, hybrid_results, alphas=[0.05, 0.5, 0.95]):
    # Unmitigated results
    time_unmitigated = unmitigated_results["time_unmitigated"]
    train_error_unmitigated = unmitigated_results["train_error_unmitigated"]
    train_violation_unmitigated = unmitigated_results["train_vio_unmitigated"]
    test_error_unmitigated = unmitigated_results["test_error_unmitigated"]
    test_violation_unmitigated = unmitigated_results["test_vio_unmitigated"]

    # Fairlearn results
    time_expgrad_all = fairlearn_results["time_expgrad_all"]
    train_error_expgrad_all = fairlearn_results["train_error_expgrad_all"]
    train_violation_expgrad_all = fairlearn_results["train_vio_expgrad_all"]
    test_error_expgrad_all = fairlearn_results["test_error_expgrad_all"]
    test_violation_expgrad_all = fairlearn_results["test_vio_expgrad_all"]

    # Hybrid results
    fractions = []

    time_expgrad_fracs_ci = np.full((len(hybrid_results), 3), np.nan)
    time_hybrid1_ci = np.full((len(hybrid_results), 3), np.nan)
    time_hybrid2_ci = np.full((len(hybrid_results), 3), np.nan)
    time_hybrid3_ci = np.full((len(hybrid_results), 3), np.nan)
    time_hybrid4_ci = np.full((len(hybrid_results), 3), np.nan)
    time_hybrid5_ci = np.full((len(hybrid_results), 3), np.nan)
    time_hybrid_combo_ci = np.full((len(hybrid_results), 3), np.nan)
    time_unmitigated_ci = np.full((len(hybrid_results), 3), np.nan)
    time_expgrad_alls_ci = np.full((len(hybrid_results), 3), np.nan)

    # baselines
    train_error_expgrad_fracs_ci = np.full((len(hybrid_results), 3), np.nan)
    train_error_expgrad_alls_ci = np.full((len(hybrid_results), 3), np.nan)
    train_error_unmitigated_ci = np.full((len(hybrid_results), 3), np.nan)

    train_vio_expgrad_alls_ci = np.full((len(hybrid_results), 3), np.nan)
    train_vio_expgrad_fracs_ci = np.full((len(hybrid_results), 3), np.nan)
    train_vio_unmitigated_ci = np.full((len(hybrid_results), 3), np.nan)

    test_error_expgrad_fracs_ci = np.full((len(hybrid_results), 3), np.nan)
    test_error_expgrad_alls_ci = np.full((len(hybrid_results), 3), np.nan)
    test_error_unmitigated_ci = np.full((len(hybrid_results), 3), np.nan)

    test_vio_expgrad_alls_ci = np.full((len(hybrid_results), 3), np.nan)
    test_vio_expgrad_fracs_ci = np.full((len(hybrid_results), 3), np.nan)
    test_vio_unmitigated_ci = np.full((len(hybrid_results), 3), np.nan)

    # Hybrid 1
    train_error_hybrids_ci = np.full((len(hybrid_results), 3), np.nan)
    train_vio_hybrids_ci = np.full((len(hybrid_results), 3), np.nan)
    test_error_hybrids_ci = np.full((len(hybrid_results), 3), np.nan)
    test_vio_hybrids_ci = np.full((len(hybrid_results), 3), np.nan)

    # Hybrid 2
    train_error_grid_pmf_fracs_ci = np.full((len(hybrid_results), 3), np.nan)
    train_vio_grid_pmf_fracs_ci = np.full((len(hybrid_results), 3), np.nan)
    test_error_grid_pmf_fracs_ci = np.full((len(hybrid_results), 3), np.nan)
    test_vio_grid_pmf_fracs_ci = np.full((len(hybrid_results), 3), np.nan)

    # Hybrid 3
    train_vio_rewts_pmf_ci = np.full((len(hybrid_results), 3), np.nan)
    train_error_rewts_pmf_ci = np.full((len(hybrid_results), 3), np.nan)
    test_vio_rewts_pmf_ci = np.full((len(hybrid_results), 3), np.nan)
    test_error_rewts_pmf_ci = np.full((len(hybrid_results), 3), np.nan)

    # Hybrid 4
    train_error_rewts_partial_ci = np.full((len(hybrid_results), 3), np.nan)
    train_vio_rewts_partial_ci = np.full((len(hybrid_results), 3), np.nan)
    test_error_rewts_partial_ci = np.full((len(hybrid_results), 3), np.nan)
    test_vio_rewts_partial_ci = np.full((len(hybrid_results), 3), np.nan)

    # Hybrid 5
    train_error_no_grid_rewts_ci = np.full((len(hybrid_results), 3), np.nan)
    train_vio_no_grid_rewts_ci = np.full((len(hybrid_results), 3), np.nan)
    test_error_no_grid_rewts_ci = np.full((len(hybrid_results), 3), np.nan)
    test_vio_no_grid_rewts_ci = np.full((len(hybrid_results), 3), np.nan)

    # Hybrid Combined
    train_err_combo_ci = {alpha: np.full((len(hybrid_results), 3), np.nan) for alpha in alphas}
    train_vio_combo_ci = {alpha: np.full((len(hybrid_results), 3), np.nan) for alpha in alphas}
    test_err_combo_ci = {alpha: np.full((len(hybrid_results), 3), np.nan) for alpha in alphas}
    test_vio_combo_ci = {alpha: np.full((len(hybrid_results), 3), np.nan) for alpha in alphas}

    for i, r in enumerate(hybrid_results):
        f = r["frac"]
        _time_expgrad_fracs = r["_time_expgrad_fracs"]
        _time_hybrid1 = r["_time_hybrid1"]
        _time_hybrid2 = r["_time_hybrid2"]
        _time_hybrid3 = r["_time_hybrid3"]
        _time_hybrid4 = r["_time_hybrid4"]
        _time_hybrid5 = r["_time_hybrid5"]
        _time_combo = r["_time_combined"]

        _train_error_expgrad_fracs = r["_train_error_expgrad_fracs"]
        _train_vio_expgrad_fracs = r["_train_vio_expgrad_fracs"]
        _train_error_hybrids = r["_train_error_hybrids"]
        _train_vio_hybrids = r["_train_vio_hybrids"]
        _train_error_grid_pmf_fracs = r["_train_error_grid_pmf_fracs"]
        _train_vio_grid_pmf_fracs = r["_train_vio_grid_pmf_fracs"]
        _train_error_rewts = r["_train_error_rewts"]
        _train_vio_rewts = r["_train_vio_rewts"]
        _train_error_rewts_partial = r["_train_error_rewts_partial"]
        _train_vio_rewts_partial = r["_train_vio_rewts_partial"]
        _train_error_no_grid_rewts = r["_train_error_no_grid_rewts"]
        _train_vio_no_grid_rewts = r["_train_vio_no_grid_rewts"]
        #     _train_err_combined = r["_train_error_combined"]
        #     _train_vio_combined = r["_train_vio_combined"]

        _train_err_combined = {alpha: [None for i in range(len(_time_combo))] for alpha in alphas}
        _train_vio_combined = {alpha: [None for i in range(len(_time_combo))] for alpha in alphas}

        _test_error_expgrad_fracs = r["_test_error_expgrad_fracs"]
        _test_vio_expgrad_fracs = r["_test_vio_expgrad_fracs"]
        _test_error_hybrids = r["_test_error_hybrids"]
        _test_vio_hybrids = r["_test_vio_hybrids"]
        _test_error_grid_pmf_fracs = r["_test_error_grid_pmf_fracs"]
        _test_vio_grid_pmf_fracs = r["_test_vio_grid_pmf_fracs"]
        _test_error_rewts = r["_test_error_rewts"]
        _test_vio_rewts = r["_test_vio_rewts"]
        _test_error_rewts_partial = r["_test_error_rewts_partial"]
        _test_vio_rewts_partial = r["_test_vio_rewts_partial"]
        _test_error_no_grid_rewts = r["_test_error_no_grid_rewts"]
        _test_vio_no_grid_rewts = r["_test_vio_no_grid_rewts"]
        #     _test_err_combined = r["_test_error_combined"]
        #     _test_vio_combined = r["_test_vio_combined"]

        _test_err_combined = {alpha: [None for i in range(len(_time_combo))] for alpha in alphas}
        _test_vio_combined = {alpha: [None for i in range(len(_time_combo))] for alpha in alphas}

        # CONSTRUCT COMBINED
        for j in range(len(_train_error_hybrids)):
            __train_err_hybrids = [_train_error_no_grid_rewts[j], _train_error_rewts[j]]
            __train_vio_hybrids = [_train_vio_no_grid_rewts[j], _train_vio_rewts[j]]
            __test_err_hybrids = [_test_error_no_grid_rewts[j], _test_error_rewts[j]]
            __test_vio_hybrids = [_test_vio_no_grid_rewts[j], _test_vio_rewts[j]]
            for alpha in alphas:
                best_index = get_combined_hybrid(__train_err_hybrids, __train_vio_hybrids, alpha=alpha)
                # Set combined train
                _train_err_combined[alpha][j] = __train_err_hybrids[best_index]
                _train_vio_combined[alpha][j] = __train_vio_hybrids[best_index]
                # Set combined test
                _test_err_combined[alpha][j] = __test_err_hybrids[best_index]
                _test_vio_combined[alpha][j] = __test_vio_hybrids[best_index]

        fractions.append(f)

        time_expgrad_alls_ci[i] = mean_confidence_interval(time_expgrad_all)
        time_unmitigated_ci[i] = mean_confidence_interval(time_unmitigated)
        time_expgrad_fracs_ci[i] = mean_confidence_interval(_time_expgrad_fracs)

        time_hybrid1_ci[i] = mean_confidence_interval(_time_hybrid1)
        time_hybrid2_ci[i] = mean_confidence_interval(_time_hybrid2)
        time_hybrid3_ci[i] = mean_confidence_interval(_time_hybrid3)
        time_hybrid4_ci[i] = mean_confidence_interval(_time_hybrid4)
        time_hybrid5_ci[i] = mean_confidence_interval(_time_hybrid5)
        time_hybrid_combo_ci[i] = mean_confidence_interval(_time_combo)

        # baseline
        train_error_expgrad_alls_ci[i] = mean_confidence_interval(train_error_expgrad_all)
        train_error_unmitigated_ci[i] = mean_confidence_interval(train_error_unmitigated)

        train_vio_expgrad_alls_ci[i] = mean_confidence_interval(train_violation_expgrad_all)
        train_vio_unmitigated_ci[i] = mean_confidence_interval(train_violation_unmitigated)

        test_error_expgrad_alls_ci[i] = mean_confidence_interval(test_error_expgrad_all)
        test_error_unmitigated_ci[i] = mean_confidence_interval(test_error_unmitigated)

        test_vio_expgrad_alls_ci[i] = mean_confidence_interval(test_violation_expgrad_all)
        test_vio_unmitigated_ci[i] = mean_confidence_interval(test_violation_unmitigated)

        # exp frac
        train_error_expgrad_fracs_ci[i] = mean_confidence_interval(_train_error_expgrad_fracs)
        train_vio_expgrad_fracs_ci[i] = mean_confidence_interval(_train_vio_expgrad_fracs)

        test_error_expgrad_fracs_ci[i] = mean_confidence_interval(_test_error_expgrad_fracs)
        test_vio_expgrad_fracs_ci[i] = mean_confidence_interval(_test_vio_expgrad_fracs)

        # Hybrid 1
        train_error_hybrids_ci[i] = mean_confidence_interval(_train_error_hybrids)
        train_vio_hybrids_ci[i] = mean_confidence_interval(_train_vio_hybrids)

        test_error_hybrids_ci[i] = mean_confidence_interval(_test_error_hybrids)
        test_vio_hybrids_ci[i] = mean_confidence_interval(_test_vio_hybrids)

        # Hybrid 2
        train_error_grid_pmf_fracs_ci[i] = mean_confidence_interval(_train_error_grid_pmf_fracs)
        train_vio_grid_pmf_fracs_ci[i] = mean_confidence_interval(_train_vio_grid_pmf_fracs)

        test_error_grid_pmf_fracs_ci[i] = mean_confidence_interval(_test_error_grid_pmf_fracs)
        test_vio_grid_pmf_fracs_ci[i] = mean_confidence_interval(_test_vio_grid_pmf_fracs)

        # Hybrid 3: re-weight using LP
        train_error_rewts_pmf_ci[i] = mean_confidence_interval(_train_error_rewts)
        train_vio_rewts_pmf_ci[i] = mean_confidence_interval(_train_vio_rewts)

        test_error_rewts_pmf_ci[i] = mean_confidence_interval(_test_error_rewts)
        test_vio_rewts_pmf_ci[i] = mean_confidence_interval(_test_vio_rewts)

        # Hybrid 4
        train_error_rewts_partial_ci[i] = mean_confidence_interval(_train_error_rewts_partial)
        train_vio_rewts_partial_ci[i] = mean_confidence_interval(_train_vio_rewts_partial)

        test_error_rewts_partial_ci[i] = mean_confidence_interval(_test_error_rewts_partial)
        test_vio_rewts_partial_ci[i] = mean_confidence_interval(_test_vio_rewts_partial)

        # Hybrid 5
        train_error_no_grid_rewts_ci[i] = mean_confidence_interval(_train_error_no_grid_rewts)
        train_vio_no_grid_rewts_ci[i] = mean_confidence_interval(_train_vio_no_grid_rewts)

        test_error_no_grid_rewts_ci[i] = mean_confidence_interval(_test_error_no_grid_rewts)
        test_vio_no_grid_rewts_ci[i] = mean_confidence_interval(_test_vio_no_grid_rewts)

        # Hybrid combined
        for alpha in alphas:
            train_err_combo_ci[alpha][i] = mean_confidence_interval(_train_err_combined[alpha])
            train_vio_combo_ci[alpha][i] = mean_confidence_interval(_train_vio_combined[alpha])

            test_err_combo_ci[alpha][i] = mean_confidence_interval(_test_err_combined[alpha])
            test_vio_combo_ci[alpha][i] = mean_confidence_interval(_test_vio_combined[alpha])


def load_data_adult(unmitigated_results, fairlearn_results, hybrid_results, alphas=[0.05, 0.5, 0.95]):
    # Unmitigated results
    feature_names = ['time']
    for phase in ['train', 'test']:
        for feature in ['error', 'vio']:
            feature_names.append(f'{phase}_{feature}')
    base_models = ['unmitigated', 'expgrad_all']

    data_dict = {}
    for model, source_dict in zip(base_models, [unmitigated_results, fairlearn_results]):
        for feature in feature_names:
            key = f'{feature}_{model}'
            data_dict[key] = source_dict[key]

    configuration_map = {'expgrad_fracs': 'expgrad_fracs',
                         'hybrid1': 'hybrids',
                         'hybrid2': 'grid_pmf_fracs',
                         'hybrid3': 'rewts',
                         'hybrid4': 'rewts_partial',
                         'hybrid5': 'no_grid_rewts',
                         'combined': 'combined'}

    null_value = np.full((len(hybrid_results), 3), np.nan)
    null_combined = {alpha: np.full((len(hybrid_results), 3), np.nan) for alpha in alphas}
    for feature in feature_names:
        for model in list(configuration_map.keys()) + base_models:
            key = f'{feature}_{model}_ci'
            if model == 'combined' and feature != 'time':
                data_dict[key] = deepcopy(null_combined)
            else:
                data_dict[key] = deepcopy(null_value)

    fractions = []
    for i, r in enumerate(hybrid_results):
        f = r["frac"]
        for feature in feature_names:
            for model in base_models:
                key = f'{feature}_{model}'
                data_dict[key + '_ci'][i] = mean_confidence_interval(data_dict[key])

            for model, mapped_key in configuration_map.items():
                key1 = f'{feature}_{model}_ci'
                if feature == 'time':
                    key2 = f'_{feature}_{model}'
                else:
                    key2 = f'_{feature}_{mapped_key}'
                if model != 'combined' or feature == 'time':
                    data_dict[key1][i] = mean_confidence_interval(r[key2])

        # combined
        new_null_combined = {alpha: [None for i in range(len(r['_time_combined']))] for alpha
                             in alphas}
        for j in range(len(r["_train_error_hybrids"])):
            for feature in np.setdiff1d(feature_names, 'time'):
                data_dict[f'_{feature}_combined'] = deepcopy(new_null_combined)
        for j in range(len(r["_train_error_hybrids"])):
            for feature in np.setdiff1d(feature_names, 'time'):
                data_dict[f'__{feature}_hybrids'] = [r[f"_{feature}_no_grid_rewts"][j], r[f"_{feature}_rewts"][j]]
            for alpha in alphas:
                best_index = get_combined_hybrid(data_dict['__train_error_hybrids'],
                                                 data_dict['__train_vio_hybrids'], alpha=alpha)
                for feature in np.setdiff1d(feature_names, 'time'):
                    data_dict[f'_{feature}_combined'][alpha][j] = data_dict[f'__{feature}_hybrids'][best_index]
        for alpha in alphas:
            for feature in np.setdiff1d(feature_names, 'time'):
                data_dict[f'{feature}_combined_ci'][alpha][i] = mean_confidence_interval(
                    data_dict[f'_{feature}_combined'][alpha])
        fractions.append(f)
    data_dict['fractions'] = fractions
    return data_dict

def plot_erro_vio_vs_fraction(data_dict, alphas=[0.05, 0.5, 0.95], phase='train'):
    cols = list(mcolors.TABLEAU_COLORS.keys())
    fr = data_dict['fractions']

    errors = {
        #     "unmitigated": ("b", "unmitigated", train_error_unmitigated_ci, None),
        #     "expgrad_fracs": ("g", "expgrad on partial", train_error_expgrad_fracs_ci, None),
        "expgrad_alls": ("r", "expgrad on full", data_dict[f'{phase}_error_expgrad_all_ci'], None),
        "no_grid_rewts": ("y", "hybrid 5 (LP)", data_dict[f'{phase}_error_hybrid5_ci'], None),
        #     "rewts_pmf": ("k", "hybrid 3 (GS + LP)", train_error_rewts_pmf_ci, None),
        #     "grid_pmf_fracs": ("c", "hybrid 2 (GS + pmf_predict)", train_error_grid_pmf_fracs_ci, None),
        #     "rewts_partial": ("m", "hybrid 4 (GS + LP+)", train_error_rewts_partial_ci, None),
        #     "hybrid_1": ("k", "hybrid 1 (GS only)", train_error_hybrids_ci, None),
    }

    violations = {
        #     "unmitigated": ("b", "unmitigated", train_vio_unmitigated_ci, None),
        #     "expgrad_fracs": ("g", "expgrad on partial", train_vio_expgrad_fracs_ci, None),
        "expgrad_alls": ("r", "expgrad on full", data_dict[f'{phase}_vio_expgrad_all_ci'], None),
        "no_grid_rewts": ("y", "hybrid 5 (LP)", data_dict[f'{phase}_vio_hybrid5_ci'], None),
        #     "rewts_pmf": ("k", "hybrid 3 (GS + LP)", train_vio_rewts_pmf_ci, None),
        #     "grid_pmf_fracs": ("c", "hybrid 2 (GS + pmf_predict)", train_vio_grid_pmf_fracs_ci, None),
        #     "rewts_partial": ("m", "hybrid 4 (GS + LP+)", train_vio_rewts_partial_ci, None),
        #     "hybrid_1": ("k", "hybrid 1 (GS only)", train_vio_hybrids_ci, None),
    }

    # Add alpha combos
    for i, alpha in enumerate(alphas):
        errors[f"combined[{alpha}]"] = (cols[i], f"hybrid combined (alpha={alpha})", data_dict[f'{phase}_error_combined_ci'][alpha], None)
        violations[f"combined[{alpha}]"] = (cols[i], f"hybrid combined (alpha={alpha})", data_dict[f'{phase}_vio_combined_ci'][alpha], None)

    # Plot errors
    for k, (c1, label, means, stds) in errors.items():
        plt.fill_between(fr, means[:, 1], means[:, 2], color=c1, alpha=0.3)
        plt.plot(fr, means[:, 0], c1, marker="o", label=label)

    plt.xlabel('Fraction (log scale)')
    plt.ylabel(f'{phase.capitalize()} Error')
    plt.title(f'{phase.capitalize()} Error v.s. Fraction')
    plt.xscale("log")
    # plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot violations
    for k, (c1, label, means, stds) in violations.items():
        plt.fill_between(fr, means[:, 1], means[:, 2], color=c1, alpha=0.3)
        plt.plot(fr, means[:, 0], c1, marker="o", label=label)

    plt.xlabel('Fraction (log scale)')
    plt.ylabel(f'{phase.capitalize()} Violation')
    plt.title(f'{phase.capitalize()} Violation v.s. Fraction')
    plt.xscale("log")
    # plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()


def plot_time_vs_fraction(data_dict, alphas=[0.05, 0.5, 0.95]):
    # Print Time and plot them
    # fr = np.log10(fractions)
    fr = data_dict['fractions']

    n = len(data_dict['time_expgrad_all_ci'])

    # plt.fill_between(fr,
    #                  time_unmitigated_ci[:,1],
    #                  time_unmitigated_ci[:,2],
    #                  color='b', alpha=0.3)
    # plt.plot(fr, time_unmitigated_ci[:,0], 'bo-', label="unmitigated")

    plt.fill_between(fr,
                     data_dict['time_expgrad_all_ci'][:, 1],
                     data_dict['time_expgrad_all_ci'][:, 2],
                     color='r', alpha=0.3)
    plt.plot(fr, data_dict['time_expgrad_all_ci'][:, 0], 'ro-', label="expgrad full")

    plt.fill_between(fr,
                     data_dict['time_expgrad_fracs_ci'][:, 1],
                     data_dict['time_expgrad_fracs_ci'][:, 2],
                     color='g', alpha=0.3)
    plt.plot(fr, data_dict['time_expgrad_fracs_ci'][:, 0], 'go-', label="expgrad sample")

    # plt.fill_between(fr,
    #                  time_hybrid1_ci[:,1],
    #                  time_hybrid1_ci[:,2],
    #                  color='k', alpha=0.3)
    # plt.plot(fr, time_hybrid1_ci[:,0], 'bo-', label="hybrid 1 (GS only)")

    # plt.fill_between(fr,
    #                  time_hybrid2_ci[:,1],
    #                  time_hybrid2_ci[:,2],
    #                  color='c', alpha=0.3)
    # plt.plot(fr, time_hybrid2_ci[:,0], 'co-', label="hybrid 2 (GS + pmf_predict)")

    plt.fill_between(fr,
                     data_dict['time_hybrid3_ci'][:, 1],
                     data_dict['time_hybrid3_ci'][:, 2],
                     color='k', alpha=0.3)
    plt.plot(fr, data_dict['time_hybrid3_ci'][:, 0], 'ko-', label="hybrid 3 (GS + LP)")

    # plt.fill_between(fr,
    #                  time_hybrid4_ci[:,1],
    #                  time_hybrid4_ci[:,2],
    #                  color='m', alpha=0.3)
    # plt.plot(fr, time_hybrid4_ci[:,0], 'mo-', label="hybrid 4 (GS + LP+)")

    plt.fill_between(fr,
                     data_dict['time_hybrid5_ci'][:, 1],
                     data_dict['time_hybrid5_ci'][:, 2],
                     color='y', alpha=0.3)
    plt.plot(fr, data_dict['time_hybrid5_ci'][:, 0], 'yo-', label="hybrid 5 (LP)")

    plt.fill_between(fr,
                     data_dict['time_combined_ci'][:, 1],
                     data_dict['time_combined_ci'][:, 2],
                     color='m', alpha=0.3)
    plt.plot(fr, data_dict['time_combined_ci'][:, 0], 'mo-', label=f"hybrid combined (alpha={alphas[-1]})")

    plt.xlabel('Dataset Fraction (log scale)')
    plt.ylabel('Time (second)')
    plt.title('Fitting time v.s. Fraction')
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.show()


if __name__ == '__main__':
    base_dir = os.path.join("results", "yeeha", "adult")
    unmitigated_results_file_name = f"{base_dir}/0.05_2021-01-25_09-33-57_unmitigated.json"
    fairlearn_results_file_name = f"{base_dir}/0.05_2021-01-25_09-33-57_fairlearn.json"
    # hybrid_results_file_name = f"{base_dir}/0.05_2021-02-23_05-29-19_hybrids.json"
    hybrid_results_file_name = f"{base_dir}/0.05_2021-01-25_09-59-57_hybrid.json"
    with open(unmitigated_results_file_name, 'r') as _file:
        unmitigated_results = json.load(_file)
    with open(fairlearn_results_file_name, 'r') as _file:
        fairlearn_results = json.load(_file)
    with open(hybrid_results_file_name, 'r') as _file:
        hybrid_results = json.load(_file)

    data = load_data_adult(unmitigated_results, fairlearn_results, hybrid_results)
    plot_time_vs_fraction(data)
    plot_erro_vio_vs_fraction(data, phase='train')
    plot_erro_vio_vs_fraction(data, phase='test')