import os
from copy import deepcopy

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()  # for plot styling
import numpy as np
import pandas as pd
from utils import mean_confidence_interval, aggregate_phase_time
from graphic_synth import get_combined_hybrid

sns.set(rc={"figure.dpi": 400, 'savefig.dpi': 400})
# sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 14})
plt.tight_layout()


# plt.rcParams["figure.figsize"] = (10,5)
# ax = global_df.pivot_table(index=['id', 'match_code'], columns=['dataset_code'], values=['pearson']).droplevel(0,1).groupby(['match_code']).plot(kind='box')
# ax['match'].get_figure().savefig(os.path.join(...))
# ax['nomatch'].get_figure().savefig(os.path.join(...), bbox_inches='tight')


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
        errors[f"combined[{alpha}]"] = (
            cols[i], f"hybrid combined (alpha={alpha})", data_dict[f'{phase}_error_combined_ci'][alpha], None)
        violations[f"combined[{alpha}]"] = (
            cols[i], f"hybrid combined (alpha={alpha})", data_dict[f'{phase}_vio_combined_ci'][alpha], None)

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


def plot_time_vs_fraction_json(data_dict, alphas=[0.05, 0.5, 0.95]):
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


def plot_time_vs_fraction(all_model_df, x_axis='frac', y_axis='time', alphas=[0.05, 0.5, 0.95]):
    # Print Time and plot them
    x_values = all_model_df[x_axis].dropna().unique()
    # fr = np.log10(fr)
    time_aggregated_df = aggregate_phase_time(all_model_df)

    # model_names = all_model_df[['model_name']].drop_duplicates()

    columns = ['label', 'color']
    data = {
        'expgrad_fracs' : ['expgrad sample', 'g'],
        'hybrid_5' : ['hybrid 5 (LP)', 'y'],
        'hybrid_1' : ['hybrid 1 (GS only)', 'b'],
        'hybrid_2' : ['hybrid 2 (GS + pmf_predict)', 'c'],
        'hybrid_3' : ['hybrid 3 (GS + LP)', 'k'],
        'hybrid_4' : ['hybrid 4 (GS + LP+)', 'm'],
        'combined' : ['hybrid combined (alpha 0.95)', 'm'],
        'fairlearn_full' : ['expgrad full', 'r'],
        'unmitigated' : ['unmitigated', 'o']}
    to_plot_models = ['expgrad_fracs',
                      'hybrid_5',
                      # 'hybrid_1',
                      # 'hybrid_2',
                      'hybrid_3',
                      # 'hybrid_4',
                      'combined',
                      'fairlearn_full',
                      # 'unmitigated'
                      ]
    map_df = pd.DataFrame.from_dict(data, columns=columns, orient='index')
    for key, turn_df in time_aggregated_df.groupby('model_name'):
        if key not in to_plot_models:
            continue
        turn_data = turn_df.pivot(index='random_seed', columns=x_axis,values=y_axis)
        ci = mean_confidence_interval(turn_data)
        label, color = map_df.loc[key, ['label','color']].values
        plt.fill_between(x_values, ci[1], ci[2],
                         color=color, alpha=0.3)
        values = ci[0]
        if len(values) < len(x_values):
            values = [values] * len(x_values)
        plt.plot(x_values, values, f'{color}o-', label=label)

    plt.xlabel('Dataset Fraction (log scale)')
    plt.ylabel(y_axis)
    plt.title(f'{y_axis} v.s. {x_axis}')
    plt.xscale("log")
    if y_axis == 'time':
        plt.yscale("log")
    sns.axes_style("whitegrid")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    base_dir = os.path.join("results", "sparc20", "adult")
    files = pd.Series(os.listdir(base_dir))
    name_df = files.str.extract(r'^(\d{4}-\d{2}-\d{2})_((?:\d{2}-{0,1}){3})_(.*)\.(.*)$', expand=True)
    name_df.rename(columns={0: 'date', 1: 'time', 2: 'model', 3: 'extension'}, inplace=True)
    name_df['full_name'] = files
    name_df = name_df.query('extension == "csv"')
    last_files = name_df.sort_values(['date', 'time'], ascending=False).groupby('model').head(1)

    df_dict = {model_name: pd.read_csv(os.path.join(base_dir, turn_name))
               for turn_name, model_name in (last_files[['full_name', 'model']].values)}
    all_model_df = pd.concat(df_dict.values())

    plot_time_vs_fraction(all_model_df)
    plot_time_vs_fraction(all_model_df, y_axis='train_violation')
    plot_time_vs_fraction(all_model_df, y_axis='train_error')
    # plot_erro_vio_vs_fraction(data, phase='train')
    # plot_erro_vio_vs_fraction(data, phase='test')

# For old json files
# if __name__ == '__main__':
#     base_dir = os.path.join("results", "yeeha", "adult")
#     unmitigated_results_file_name = f"{base_dir}/0.05_2021-01-25_09-33-57_unmitigated.json"
#     fairlearn_results_file_name = f"{base_dir}/0.05_2021-01-25_09-33-57_fairlearn.json"
#     # hybrid_results_file_name = f"{base_dir}/0.05_2021-02-23_05-29-19_hybrids.json"
#     hybrid_results_file_name = f"{base_dir}/0.05_2021-01-25_09-59-57_hybrid.json"
#     with open(unmitigated_results_file_name, 'r') as _file:
#         unmitigated_results = json.load(_file)
#     with open(fairlearn_results_file_name, 'r') as _file:
#         fairlearn_results = json.load(_file)
#     with open(hybrid_results_file_name, 'r') as _file:
#         hybrid_results = json.load(_file)
#
#     data = load_data_adult(unmitigated_results, fairlearn_results, hybrid_results)
#     plot_time_vs_fraction(data)
#     plot_erro_vio_vs_fraction(data, phase='train')
#     plot_erro_vio_vs_fraction(data, phase='test')
