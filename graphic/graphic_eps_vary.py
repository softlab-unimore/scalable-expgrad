import ast
import itertools
import os
from copy import deepcopy

import utils_results_data
from graphic_utility import plot_routine_performance_violation, PlotUtility, restricted_list, plot_all_df_subplots, \
    plot_by_df
from utils_results_data import prepare_for_plot

if __name__ == '__main__':
    save = True
    show = False

    experiment_code_list = [
        "s_h_1.0",
        's_h_EO_1.0',
        "acs_h_eps_1.0",
        "acs_h_eps_1.E0",
        "acs_h_eps_1.LGBM0",
        'acs_eps_EO_1.0',
        'acs_eps_EO_2.0',
        'acs_eps_EO_2.1',

        "s_c_1.0",
        "s_zDI_1.1",
        "s_tr_1.0",
    ]

    dataset_results_path = os.path.join("../run_experiments", "results", "fairlearn-2")
    base_plot_dir = os.path.join("../run_experiments", 'results', 'plots')
    all_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    # check results
    # a = utils_results_data.load_results_experiment_id(['acs_h_gs1_1.0'], dataset_results_path)
    # a[a['dataset_name'].str.startswith('ACS')][['random_seed','train_test_fold', 'train_test_seed']].apply(lambda x: '_'.join(x.astype(str).values),axis=1).unique()
    # a.query('dataset_name == "ACSEmployment"')[np.intersect1d(x.columns, utils_results_data.cols_to_aggregate)].apply(lambda x: '_'.join(x.astype(str)), axis=1).unique().tolist()
    model_list = PlotUtility.other_models + ['hybrid_7_exp',
                                             'expgrad_fracs_exp',
                                             'sub_hybrid_6_exp_gf_1',
                                             'unconstrained_frac_exp',
                                             ]
    model_list = [x.replace('_exp', '_eps') for x in model_list]

    restricted = ['sub_hybrid_6_exp_gf_1']  # PlotUtility.other_models + ['hybrid_7_exp',]
    restricted = [x.replace('_exp', '_eps') for x in restricted]
    restricted += ['Calmon', 'ZafarDI', 'ThresholdOptimizer']

    plot_all_df_subplots(all_df, model_list=restricted, model_set_name='', grouping_col='eps', save=save, show=show)

    filtered_df = all_df[all_df['model_code'].isin(model_list)]
    mean_error_df = prepare_for_plot(filtered_df, 'eps')
    mean_error_df_filtered = mean_error_df  # [mean_error_df['eps'].fillna(0.005) == 0.005]
    mean_error_df_filtered = mean_error_df_filtered.sort_values(
        ['dataset_name', 'base_model_code', 'model_code', 'eps'])
    host_name, current_time_str = utils_results_data.get_info()
    result_path_name = 'all_df'
    dir_path = os.path.join(base_plot_dir, result_path_name, host_name)
    mean_error_df_filtered.to_csv(os.path.join(dir_path, 'all_model_all_metrics_mean_error.csv'))

    pl_util = PlotUtility(save=save, show=show)
    plot_by_df(pl_util, all_df, to_plot_models=model_list, model_set_name='baselines', grouping_col='eps')

