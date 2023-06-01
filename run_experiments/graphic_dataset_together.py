import ast
import itertools
import os
from copy import deepcopy


import utils_results_data
from graphic_utility import plot_routine_performance_violation, PlotUtility, restricted_list, plot_all_df

if __name__ == '__main__':
    save = True
    show = False


    experiment_code_list = [
        "acs_h_eps_1.0",
        "acs_h_eps_1.E0",
        "acs_h_eps_1.LGBM0",
        "s_h_1.0",
        "s_c_1.0",
        "s_zDI_1.1",
        "s_tr_1.0",
        'acs_eps_EO_1.0',
        's_h_EO_1.0'
    ]

    dataset_results_path = os.path.join(".", "results", "fairlearn-2")
    base_plot_dir = os.path.join(".", 'results', 'plots')
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

    plot_all_df(all_df, model_list=restricted, save_dir=PlotUtility.base_plot_dir, grouping_col='eps',save=save,show=show)

    filtered_df = all_df[all_df['model_code'].isin(model_list)]
    time_aggregated_df = utils_results_data.aggregate_phase_time(filtered_df)
    mean_error_df = time_aggregated_df.groupby(utils_results_data.cols_to_index, as_index=False, dropna=False)[
        utils_results_data.numerical_cols].agg(['mean', ('error', PlotUtility.get_error)]).reset_index()
    mean_error_df_filtered = mean_error_df#[mean_error_df['eps'].fillna(0.005) == 0.005]
    mean_error_df_filtered = mean_error_df_filtered.sort_values(['dataset_name', 'base_model_code', 'model_code', 'eps'])
    host_name, current_time_str = utils_results_data.get_info()
    result_path_name = 'all_df'
    dir_path = os.path.join(base_plot_dir, result_path_name, host_name)
    mean_error_df_filtered.to_csv(os.path.join(dir_path, 'all_model_all_metrics_mean_error.csv'))

    grouped = all_df.groupby(['base_model_code', 'dataset_name'])
    for key, turn_df in grouped:
        suffix = 'ALL MODELS_' + key[0]
        dataset_name = key[1]
        # plot_routine_performance_violation(turn_df, dataset_name=key[0], save=save, show=show,
        #                                    suffix='ALL MODELS' + key[1])

        pl_util = PlotUtility(show=show, save=save, path_suffix=suffix)
        original_list = [x.replace('exp', 'eps') for x in pl_util.to_plot_models]
        restricted_list = [x.replace('exp', 'eps') for x in restricted_list]
        restricted = pl_util.other_models + ['hybrid_7_exp',
                                             'expgrad_fracs_exp',
                                             'sub_hybrid_6_exp_gf_1',
                                             'unconstrained_frac_exp',
                                             # 'unconstrained_exp',
                                             ]
        model_set_list = [(restricted, 'all'),
                          # (restricted_list, 'restricted'),
                          # (['unconstrained_frac_eps'], 'unconstrained'),
                          ]
        to_iter = list(itertools.product(['train', 'test'], ['error', 'violation', 'di', ],
                                         [('di', turn_df), ('time', turn_df), ],
                                         model_set_list
                                         ))
        for phase, metric_name, (x_axis, turn_df), (to_plot_models, model_set_name) in to_iter:
            to_plot_models = [x.replace('_exp', '_eps') for x in to_plot_models]
            if model_set_name != '' and x_axis == 'frac':
                continue
            if x_axis not in turn_df.columns:
                x_axis = f'{phase}_{x_axis}'
                if x_axis not in turn_df.columns:
                    raise ValueError(f'{x_axis} is not a valid x_axis')
            y_axis = f'{phase}_{metric_name}'
            if y_axis == x_axis:
                continue
            pl_util.to_plot_models = to_plot_models
            pl_util.plot(turn_df, y_axis=y_axis, x_axis=x_axis, groupby_col='eps', dataset_name=dataset_name)
            pl_util.save_figure(base_plot_dir, dataset_name=dataset_name,
                                name=f'{x_axis}_vs_{y_axis}_{model_set_name}')
