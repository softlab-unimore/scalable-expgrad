import itertools
import os

import pandas as pd

import utils_results_data
from graphic.graphic_utility import select_oracle_call_time, PlotUtility, plot_by_df, plot_all_df_subplots, \
    extract_oracle_time

if __name__ == '__main__':
    save = True
    show = False

    experiment_code_list = set(list([
        'sigmod_h_exp_1.0',
        'acs_h_gs1_1.0',
        'acsE_h_gs1_1.0',
        'acs_h_gs1_1.1',  # still missing

        's_h_exp_EO_1.0',
        'acs_h_gs1_EO_1.0',
        'acs_h_gs1_EO_2.0',

        'sigmod_h_exp_2.0',
        'sigmod_h_exp_3.0',
        'acs_h_gsSR_1.0',
        'acs_h_gsSR_1.1',
        'acsE_h_gsSR_1.1',
        'acs_h_gsSR_2.0',
        'acs_h_gsSR_2.1',
    ]))

    grid_chart_models = [
        # 'expgrad_fracs_exp',
        # 'hybrid_3_exp_gf_1',
        # 'hybrid_5_exp',
        # 'hybrid_3_exp',
        # 'hybrid_3_exp_gf_1',
        'hybrid_7_exp',
        'sub_hybrid_3_exp',  # sqrt
        # 'sub_hybrid_5_exp',
        'sub_hybrid_3_exp_gf_1',
    ]
    sqrt_conf = [x for x in experiment_code_list if 'SR' in x] + ['sigmod_h_exp_2.0', 'sigmod_h_exp_3.0', ]
    gf_1_conf = set(experiment_code_list) - set(sqrt_conf)

    sort_map = {name: i for i, name in enumerate(grid_chart_models)}


    dataset_results_path = os.path.join("results", "fairlearn-2")

    all_df = utils_results_data.load_results_experiment_id(gf_1_conf, dataset_results_path)
    exp_frac_models = ['hybrid_5_exp',
                       'hybrid_7_exp',]
    all_df = all_df.query('train_test_fold == 0 and random_seed == 0 and train_test_seed == 0')
    all_df = all_df.assign(model_sort=all_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, True, True]).drop(columns=['model_sort'])
    exp_mask = all_df['phase'] == 'expgrad_fracs'
    extract_oracle_time(all_df, new_col_name='time_oracles', cols_to_select='all')

    plot_all_df_subplots(all_df, model_list=exp_frac_models,
                         model_set_name='exp_frac', grouping_col='exp_frac',
                         save=save, show=show, sharex=False, increasing_marker_size=False, ylim_list=[None, None, (0, 0.1)],
                         sharey='row', xlog=True,
                         axis_to_plot=list(itertools.product(['exp_frac'], ['time', 'test_error', 'test_violation',
                                                                            'train_error', 'train_violation',
                                                                            'n_oracle_calls_', 'time_oracles'
                                                                            ])))

    sqrt_df = utils_results_data.load_results_experiment_id(sqrt_conf, dataset_results_path)
    gf_1_df = utils_results_data.load_results_experiment_id(gf_1_conf, dataset_results_path)

    sqrt_df = sqrt_df[sqrt_df['model_code'].isin(['hybrid_7_exp', 'sub_hybrid_3_exp'])]
    gf_1_df = gf_1_df[gf_1_df['model_code'].isin(['sub_hybrid_3_exp_gf_1'])]
    all_df = pd.concat([sqrt_df, gf_1_df])
    all_df = all_df.query('train_test_fold == 0 and random_seed == 0 and train_test_seed == 0')

    # all_df = all_df[all_df['model_code'].isin(grid_chart_models)]

    all_df = select_oracle_call_time(all_df)
    all_df = all_df.assign(model_sort=all_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, True, True]).drop(columns=['model_sort'])

    pl_util = PlotUtility(save=save, show=show)
    # Align random seed values --> check sqrt time is at least equal to adaptive or greater.

    x_axis_list = ['time']
    y_axis_list = ['_'.join(x) for x in itertools.product(['test'], ['error', 'violation'])] + ['train_violation']
    # y_axis_list = ['test_error','train_violation']
    plot_all_df_subplots(all_df, model_list=grid_chart_models, model_set_name='oracle_calls', grouping_col='exp_frac',
                         save=save, show=show, sharex=False, increasing_marker_size=True, ylim_list=[None, (0, 0.1), (0, 0.1)],
                         sharey='row', axis_to_plot=list(itertools.product(x_axis_list, y_axis_list)))
    #
    # plot_by_df(pl_util, all_df, grid_chart_models, model_set_name='oracle_calls',
    #            grouping_col='exp_frac')
