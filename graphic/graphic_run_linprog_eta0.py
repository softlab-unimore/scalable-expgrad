import itertools
import os

import pandas as pd

from graphic import utils_results_data
from graphic_utility import PlotUtility, plot_all_df_subplots
from graphic.utils_results_data import best_gap_filter_on_eta0

if __name__ == '__main__':
    save = True
    show = False

    experiment_code_list = [
        'f_eta0_1.0',
        'f_eta0_2.0',
        'f_eta0_1.1',
        'f_eta0_2.1',
    ]

    dataset_results_path = os.path.join("results")
    all_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    str_map = {True: 'T', False: 'F'}
    model_code = 'RLP=' + all_df['run_linprog_step'].map(str_map)  # + ' max_iter==' + all_df['max_iter'].astype(str)
    all_df['model_code'] = model_code

    non_selected_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    str_map = {True: 'T', False: 'F'}
    model_code = 'RLP=' + non_selected_df['run_linprog_step'].map(str_map) + ' eta0==' + non_selected_df['eta0'].astype(
        str)
    non_selected_df['model_code'] = model_code

    hybrids = utils_results_data.load_results_experiment_id(['s_h_EO_1.0r', 's_h_1.0r'], dataset_results_path)
    normal_expgrad = hybrids[
        hybrids['model_code'].isin(['expgrad_fracs_eps']) &
        (hybrids['exp_frac'] == 1)]
    assert all_df['eps'].nunique() == 1

    # filter extra configs

    for col in utils_results_data.cols_to_synch:
        normal_expgrad = normal_expgrad[normal_expgrad[col].isin(all_df[col].unique())]

    normal_expgrad['max_iter'] = 50
    all_df = best_gap_filter_on_eta0(all_df)
    normal_expgrad['model_code'] = 'EXPGRAD'
    all_df = pd.concat([all_df, normal_expgrad  # , non_selected_df
                        ])

    y_axis_list_long = ['_'.join(x) for x in itertools.product(['train', 'test'], ['error', 'violation'])] + [
        'n_oracle_calls_']
    y_axis_list_short = [x for x in y_axis_list_long if 'test' not in x ]
    for y_axis_list, suffix in [(y_axis_list_short, '_v2'), (y_axis_list_long, '')]:
        plot_all_df_subplots(all_df, model_list=all_df['model_code'].unique(), chart_name='eta0' + suffix,
                             grouping_col='max_iter',
                             save=save, show=show, sharex=False, annotate_col='n_oracle_calls_',
                             sharey='row', axis_to_plot=list(itertools.product(['time'], y_axis_list)),
                             add_threshold=True)
    pl_util = PlotUtility(save=save, show=show)

    # plot_by_df(pl_util, all_df, all_df['model_code'].unique(), model_set_name='eta0',
    #            grouping_col='max_iter')
