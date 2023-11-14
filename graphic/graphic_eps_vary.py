import itertools
import os

import pandas as pd

from graphic import utils_results_data
from graphic_utility import plot_all_df_subplots

if __name__ == '__main__':
    save = True
    show = False

    experiment_code_list = [
        # "s_h_1.0", old not aligned seeds
        # 's_h_EO_1.0',
        "s_h_1.0r",
        's_h_EO_1.0r',
        "acs_h_eps_1.0r",  # PublicC
        'acs_eps_EO_1.0r',  # PublicC

        'acsE_eps_EO_1.0r',
        'acsE_h_eps_1.0r',
        # "acs_h_eps_1.LGBM0",  # Employment + PublicC lgbm DP
        # "acs_h_eps_1.E0",  # Employment LR DP
        # 'acs_eps_EO_1.1',  # Employment LR EO
        # 'acs_eps_EO_2.1',  # Employment lgbm EO

        "s_c_1.0r",
        "s_zDI_1.1",
        "s_tr_1.0r",
        "s_tr_1.1r",
        's_tr_2.0r',
        's_tr_2.1r',
        's_zDI_1.2',
        's_zDI_1.22',
        's_f_1.0r',
        's_f_1.1r',
        's_zEO_1.1',

        'most_frequent_sig.0r',

    ]
    rlp_false_conf_list = [
        'f_eta0_eps.1',
        'f_eta0_eps.2',
    ]

    dataset_results_path = os.path.join("results")
    base_plot_dir = os.path.join('results', 'plots')
    results_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    # Check number of replication
    # all_df.loc[all_df['model_name'] == 'ZafarDI', ['train_test_fold', 'random_seed', 'train_test_seed']].apply(lambda x: '_'.join(x.astype(str)), 1).nunique()
    # a = utils_results_data.load_results_experiment_id(['acs_h_gs1_1.0'], dataset_results_path)
    # a[a['dataset_name'].str.startswith('ACS')][['random_seed','train_test_fold', 'train_test_seed']].apply(lambda x: '_'.join(x.astype(str)),axis=1).unique() # value_counts()
    # a.query('dataset_name == "ACSEmployment"')[np.intersect1d(x.columns, utils_results_data.cols_to_aggregate)].apply(lambda x: '_'.join(x.astype(str)), axis=1).unique().tolist()

    rlp_df = utils_results_data.load_results_experiment_id(rlp_false_conf_list, dataset_results_path)
    model_code = 'RLP=' + rlp_df['run_linprog_step'].map({True: 'T', False: 'F'}) + ' max_iter=' + rlp_df[
        'max_iter'].astype(str)
    rlp_df['model_code'] = model_code
    rlp_df = utils_results_data.best_gap_filter_on_eta0(rlp_df)
    rlp_df_filtered = rlp_df[rlp_df['max_iter'].isin([5, 10, 100])]

    all_df = pd.concat([results_df, rlp_df_filtered])

    restricted = ['hybrid_7_exp', 'unconstrained_exp', ]  # PlotUtility.other_models + ['hybrid_7_exp',]
    restricted = [x.replace('_exp', '_eps') for x in restricted]
    restricted_v1 = restricted + ['Calmon', 'ZafarDI', 'ThresholdOptimizer', 'Feld', 'ZafarEO'] + rlp_df_filtered[
        'model_code'].unique().tolist()
    # todo add most_frequent

    sort_map = {name: i for i, name in enumerate(restricted_v1)}
    all_df = all_df.assign(model_sort=all_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, True, True]).drop(columns=['model_sort'])
    all_df.loc[all_df['model_code'].str.contains('unconstrained'), 'eps'] = pd.NA

    grouping_col = 'eps'
    x_axis_list = ['eps']

    rlp_df_filtered_v2 = rlp_df[rlp_df['max_iter'].isin([50])]
    restricted_v2 = restricted + ['Calmon', 'ZafarDI', 'ThresholdOptimizer', 'Feld', 'ZafarEO'] + rlp_df_filtered_v2[
        'model_code'].unique().tolist()
    plot_all_df_subplots(all_df, model_list=restricted_v2, chart_name='eps_v3', grouping_col='eps',
                         save=save, show=show, sharex=False, sharey=False,
                         axis_to_plot=[['test_violation', 'test_error'],
                                       ['test_violation', 'time'],
                                       ])

    y_axis_list_short = ['time'] + ['_'.join(x) for x in itertools.product(['test'], ['error', 'violation'])]
    y_axis_list_long = y_axis_list_short + ['train_error', 'train_violation']
    for y_axis_list, suffix in [(y_axis_list_short, '_v2'), (y_axis_list_long, '')]:
        plot_all_df_subplots(all_df, model_list=restricted_v1, chart_name='eps' + suffix, grouping_col='eps',
                             save=save, show=show,
                             axis_to_plot=list(itertools.product(x_axis_list, y_axis_list)))
