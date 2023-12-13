import itertools
import os

import pandas as pd
from matplotlib import pyplot as plt

from graphic import utils_results_data
from graphic.style_utility import StyleUtility
from graphic.utils_results_data import prepare_for_plot
from graphic_utility import plot_all_df_subplots, PlotUtility
import graphic_utility

if __name__ == '__main__':
    save = True
    show = False

    experiment_code_list = [
        'acsER_bin2.0r',
        'acsER_bin2.1r',
        'acsER_bin3.0r',
        'acsER_bin4.0r',
    ]

    dataset_results_path = os.path.join("results")
    base_plot_dir = os.path.join('results', 'plots')
    all_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    all_df.columns = [x.replace('violation', 'DemographicParity') for x in all_df.columns]
    multivalued_sensitive_df = utils_results_data.load_results_experiment_id([
        'acsER_bin2.2r',
        'acsER_bin3.1r',
    ], dataset_results_path)
    multivalued_sensitive_df.columns = [x.replace('violation', 'DemographicParity') for x in
                                        multivalued_sensitive_df.columns]
    multivalued_sensitive_df = multivalued_sensitive_df.rename(
        columns={f'{pre}_violation': f'{pre}_DemographicParity' for pre in ['test', 'train']}).rename(
        columns={f'{pre}_{metric}': f'{pre}_{metric} Multi' for pre in ['test', 'train'] for metric in
                 ['error', 'DemographicParity', 'EqualizedOdds']}
    )
    new_cols = [f'{pre}_{metric} Multi' for pre in ['test', 'train'] for metric in
                ['error', 'EqualizedOdds', 'DemographicParity']]
    to_synch = utils_results_data.cols_to_index + utils_results_data.seed_columns + ['phase']
    all_df = all_df.merge(
        multivalued_sensitive_df[to_synch + new_cols],
        on=to_synch, how='outer')

    restricted = ['hybrid_7', 'unconstrained', ]
    # restricted = [x.replace('_exp', '_eps') for x in restricted]
    restricted += ['ThresholdOptimizer', 'Feld', 'ZafarDI', ]

    sort_map = {name: i for i, name in enumerate(restricted)}

    all_df = all_df.assign(model_sort=all_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, True, True]).drop(columns=['model_sort'])
    all_df.loc[all_df['model_code'].str.contains('unconstrained'), 'eps'] = pd.NA

    grouping_col = 'eps'
    x_axis_list = ['eps']
    y_axis_list_long = ['_'.join(x) for x in
                        itertools.product([#'test',
                                           'train'], [# 'error',
                                                      'DemographicParity', 'DemographicParity_orig',
                                                              'DemographicParity Multi'])]
    y_axis_list_short = [x for x in y_axis_list_long if 'test' in x]

    tranformed_df_list = []
    # create a new set of models starting from models code but appending '_orig' and using the columns suffixed with '_orig'
    for conf in ['_orig', ' Multi']:
        all_df_tranformed = all_df.copy()
        turn_cols = [x for x in all_df.columns if conf in x]
        all_df_tranformed['model_code'] = all_df_tranformed['model_code'] + conf
        for col in turn_cols:
            all_df_tranformed[col.replace(conf, '')] = all_df_tranformed[col]
        # all_df_tranformed['model_code'] = all_df_tranformed['model_code'].str.replace('hybrid_7_orig','EXPGRAD=adaptive GS=No LP=Yes ORIG')
        all_df_tranformed['model_code'] = all_df_tranformed['model_code'].map(
            {'hybrid_7' + conf: 'EXPGRAD=adaptive GS=No LP=Yes' + conf,
             'unconstrained' + conf: 'UNMITIGATED full' + conf})
        tranformed_df_list.append(all_df_tranformed)

    orig_model_list = [x + '_orig' for x in restricted] + ['EXPGRAD=adaptive GS=No LP=Yes_orig',
                                                           'UNMITIGATED full_orig']
    model_list = (restricted + orig_model_list + [x.replace('_orig', ' Multi') for x in orig_model_list])
    plot_all_df_subplots(pd.concat(tranformed_df_list + [all_df]),
                         model_list=model_list, chart_name='bin_v3',
                         grouping_col='eps',
                         save=save, show=show, sharex=False, sharey=False,
                         axis_to_plot=[['train_violation', 'train_error'], ],
                         )

    model_list = list(restricted)
    mean_error_df = prepare_for_plot(all_df[all_df['model_code'].isin(model_list)], grouping_col)
    mean_error_df['model_code'] = mean_error_df['model_code'].map(StyleUtility.get_label)
    mean_error_df['model_code'] = mean_error_df['model_code'].str.replace('EXPGRAD=adaptive GS=No LP=Yes',
                                                                          'EXPGRAD=adaptive')

    pl_util = PlotUtility(save=save, show=show, suffix='', annotate_mode='all')
    for (y_axis_list, suffix), (t_constraint, cc) in itertools.product(
            [(y_axis_list_long, ''), (y_axis_list_short, '_v2')],
            [('DemographicParity', 'dp'), ('EqualizedOdds', 'eo')]):
        # plot_all_df_subplots(all_df, model_list=restricted, chart_name='eps' + suffix, grouping_col='eps',
        #                      save=save, show=show,
        #                      axis_to_plot=list(itertools.product(x_axis_list, y_axis_list)),
        #                      custom_add_graphic_object='bar')
        y_axis_list = [x.replace('DemographicParity', t_constraint) for x in y_axis_list]

        pl_util.apply_plot_function_and_save(df=mean_error_df.query(f'constraint_code == "{cc}"'),
                                             additional_dir_path='all_df',
                                             plot_function=graphic_utility.bar_plot_function_by_model,
                                             name=f'binary_{cc}' + suffix,
                                             y_axis_list=y_axis_list)

        # plt.figure(figsize=(10, 6))
        # axes = all_df.boxplot(column=y_axis_list, by=['model_code'], rot=90, figsize=(10, 10))
        pass
