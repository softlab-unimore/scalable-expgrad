import ast
import itertools
import os
from copy import deepcopy

import numpy as np
import pandas as pd

from graphic_utility import plot_routine_performance_violation, PlotUtility, restricted_list
from utils_results_data import load_results, filter_results, add_sigmod_metric, prepare_data, load_results_experiment_id

if __name__ == '__main__':
    save = True
    show = False

    experiment_code_list = [
        "s_h_1.0",
        "s_c_1.0",
        "s_zDI_1.0",
        "s_tr_1.0",
    ]

    dataset_results_path = os.path.join("..", "results", "fairlearn-2")
    base_plot_dir = os.path.join("..", 'results', 'plots')
    all_df = load_results_experiment_id(experiment_code_list, dataset_results_path)

    df = all_df.copy()

    grouped = all_df.groupby(['base_model_code', 'dataset_name', ])
    for key, turn_df in grouped:
        suffix = 'ALL MODELS_' + key[0]
        dataset_name = key[1]
        # plot_routine_performance_violation(turn_df, dataset_name=key[0], save=save, show=show,
        #                                    suffix='ALL MODELS' + key[1])

        pl_util = PlotUtility(show=show, save=save, path_suffix=suffix)
        sigmod_list = pl_util.other_models + ['hybrid_7_exp',
                                              'expgrad_fracs_exp',
                                              'sub_hybrid_6_exp_gf_1',
                                              'unconstrained_exp',
                                              #'unconstrained_exp',
                                              ]
        model_set_list = [(sigmod_list, 'all'),
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
                                name=f'{phase}_{metric_name}_vs_{x_axis}_{model_set_name}')