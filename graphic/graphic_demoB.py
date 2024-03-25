import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from graphic import utils_results_data, graphic_utility
from graphic.style_utility import StyleUtility
from graphic.utils_results_data import prepare_for_plot
from graphic_utility import plot_all_df_subplots, PlotUtility, plot_demo_subplots

if __name__ == '__main__':
    save = True
    show = False

    experiment_code_list = [
        'demo.D.0r',
        #'demo.C.1r',
        ]

    dataset_results_path = os.path.join("results")
    base_plot_dir = os.path.join('results', 'plots')
    results_df = utils_results_data.load_results_experiment_id(experiment_code_list, dataset_results_path)
    results_df['model_code'] = results_df['model_code'].replace('hybrid_7', 'EXPGRAD')
    results_df['model_code'] = results_df['model_code'].replace('fairlearn', 'EXPGRAD')
    model_list = [
        # 'unconstrained',
        'EXPGRAD',
        #'ThresholdOptimizer', 'Calmon', 'Feld', 'ZafarDI', 'ZafarEO',
        ]
    sort_map = {name: i for i, name in enumerate(model_list)}
    all_df = results_df.assign(model_sort=results_df['model_code'].map(sort_map)).sort_values(
        ['dataset_name', 'base_model_code', 'constraint_code', 'model_sort'],
        ascending=[True, False, False, True]).drop(columns=['model_sort'])

    x_axis = 'train_fractions'
    plot_demo_subplots(all_df, model_list=model_list, chart_name='D', save=save, show=show,
                       axis_to_plot=[[x_axis, y_axis ] for y_axis in ['test_error', 'test_DemographicParity','time']],
                       sharex=False, result_path_name='demo',
                        single_plot=True, grouping_col=x_axis)
