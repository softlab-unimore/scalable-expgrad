import ast
import functools
import itertools
from copy import deepcopy

import numpy as np
import os, re
import matplotlib.pyplot as plt
import seaborn as sns;
import pandas as pd
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

import utils_results_data
from utils_results_data import get_info, get_confidence_error, mean_confidence_interval, \
    aggregate_phase_time, load_results, filter_results, cols_to_aggregate
import matplotlib as mpl

sns.set()  # for plot styling
# sns.set(rc={'figure.figsize':(8,6)})
# sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 16, "figure.dpi": 400, 'savefig.dpi': 600,
                     # 'figure.figsize': (16 * 2 / 3, 9 * 2 / 3)
                     })
plt.rcParams['figure.constrained_layout.use'] = True
sns.set_context(rc={"legend.fontsize": 7})

restricted_list = [
    'expgrad_fracs_exp',
    'expgrad_fracs_LP_off_exp',
    'hybrid_5_exp',
    # 'hybrid_5_U_exp',
    # # 'hybrid_1_exp',
    # # 'hybrid_2_exp',
    # 'hybrid_3_exp',
    # 'hybrid_3_U_exp',
    # # 'hybrid_4_exp',
    # 'hybrid_6_U_exp',
    'hybrid_6_exp',

    'hybrid_7_exp',  # ExpGradSub
    'hybrid_7_LP_off_exp',
    # 'sub_hybrid_3_exp',
    # 'sub_hybrid_4_exp',
    # 'sub_hybrid_5_U_exp',
    # 'sub_hybrid_5_exp',
    # 'sub_hybrid_6_U_exp',
    'sub_hybrid_6_exp',
    'unconstrained_frac_exp',
]


def generate_map_df():
    values_dict = {}
    model_names = ['hybrid_1', 'hybrid_2', 'hybrid_3', 'hybrid_4', 'hybrid_5', 'hybrid_6']
    unconstrained = [True, False]
    active_sampling = [True, False]
    run_linprog = [True, False]
    grid_mode = ['sqrt', 'gf_1']
    to_iter = itertools.product(model_names, unconstrained, active_sampling, grid_mode)

    for t_varing in ['exp', 'gri', 'eps']:
        for t_run_lp in run_linprog:
            for t_model_name, t_unconstrained, t_active_sampling, t_grid_mode in deepcopy(to_iter):
                name = 'sub_' if t_active_sampling else ''
                name += t_model_name + ('_U' if t_unconstrained else '')
                name += '_LP_off' if t_run_lp else ''
                name += f'_{t_varing}'
                if t_grid_mode == 'gf_1':
                    name += f'_gf_1'

                label = f'EXPGRAD=' + ('adaptive' if t_active_sampling else 'static')
                label += ' GS=' + (
                    t_grid_mode if any('_' + x in t_model_name for x in ['1', '2', '3', '4', '6']) else 'No  ')
                label += ' LP=Yes'
                label += ' +U' if t_unconstrained else ''
                if 'hybrid_6' in t_model_name:
                    label += ' *e&g'
                label += ' run_linprog=F' if t_run_lp else ''
                values_dict[name] = label

            rlp_name = "_LP_off" if t_run_lp else ""
            rlp_label = ' run_linprog=F' if t_run_lp else ''
            values_dict[f'hybrid_7{rlp_name}_{t_varing}'] = 'EXPGRAD=adaptive GS=No LP=Yes' + rlp_label
            values_dict[f'expgrad_fracs{rlp_name}_{t_varing}'] = 'EXPGRAD=static GS=No LP=No' + rlp_label
        values_dict[f'unconstrained_{t_varing}'] = 'UNMITIGATED full'
        values_dict[f'unconstrained_frac_{t_varing}'] = 'UNMITIGATED=static'

    return pd.DataFrame.from_dict(values_dict, orient='index', columns=['label'])


class PlotUtility():
    base_plot_dir = os.path.join('results', 'plots')
    map_df = generate_map_df()
    other_models = ['ThresholdOptimizer', 'Calmon', 'ZafarDI']
    map_df = pd.concat([map_df,
                        pd.DataFrame.from_dict({x: x for x in other_models},
                                               orient='index', columns=['label'])
                        ])

    def get_label(self, model_code):
        return self.map_df.loc[model_code, 'label']

    to_plot_models = [
                         # 'expgrad_fracs_gri',
                         # 'hybrid_5_gri',
                         # 'hybrid_5_U_gri',
                         # 'hybrid_1_gri',
                         # 'hybrid_2_gri',
                         # 'hybrid_3_gri',
                         # 'hybrid_3_U_gri',
                         # 'hybrid_4_gri',
                         # 'hybrid_6_gri',
                         # 'hybrid_6_U_gri',
                         'expgrad_fracs_exp',
                         'expgrad_fracs_LP_off_exp',
                         'hybrid_5_exp',
                         # 'hybrid_5_U_exp',
                         # 'hybrid_1_exp',
                         # 'hybrid_2_exp',
                         'hybrid_3_exp',
                         # 'hybrid_3_U_exp',
                         # 'hybrid_4_exp',
                         # 'hybrid_6_U_exp',
                         'hybrid_6_exp',
                         'hybrid_6_exp_gf_1',
                         'hybrid_7_exp',
                         'hybrid_7_LP_off_exp',
                         'unconstrained_exp',
                         # 'fairlearn_full',
                         # 'unmitigated',

                         ## exp subsample models
                         # 'sub_hybrid_1_exp',
                         # 'sub_hybrid_2_exp',
                         # 'sub_hybrid_3_U_exp',
                         'sub_hybrid_3_exp',
                         # 'sub_hybrid_4_exp',
                         # 'sub_hybrid_5_U_exp',
                         # 'sub_hybrid_5_exp',
                         # 'sub_hybrid_6_U_exp',
                         'sub_hybrid_6_exp',
                         'unconstrained_frac_exp',

                         'sub_hybrid_6_exp_gf_1',

                         ## eps models
                         # 'expgrad_fracs_eps',
                         # 'hybrid_1_eps',
                         # 'hybrid_2_eps',
                         # 'hybrid_3_U_eps',
                         # 'hybrid_3_eps',
                         # 'hybrid_4_eps',
                         # 'hybrid_5_U_eps',
                         # 'hybrid_5_eps',
                         # 'hybrid_6_U_eps',
                         # 'hybrid_6_eps',
                         # 'fairlearn_full_eps',

                     ] + other_models

    color_list = mpl.colormaps['tab20'].colors
    suffix = ''

    # sns.color_palette("hls", len(self.to_plot_models))
    # color_list = list(mcolors.TABLEAU_COLORS.keys())
    def __init__(self, show=True, path_suffix='', save=True):
        self.markersize = 8
        self.linewidth = 0.5
        self.show = show
        self.suffix = path_suffix
        self.save_flag = save
        # plt.rcParams['lines.markersize'] = self.markersize
        # plt.rcParams['lines.linewidth'] = self.linewidth

    def _start_plot(self):
        plt.close('all')
        self.fig = plt.figure()
        self.ax = plt.subplot()

    def _end_plot(self, x_axis, y_axis, title):
        self.ax.set_ylabel(y_axis)
        self.ax.set_title(f'{title} - {x_axis} v.s. {y_axis}')
        if x_axis == 'time':
            self.ax.set_xscale("log")
            self.ax.set_xlabel(f'{x_axis} (log scale)')
        else:
            self.ax.set_xlabel(f'{x_axis}')
            # ax.set_xscale("log")
            # ax.set_xlabel(f'{x_axis} (log scale)')
        if y_axis == 'time':
            self.ax.set_yscale("log")
            ylabel = self.ax.get_ylabel()
            self.ax.set_ylabel(f'{ylabel} (log)')
        self.ax.legend()
        self.ax = self.ax
        if self.show:
            self.fig.show()

    def plot(self, all_model_df, dataset_name, x_axis='frac', y_axis='time', groupby_col='frac'):
        self._start_plot()
        self.cols_to_aggregate = np.intersect1d(cols_to_aggregate, all_model_df.columns).tolist()
        self.groupby_col = groupby_col

        all_model_df = all_model_df[all_model_df['model_code'].isin(self.to_plot_models)]
        time_aggregated_df = aggregate_phase_time(all_model_df)
        time_aggregated_df[self.groupby_col].fillna(1, inplace=True)
        self.x_values = time_aggregated_df[self.groupby_col].unique()
        self.n_points = len(self.x_values)
        to_iter = time_aggregated_df[time_aggregated_df['model_code'].isin(self.to_plot_models)].groupby(['model_code'],
                                                                                                         dropna=False)
        self.n_models = len(self.to_plot_models)
        for model_code, turn_df in to_iter:
            self.curr_index = self.to_plot_models.index(model_code)
            errorbar_params = self.get_line_params(self.curr_index)
            errorbar_params.update(self.get_label_params(self.curr_index, model_code))
            if x_axis == 'frac':
                x_offset = (((self.curr_index / self.n_models) - 0.5) * 20 / 100) + 1
            else:
                x_offset = 1
            self.add_plot(self.ax, turn_df, x_axis, y_axis, errorbar_params=errorbar_params, x_offset_relative=x_offset)

        self._end_plot(x_axis, y_axis, title=f'{dataset_name} {x_axis} vs {y_axis}')

    def add_plot(self, ax, turn_df, x_axis, y_axis, errorbar_params, x_offset_relative=1, ):
        agg_x_axis = self.groupby_col  # if x_axis == 'time' else x_axis
        turn_data = turn_df.pivot(index=self.cols_to_aggregate, columns=agg_x_axis, values=y_axis)
        ci = mean_confidence_interval(turn_data)
        yerr = (ci[2] - ci[1]) / 2
        y_values = ci[0]
        zorder = 10 if len(y_values) == 1 else None
        if x_axis != 'frac':
            time_data = turn_df.pivot(index=self.cols_to_aggregate, columns=agg_x_axis, values=x_axis)
            ci_x = mean_confidence_interval(time_data)
            xerr = (ci_x[2] - ci_x[1]) / 2
            x_values = ci_x[0]
        else:
            xerr = None
            x_values = turn_data.columns
        # if label not in ['UNMITIGATED full']:

        ax.errorbar(x_values * x_offset_relative, y_values, xerr=xerr, yerr=yerr, zorder=zorder, **errorbar_params)
        label = errorbar_params['label']
        color = errorbar_params['color']
        if label in ['UNMITIGATED full', 'EXPGRAD=static GS=off LP=off', 'UNMITIGATED=static'] + ['ThresholdOptimizer']:
            if label in ['UNMITIGATED full']:
                y_values = [y_values.mean()]
            elif label in ['UNMITIGATED=static']:
                y_values = [y_values[-1]]
            ax.axhline(y_values[-1], linestyle="-.", color=color, zorder=10,
                       linewidth=self.linewidth)  # y_values[-1] > min(y_values) and 'error' in y_axis and 'un' not in label

        # ax.fill_between(x_values, ci[1], ci[2], color=color, alpha=0.3)
        # if len(y_values) == 1:
        #     ax.plot(self.x_values, np.repeat(y_values, self.n_points), "-.", color=color, zorder=10, label=label)
        # else:
        #     ax.plot(x_values, y_values, color=color, label=label, marker="x", linestyle='--', markersize=self.markersize)

    def get_color(self, index):
        return self.color_list[index % len(self.color_list)]

    def get_marker(self, index, total=None):
        if total is None:
            total = len(self.to_plot_models)
        rot = Affine2D().rotate_deg(index / total * 120)  # rotation for markers
        return MarkerStyle('1', 'left', rot)

    def get_line_params(self, index):
        return dict(color=self.get_color(index),
                    fmt='--', linewidth=self.linewidth, elinewidth=self.linewidth / 2)

    def get_marker_params(self, index, total, grouping_values=None):
        marker_size = self.markersize ** 2
        if grouping_values is not None:
            n = len(grouping_values)
            marker_size = (self.markersize ** 2) * (0.3 + 6 * np.arange(n) / n)
            # dimension between start-stop original marker size
        return dict(color=self.get_color(index), marker=self.get_marker(index, total), s=marker_size)

    def get_label_params(self, index, model_code, total=None):
        tmp_dict = self.get_line_params(index)
        tmp_dict.update(label=self.get_label(model_code=model_code),
                        marker=self.get_marker(index, total), fmt='--', markersize=self.markersize)
        return tmp_dict

    def save_figure(self, base_dir, dataset_name, name, fig=None):
        if self.save_flag:
            if fig is None:
                fig = self.fig
            self.save_figure_static(base_dir, dataset_name, name, fig, suffix=self.suffix)

    @staticmethod
    def save_figure_static(base_dir, dataset_name, name, fig, suffix=''):
        host_name, current_time_str = get_info()
        dir_path = os.path.join(base_dir, dataset_name, host_name + suffix)
        for t_dir in [dir_path]:
            for t_name in [
                # f'{current_time_str}_{name}',
                f'{name}']:
                t_full_path = os.path.join(t_dir, t_name)
                os.makedirs(t_dir, exist_ok=True)
                fig.savefig(t_full_path + '.png')
                t_full_path_svg = os.path.join(t_dir + '_svg', t_name)
                os.makedirs(t_dir + '_svg', exist_ok=True)
                fig.savefig(t_full_path_svg + '.svg', format='svg')

    def apply_plot_function_and_save(self, df, plot_name, plot_function, dataset_name):
        plt.close('all')
        fig, ax = plt.subplots()
        plot_function(df, ax=ax, fig=fig)
        self.save_figure(self.base_plot_dir, dataset_name=dataset_name, name=plot_name, fig=fig)
        if self.show:
            plt.show()

    @staticmethod
    def get_error(df):
        ci = mean_confidence_interval(df)
        err = (ci[2] - ci[1]) / 2
        return pd.Series({'error': err})


def time_stacked_by_phase(df, ax, fig: plt.figure):
    fig.set_figheight(8)
    fig.set_figwidth(20)
    to_plot = df.groupby(['frac', 'model_code', 'phase']).agg(
        {'time': ['mean', ('error', get_confidence_error)]}).unstack(['phase'])
    yerr = to_plot.loc[:, ('time', 'error', slice(None))]
    to_plot.plot.bar(stacked=True, y=('time', 'mean'), yerr=yerr.values.T, rot=45, ax=ax)
    xticklabels = ax.xaxis.get_ticklabels()
    for label in xticklabels:
        label.set_ha('right')


def phase_time_vs_frac(df, ax, fig, y_log=True):
    to_plot = df.groupby(['frac', 'phase']).agg({'time': ['mean', ('error', get_confidence_error)]}).unstack('phase')
    yerr = to_plot.loc[:, ('time', 'error', slice(None))]
    to_plot.plot(y=('time', 'mean'), yerr=yerr.values.T, rot=0, ax=ax, ylabel='time')
    if y_log:
        ax.set_yscale("log")
        ax.set_ylabel('time (log)')


def plot_metrics_time(df, ax, fig):
    to_plot = df.query('phase == "evaluation"').copy().reset_index(drop=True)
    convert_df = lambda x: pd.DataFrame(ast.literal_eval(x)).set_index('metric').T
    metric_times = pd.concat(to_plot['metrics_time'].apply(convert_df).values).reset_index()
    cols = ['frac']
    train_cols = list(metric_times.columns[metric_times.columns.str.startswith('train')])
    all_df = pd.concat([to_plot[cols], metric_times[train_cols]], 1)
    # all_df.boxplot(column=train_cols, ax=ax, rot=45);

    to_plot = all_df.groupby(cols).agg(['mean', ('error', get_confidence_error)])
    yerr = to_plot.loc[:, (slice(None), 'error')]
    to_plot.loc[:, (slice(None), 'mean')].plot(yerr=yerr.values.T, rot=0, ax=ax, ylabel='time')


def plot_routine_performance_violation(all_model_df, dataset_name, save=True, show=True, suffix='',
                                       base_plot_dir=PlotUtility.base_plot_dir, ):
    missed_conf = np.setdiff1d(all_model_df['model_code'].unique(),
                               list(PlotUtility.map_df.index.values)).tolist()
    assert len(missed_conf) == 0, missed_conf

    pl_util = PlotUtility(show=show, save=save, path_suffix=suffix)

    original_list = deepcopy(pl_util.to_plot_models)

    model_set_list = [(original_list, ''),
                      (restricted_list, '_restricted_'),
                      (['unconstrained_frac_exp'], '_unconstrained_'),
                      ]
    to_iter = list(itertools.product(['train', 'test'], ['error', 'violation', 'di'],
                                     [('time', all_model_df), ('frac', all_model_df)],
                                     model_set_list
                                     ))
    for phase, metric_name, (x_axis, turn_df), (to_plot_models, model_set_name) in to_iter:
        y_axis = f'{phase}_{metric_name}'
        plt.close('all')
        if model_set_name != '' and x_axis == 'frac':
            continue
        pl_util.to_plot_models = to_plot_models
        pl_util.plot(turn_df, dataset_name=dataset_name, y_axis=y_axis, x_axis=x_axis)
        pl_util.save_figure(base_plot_dir, dataset_name=dataset_name,
                            name=f'{model_set_name}{x_axis}_vs_{y_axis}')


def plot_routine_other(all_model_df, dataset_name, save=True, show=True, suffix='',
                       base_plot_dir=PlotUtility.base_plot_dir):
    pl_util = PlotUtility(show=show, save=save, path_suffix=suffix)
    df = all_model_df
    df = df[df['model_code'].isin(pl_util.to_plot_models)]
    df.loc[:, 'model_code'] = PlotUtility.map_df.loc[df['model_code'], 'label'].values
    split_name_value = re.compile("(?P<name>[a-zA-Z\_]+)\=(?P<value>[a-zA-Z]+)")
    model_code_map = {}
    for name in df['model_code'].unique():
        model_code_map[name] = ' '.join([f'{x[0][0]}={x[1][0]}' for x in split_name_value.findall(name)])
    df['model_code'] = df['model_code'].map(model_code_map)
    for name, plot_f in [
        ['metrics_time_vs_frac', plot_metrics_time],
        ['time_stacked_by_phase', time_stacked_by_phase],
        ['phase_time_vs_frac', phase_time_vs_frac],
    ]:

        if name == 'time_stacked_by_phase':
            old = plt.rcParams.get('savefig.dpi')
            plt.rcParams.update({'savefig.dpi': 400})
        pl_util.apply_plot_function_and_save(df=df, plot_name=name, plot_function=plot_f, dataset_name=dataset_name)

        if name == 'time_stacked_by_phase':
            plt.rcParams.update({'savefig.dpi': old})

    pl_util.plot(all_model_df, x_axis='frac', y_axis='time', dataset_name=dataset_name)
    if save is True:
        pl_util.save_figure(base_plot_dir, dataset_name=dataset_name, name=f'frac_vs_time')

    ### train_error_vs_eps
    # pl_util = PlotUtility(show=show)
    # phase, metric_name, x_axis = 'train', 'error', 'eps'
    # y_axis = f'{phase}_{metric_name}'
    # y_axis = 'time'
    # pl_util.to_plot_models = ['fairlearn_full_eps', 'expgrad_fracs_eps']
    # pl_util.plot(eps_df, y_axis=y_axis, x_axis='eps')
    # if save is True:
    #     pl_util.save(base_plot_dir, dataset_name=dataset_name, name=f'{y_axis}_vs_{x_axis}')


def select_rename_columns_to_plot(df, x_axis, y_axis):
    for key, column in {'x': x_axis, 'y': y_axis}.items():
        for (suffix, sub_col) in {'': 'mean', 'err': 'error'}.items():
            df[f'{key}{suffix}'] = df[f'{column}_{sub_col}']
    return df


def plot_all_df(all_df, model_list, save_dir, grouping_col, save, show):
    filtered_df = utils_results_data.prepare_data(all_df)
    filtered_df = filtered_df[filtered_df['model_code'].isin(model_list)]

    time_aggregated_df = utils_results_data.aggregate_phase_time(filtered_df)
    groupby_col = np.intersect1d(utils_results_data.cols_to_index, time_aggregated_df.columns).tolist()
    mean_error_df = time_aggregated_df.groupby(groupby_col, as_index=False, dropna=False)[
        utils_results_data.numerical_cols + [grouping_col]].agg(['mean', ('error', PlotUtility.get_error)]).reset_index()
    mean_error_df.columns = mean_error_df.columns.map('_'.join).str.strip('_')
    pl_util = PlotUtility(show=show, save=save, path_suffix='')
    result_path_name = 'all_df'
    axis_to_plot = [[grouping_col, 'time'],
                    [grouping_col, 'test_error'],
                    [grouping_col, 'test_violation']]
    dataset_name_list = mean_error_df['dataset_name'].unique()
    # Check available combinations
    # mean_error_df[['base_model_code', 'constraint_code', 'dataset_name']].astype('str').apply(
    #     lambda x: '_'.join(x.astype(str)), axis=1).unique().tolist()
    pl_util.show = False
    for keys, df_to_plot in mean_error_df.groupby(['base_model_code', 'constraint_code']):
        base_model_code, constraint_code = keys
        fig, axes_array = plt.subplots(nrows=len(axis_to_plot), ncols=len(dataset_name_list), sharex=True,
                                       sharey='row', figsize=np.array([6.4 * 1.5, 4.8]) * 1.5)
        pl_util.fig = fig
        for row, (x_axis, y_axis) in enumerate(axis_to_plot):
            df_to_plot = select_rename_columns_to_plot(df_to_plot, x_axis, y_axis)
            df_groups = df_to_plot.groupby(['dataset_name', 'base_model_code'])[
                ['x', 'xerr', 'y', 'yerr', grouping_col, 'model_code']]
            n_lines = len(df_groups)
            for col, ((dataset_name, base_model_code), value) in enumerate(df_groups):
                pl_util.ax = axes_array[row, col]
                for i, (model_code, value) in enumerate(value.groupby('model_code')):
                    value_dict = value[['x', 'xerr', 'y', 'yerr']].to_dict(orient='list')
                    x, xerr, y, yerr = value_dict.values()
                    grouping_values = value[grouping_col]
                    line_params = pl_util.get_line_params(i)
                    markers_params = pl_util.get_marker_params(i, total=n_lines, grouping_values=None)
                    label_params = pl_util.get_label_params(i, model_code=model_code)
                    label_params['label'] += f' | {dataset_name}'
                    # pl_util.ax.errorbar(**turn_values_dict, **line_params)
                    pl_util.ax.errorbar(**value_dict, **line_params)
                    pl_util.ax.scatter(x, y, **markers_params)
                    pl_util.ax.errorbar([], [], xerr=[], yerr=[], **label_params)
                    pl_util._end_plot(x_axis, y_axis, f'{dataset_name}')
                    pl_util.ax.set_title(f'{dataset_name}')
                    pl_util.ax.get_legend().remove()

        axes_array[1, 0].set_ylim(0, 0.4)
        for ax in axes_array[1:].flat:
            ax.set_title('')
        for ax in axes_array.flat:
            ax.label_outer()
        pl_util.fig.suptitle(f'{base_model_code} - {constraint_code}')
        # pl_util.fig.show()
        pl_util.save_figure(save_dir, dataset_name=result_path_name,
                            name=f'all_{base_model_code}_{constraint_code}_VARY_{grouping_col}_subplots')

    for keys, df_to_plot in mean_error_df.groupby(['base_model_code', 'constraint_code']):
        base_model_code, constraint_code = keys
        for x_axis, y_axis in [
            ['time', 'test_error'],
            ['time', 'test_violation'],
            ['time', 'test_di', ],
            ['test_violation', 'test_error'],
            ['test_di', 'test_error'],
        ]:
            pl_util._start_plot()
            df_to_plot = select_rename_columns_to_plot(df_to_plot, x_axis, y_axis)
            df_groups = df_to_plot.groupby(['dataset_name', 'model_code', 'base_model_code'])[
                ['x', 'xerr', 'y', 'yerr', grouping_col]]
            n_lines = len(df_groups)
            for i, (key, value) in enumerate(df_groups):
                dataset_name, model_code, base_model_code = key
                value_dict = value[['x', 'xerr', 'y', 'yerr']].to_dict(orient='list')
                x, xerr, y, yerr = value_dict.values()
                grouping_values = value[grouping_col]
                line_params = pl_util.get_line_params(i)
                markers_params = pl_util.get_marker_params(i, total=n_lines, grouping_values=grouping_values)
                label_params = pl_util.get_label_params(index=i, model_code=model_code)
                label_params['label'] += f' | {dataset_name}'
                pl_util.ax.errorbar(**value_dict, **line_params)
                pl_util.ax.scatter(x, y, **markers_params)
                pl_util.ax.errorbar([], [], xerr=[], yerr=[], **label_params)
                to_annotate = list(zip(x, y, grouping_values))
                for tx, ty, tv in [to_annotate[0], to_annotate[-1]]:
                    plt.text(tx, ty, f' {tv:.3g}', fontsize=10, va='center')
            pl_util._end_plot(x_axis, y_axis, title=f'{base_model_code} - VARY {grouping_col}')
            pl_util.save_figure(save_dir, dataset_name=result_path_name,
                                name=f'all_{base_model_code}_{constraint_code}_VARY_{grouping_col}_{x_axis}_vs_{y_axis}')


save = True
show = False
if __name__ == '__main__':
    df_list = []

    datasets = [
        "ACSPublicCoverage",
        "ACSEmployment",
        "adult"
    ]

    dataset_results_path = os.path.join("results", "fairlearn-2")
    for dataset_name in datasets:
        dirs_df = load_results(dataset_results_path, dataset_name)
        df_list.append(dirs_df)
    all_dirs_df = pd.concat(df_list)
    all_results_df = pd.concat(all_dirs_df['df'].values)

    df = all_results_df.copy()

    # Evaluate delta error
    df['delta_error'] = df['train_error'] - df['test_error']
    curr_path = os.path.join(dataset_results_path, 'all_dataset_stats')
    os.makedirs(curr_path, exist_ok=True)
    df.groupby(['dataset_name', 'base_model_code']).agg({'delta_error': 'describe'}).to_csv(
        os.path.join(curr_path, 'delta_error.csv'))

    grid_chart_models = ['hybrid_3_exp_gf_1',  'sub_hybrid_3_exp_gf_1',
                         'hybrid_5_exp', 'sub_hybrid_5_exp',
                         'expgrad_fracs_exp', 'hybrid_7_exp',
                         'hybrid_3_exp', 'sub_hybrid_3_exp' #sqrt
                         ]

    # all datasets
    selected_model = ['sub_hybrid_6_exp_gf_1']
    plot_all_df(all_results_df, model_list=selected_model, save_dir=PlotUtility.base_plot_dir, grouping_col='exp_frac',
                save=save, show=show)

    del df
    for dataset_name in datasets:
        for base_model_code in ['lr', 'lgbm']:
            turn_results_all = all_dirs_df.query(f'dataset_name == "{dataset_name}" and '
                                                 f'base_model_code == "{base_model_code}"')
            hybrids_results = filter_results(turn_results_all, conf=dict(
                exp_grid_ratio='sqrt',
                states='',
                exp_subset='True',
                eps='0.01',
                # run_linprog_step='True'
            ))
            other_results = filter_results(turn_results_all.query('model != "hybrids"'))

            # all_model_df.query('frac > 0.04').pivot_table(index=['frac'], columns=['model_name', grouping_col],
            #                          values=['train_violation', 'train_di', 'test_violation', 'test_di']).plot(
            #     kind='scatter')

            # plt.show()
            if not hybrids_results.empty:
                hybrids_results = hybrids_results[~hybrids_results['model_code'].str.contains('eps')]
                suffix = f'_bmc({base_model_code})' if base_model_code != 'lr' else ''

                plot_routine_performance_violation(pd.concat([hybrids_results, other_results]), dataset_name,
                                                   save=save, show=show,
                                                   suffix='ALL MODELS' + suffix)

                # if base_model_code == 'lr': # take always max grid oracle times
                # Take max of oracle calls time for grid search
                grid_mask = hybrids_results['phase'] == 'grid_frac'
                grid_time_series = hybrids_results[grid_mask]['grid_oracle_times'].apply(
                    lambda x: np.array(ast.literal_eval(x)).max())
                hybrids_results.loc[grid_mask, 'time'] = grid_time_series

                # ONLY ORACLE CALLLS
                df_only_oracle_calls = hybrids_results[hybrids_results['phase'].isin(['expgrad_fracs', 'grid_frac'])]
                exp_mask = hybrids_results['phase'] == 'expgrad_fracs'
                exp_time_df = hybrids_results[exp_mask]['oracle_execution_times_'].agg(
                    lambda x: pd.DataFrame(ast.literal_eval(x)).sum())
                exp_time_df.columns += '_sum'
                df_only_oracle_calls.loc[exp_mask, 'time'] = exp_time_df['fit_sum']
                plot_routine_performance_violation(df_only_oracle_calls, dataset_name=dataset_name,
                                                   save=save, show=show, suffix='ONLY ORACLE CALLS' + suffix)
                # df_cut = df_cut.join(exp_time_df)

                plot_routine_performance_violation(hybrids_results, dataset_name,
                                                   save=save, show=show, suffix=suffix)
                plot_routine_other(hybrids_results, save=save, show=show, suffix=suffix, dataset_name=dataset_name)
                hybrids_results['dataset_name'] = dataset_name


            else:
                print(f'{dataset_name} - {base_model_code} MISSING')
