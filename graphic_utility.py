import ast
import itertools
from copy import deepcopy

import numpy as np
import os, re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns;
import pandas as pd
from utils_results_data import load_results_single_directory, get_info, get_confidence_error, mean_confidence_interval, \
    add_combined_stats, aggregate_phase_time, load_datasets_from_directory, set_frac_values, filter_results
import matplotlib as mpl
from run import params_initials_map
from utils_values import index_cols

sns.set()  # for plot styling
# sns.set(rc={'figure.figsize':(8,6)})
# sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 16, "figure.dpi": 400, 'savefig.dpi': 600,
                     # 'figure.figsize': (16 * 2 / 3, 9 * 2 / 3)
                     })
plt.rcParams['figure.constrained_layout.use'] = True
sns.set_context(rc={"legend.fontsize": 7})
base_plot_dir = os.path.join('results', 'plots')

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
    map_df = generate_map_df()

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
    ]

    color_list = mpl.colormaps['tab20'].colors
    suffix = ''

    # sns.color_palette("hls", len(self.to_plot_models))
    # color_list = list(mcolors.TABLEAU_COLORS.keys())
    def __init__(self, show=True, suffix='', save=True):
        self.markersize = 4
        self.linewidth = 0.5
        self.show = show
        self.suffix = suffix
        self.save_flag = save
        # plt.rcParams['lines.markersize'] = self.markersize
        # plt.rcParams['lines.linewidth'] = self.linewidth

    def plot(self, all_model_df, x_axis='frac', y_axis='time', alphas=[0.05, 0.5, 0.95],
             grid_fractions=[0.1, 0.2, 0.5], groupby_col='frac'):
        self.groupby_col = groupby_col
        self.fig = plt.figure()
        ax = plt.subplot()

        def_alpha = .5
        all_model_df = all_model_df[all_model_df['model_code'].isin(self.to_plot_models)]
        time_aggregated_df = aggregate_phase_time(all_model_df)
        time_aggregated_df[self.groupby_col].fillna(1, inplace=True)
        self.x_values = time_aggregated_df[self.groupby_col].unique()
        self.n_points = len(self.x_values)
        to_iter = time_aggregated_df[time_aggregated_df['model_code'].isin(self.to_plot_models)].groupby(['model_code'],
                                                                                                         dropna=False)
        for model_code, turn_df in to_iter:
            # label, color = map_df.loc[model_code, ['label', 'color']].values
            label = self.map_df.loc[model_code, 'label']
            # label = model_code
            index = self.to_plot_models.index(model_code)
            color = self.color_list[index % len(self.color_list)]
            n_models = len(self.to_plot_models)

            if x_axis == 'frac':
                x_offset = (((index / n_models) - 0.5) * 20 / 100) + 1
            else:
                x_offset = 1
            self.add_plot(ax, turn_df, x_axis, y_axis, color, label, x_offset_relative=x_offset)

        ax.set_xlabel(f'{x_axis} (log scale)')
        ax.set_ylabel(y_axis)
        ax.set_title(f'{y_axis} v.s. {x_axis}')
        ax.set_xscale("log")
        if y_axis == 'time':
            ax.set_yscale("log")
            ylabel = ax.get_ylabel()
            ax.set_ylabel(f'{ylabel} (log)')

        ax.legend()
        self.ax = ax
        if self.show:
            self.fig.show()

    def add_plot(self, ax, turn_df, x_axis, y_axis, color, label, x_offset_relative=1, ):
        agg_x_axis = self.groupby_col if x_axis == 'time' else x_axis
        turn_data = turn_df.pivot(index=index_cols, columns=agg_x_axis, values=y_axis)
        ci = mean_confidence_interval(turn_data)
        yerr = (ci[2] - ci[1]) / 2
        y_values = ci[0]
        zorder = 10 if len(y_values) == 1 else None
        if x_axis == 'time':
            time_data = turn_df.pivot(index=index_cols, columns=agg_x_axis, values='time')
            ci_x = mean_confidence_interval(time_data)
            xerr = (ci_x[2] - ci_x[1]) / 2
            x_values = ci_x[0]
        else:
            xerr = None
            x_values = turn_data.columns
        if label not in ['UNMITIGATED full']:
            ax.errorbar(x_values * x_offset_relative, y_values, xerr=xerr, yerr=yerr, color=color, label=label,
                        fmt='--x',
                        zorder=zorder,
                        markersize=self.markersize, linewidth=self.linewidth, elinewidth=self.linewidth / 2)

        if label in ['UNMITIGATED full', 'EXPGRAD=static GS=off LP=off', 'UNMITIGATED=static']:
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

    def save(self, base_dir, dataset_name, name, fig=None):
        if self.save_flag:
            if fig is None:
                fig = self.fig
            self.save_figure(base_dir, dataset_name, name, fig, suffix=self.suffix)

    @staticmethod
    def save_figure(base_dir, dataset_name, name, fig, suffix=''):
        host_name, current_time_str = get_info()
        dir_path = os.path.join(base_dir, dataset_name, host_name + suffix)
        for t_dir in [dir_path]:
            for t_name in [
                # f'{current_time_str}_{name}',
                f'last_{name}']:
                t_full_path = os.path.join(t_dir, t_name)
                os.makedirs(t_full_path, exist_ok=True)
                fig.savefig(t_full_path + '.png')
                t_full_path_svg = os.path.join(t_dir + '_svg', t_name)
                os.makedirs(t_full_path_svg, exist_ok=True)
                fig.savefig(t_full_path_svg + '.svg', format='svg')

    def apply_plot_function_and_save(self, df, plot_name, plot_function):
        plt.close('all')
        fig, ax = plt.subplots()
        plot_function(df, ax=ax, fig=fig)
        self.save(base_plot_dir, dataset_name=dataset_name, name=plot_name, fig=fig)
        if self.show:
            plt.show()


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


def plot_routine_performance_violation(all_model_df, save=True, show=True, suffix='', base_plot_dir=base_plot_dir):
    missed_conf = np.setdiff1d(all_model_df['model_code'].unique(),
                               list(PlotUtility.map_df.index.values)).tolist()
    assert len(missed_conf) == 0, missed_conf

    pl_util = PlotUtility(show=show, save=save, suffix=suffix)

    original_list = deepcopy(pl_util.to_plot_models)

    model_set_list = [(original_list, ''),
                      (restricted_list, '_restricted'),
                      (['unconstrained_frac_exp'], '_unconstrained'),
                      ]
    to_iter = list(itertools.product(['train', 'test'], ['error', 'violation'],
                                     [('time', all_model_df), ('frac', all_model_df)],
                                     model_set_list
                                     ))
    for phase, metric_name, (x_axis, turn_df), (to_plot_models, model_set_name) in to_iter:
        plt.close('all')
        if model_set_name != '' and x_axis == 'frac':
            continue
        pl_util.to_plot_models = to_plot_models
        pl_util.plot(turn_df, y_axis=f'{phase}_{metric_name}', x_axis=x_axis)
        pl_util.save(base_plot_dir, dataset_name=dataset_name,
                     name=f'{phase}_{metric_name}_vs_{x_axis}{model_set_name}')


def plot_routine_other(all_model_df, save=True, show=True, suffix='', base_plot_dir=base_plot_dir):
    pl_util = PlotUtility(show=show, save=save, suffix=suffix)
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
        pl_util.apply_plot_function_and_save(df=df, plot_name=name, plot_function=plot_f)

        if name == 'time_stacked_by_phase':
            plt.rcParams.update({'savefig.dpi': old})

    pl_util.plot(all_model_df, x_axis='frac', y_axis='time')
    if save is True:
        pl_util.save(base_plot_dir, dataset_name=dataset_name, name=f'frac_vs_time')

    ### train_error_vs_eps
    # pl_util = PlotUtility(show=show)
    # phase, metric_name, x_axis = 'train', 'error', 'eps'
    # y_axis = f'{phase}_{metric_name}'
    # y_axis = 'time'
    # pl_util.to_plot_models = ['fairlearn_full_eps', 'expgrad_fracs_eps']
    # pl_util.plot(eps_df, y_axis=y_axis, x_axis='eps')
    # if save is True:
    #     pl_util.save(base_plot_dir, dataset_name=dataset_name, name=f'{y_axis}_vs_{x_axis}')


if __name__ == '__main__':
    save = True
    show = False
    df_list = []

    datasets = [
        "ACSPublicCoverage",
        "ACSEmployment",
        "adult",
    ]

    dataset_results_path = os.path.join("results", "fairlearn-2")
    for dataset_name in datasets:
        dirs_df = load_datasets_from_directory(dataset_results_path, dataset_name)
        df_list.append(dirs_df)
    all_dirs_df = pd.concat(df_list)
    all_results_df = pd.concat(all_dirs_df['df'].values)

    df = all_results_df.copy()
    df['delta_error'] = df['train_error'] - df['test_error']
    curr_path = os.path.join(dataset_results_path, 'all_dataset_stats')
    os.makedirs(curr_path, exist_ok=True)
    df.groupby(df['dataset_name','base_model_code']).agg({'delta_error': 'describe'}).to_csv(os.path.join(curr_path, 'delta_error.csv'))
    del df

    for dataset_name in datasets:
        for base_model_code in ['lgbm', 'lr', ]:
            all_model_df = filter_results(all_dirs_df, conf=dict(
                # exp_grid_ratio='sqrt',
                states='',
                exp_subset='True', base_model_code=base_model_code,
                dataset_name=dataset_name,
                # run_linprog_step='True'
            ))
            if not all_model_df.empty:
                all_model_df = all_model_df[~all_model_df['model_code'].str.contains('eps')]
                suffix = f'_bmc({base_model_code})' if base_model_code != 'lr' else ''

                # if base_model_code == 'lr': # take always max grid oracle times
                grid_mask = all_model_df['phase'] == 'grid_frac'
                grid_time_series = all_model_df[grid_mask]['grid_oracle_times'].apply(
                    lambda x: np.array(ast.literal_eval(x)).max())
                all_model_df.loc[grid_mask, 'time'] = grid_time_series

                df_cut = all_model_df[all_model_df['phase'].isin(['expgrad_fracs', 'grid_frac'])]
                exp_mask = all_model_df['phase'] == 'expgrad_fracs'
                exp_time_df = all_model_df[exp_mask]['oracle_execution_times_'].agg(
                    lambda x: pd.DataFrame(ast.literal_eval(x)).sum())
                exp_time_df.columns += '_sum'
                df_cut.loc[exp_mask, 'time'] = exp_time_df['fit_sum']
                plot_routine_performance_violation(df_cut, save=save, show=show, suffix='ONLY ORACLE CALLS' + suffix)
                # df_cut = df_cut.join(exp_time_df)

                plot_routine_performance_violation(all_model_df, save=save, show=show, suffix=suffix)
                plot_routine_other(all_model_df, save=save, show=show, suffix=suffix)
                all_model_df['dataset_name'] = dataset_name


            else:
                print(f'{dataset_name} - {base_model_code} MISSING')
