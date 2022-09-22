import itertools

import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()  # for plot styling
import pandas as pd
from utils import mean_confidence_interval, aggregate_phase_time, get_last_results, add_combined_stats, get_info

# sns.set(rc={'figure.figsize':(8,6)})
# sns.set_context('notebook')
sns.set_style('whitegrid', rc={"figure.dpi": 300, 'savefig.dpi': 300})
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
sns.set_context(rc={"legend.fontsize": 8})


# plt.rcParams["figure.figsize"] = (10,5)
# ax = global_df.pivot_table(index=['id', 'match_code'], columns=['dataset_code'], values=['pearson']).droplevel(0,1).groupby(['match_code']).plot(kind='box')
# ax['match'].get_figure().savefig(os.path.join(...))
# ax['nomatch'].get_figure().savefig(os.path.join(...), bbox_inches='tight')


class PlotUtility():
    color_list = list(mcolors.TABLEAU_COLORS.keys())
    columns = ['label', 'color']
    model_to_params_map = {
        'expgrad_fracs': ['expgrad sample', 'red'],
        'hybrid_5': ['hybrid 5 (LP)', 'gold'],
        'hybrid_1': ['hybrid 1 (GS only)', 'blue'],
        'hybrid_2': ['hybrid 2 (GS + pmf_predict)', 'cyan'],
        'hybrid_3': ['hybrid 3 (GS + LP)', 'brown'],
        'hybrid_4': ['hybrid 4 (GS + LP+)', 'magenta'],
        'combined': ['hybrid combined', 'lime'],
        'fairlearn_full': ['expgrad full', 'black'],
        'unmitigated': ['unmitigated', 'orange']}

    to_plot_models = [
        'expgrad_fracs',
        'hybrid_5',
        # 'hybrid_1',
        # 'hybrid_2',
        'hybrid_3',
        # 'hybrid_4',
        # 'combined',
        'fairlearn_full',
        'unmitigated'
    ]

    def __init__(self, all_model_df, x_axis='frac', y_axis='time', alphas=[0.05, 0.5, 0.95],
                 grid_fractions=[0.1, 0.2, 0.5], groupby_col='frac'):
        self.groupby_col = groupby_col
        self.fig = plt.figure()
        ax = plt.subplot()
        self.base_plot(all_model_df, x_axis, y_axis, alphas, grid_fractions, ax)
        sns.axes_style("whitegrid")
        ax.legend()
        self.ax = ax
        self.fig.show()

    def base_plot(self, all_model_df, x_axis, y_axis, alphas, grid_fractions, ax):
        def_alpha = .5
        time_aggregated_df = aggregate_phase_time(all_model_df)
        time_aggregated_df[self.groupby_col].fillna(1, inplace=True)
        self.x_values = time_aggregated_df[self.groupby_col].unique()
        self.n_points = len(self.x_values)
        map_df = pd.DataFrame.from_dict(self.model_to_params_map, columns=self.columns, orient='index')

        to_iter = []
        combined_mask = time_aggregated_df['model_name'] == "combined"
        to_iter.append(time_aggregated_df[combined_mask].groupby(['model_name', 'alpha', 'grid_frac'], dropna=False))
        to_iter.append(time_aggregated_df[~combined_mask].groupby(['model_name'], dropna=False))
        for turn_group in to_iter:
            for key, turn_df in turn_group:
                if isinstance(key, tuple):
                    model_name, alpha, grid_frac = key
                    label, color = map_df.loc[model_name, ['label', 'color']].values
                    if alpha in alphas:
                        color = self.color_list[alphas.index(alpha)]
                        label += f' (alpha={alpha})'
                    if grid_frac in grid_fractions and model_name == 'combined':
                        if alpha == def_alpha:
                            color = self.color_list[grid_fractions.index(grid_frac)]
                        label += f' grid_fr={grid_frac})'
                else:
                    model_name = key
                    label, color = map_df.loc[model_name, ['label', 'color']].values
                if model_name not in self.to_plot_models:
                    continue
                if turn_df['grid_frac'].nunique() > 1:
                    turn_df = turn_df.query(f'grid_frac == 0.1')
                self.add_plot(ax, turn_df, x_axis, y_axis, color, label)
        ax.set_xlabel(f'{x_axis} (log scale)')
        ax.set_ylabel(y_axis)
        ax.set_title(f'{y_axis} v.s. {x_axis}')
        ax.set_xscale("log")
        if y_axis == 'time':
            ax.set_yscale("log")

    def add_plot(self, ax, turn_df, x_axis, y_axis, color, label):
        agg_x_axis = self.groupby_col if x_axis == 'time' else x_axis
        turn_data = turn_df.pivot(index='random_seed', columns=agg_x_axis, values=y_axis)
        ci = mean_confidence_interval(turn_data)
        y_values = ci[0]
        if x_axis == 'time':
            time_data = turn_df.pivot(index='random_seed', columns=agg_x_axis, values='time')
            ci_x = mean_confidence_interval(time_data)
            xerr = (ci_x[2] - ci_x[1]) / 2
            yerr = (ci[2] - ci[1]) / 2
            x_values = ci_x[0]
            if len(y_values) == 1:
                ax.axhline(y_values, linestyle="-.", color=color, zorder=10)
            zorder = 10 if len(y_values) == 1 else None
            ax.errorbar(x_values, y_values, xerr=xerr, yerr=yerr, color=color, label=label, fmt='--o', zorder=zorder)
        else:
            x_values = turn_data.columns
            ax.fill_between(x_values, ci[1], ci[2], color=color, alpha=0.3)
            if len(y_values) == 1:
                ax.plot(self.x_values, np.repeat(y_values, self.n_points), "-.", color=color, zorder=10, label=label)
            else:
                ax.plot(x_values, y_values, color, label=label, marker="o", linestyle='--')

    def save_figure(self, base_dir, dataset_name, name):
        host_name, current_time_str = get_info()
        base_dir = os.path.join(base_dir, dataset_name, host_name)
        path = os.path.join(base_dir, f'{current_time_str}_{name}')
        last_path = os.path.join(base_dir, f'last_{name}')
        try:
            os.makedirs(base_dir)
        except:
            pass
        for full_path in [
            # path,
            last_path]:
            self.fig.savefig(full_path + '.svg', format='svg')
            self.fig.savefig(full_path + '.png')




if __name__ == '__main__':
    save = True
    base_dir = os.path.join("results", "sparc20", "adult")
    all_model_df = get_last_results(base_dir)
    all_model_df = add_combined_stats(all_model_df)

    base_plot_dir = os.path.join('results', 'plots')
    pl = PlotUtility(all_model_df, x_axis='frac', y_axis='time')
    if save is True:
        pl.save_figure(base_plot_dir, dataset_name='adult', name=f'frac_vs_time')

    to_iter = list(itertools.product(['train', 'test'], ['violation', 'error'],[
                                         # 'frac',
                                         'time']
                                     ))
    for phase, metric_name, x_axis in to_iter:
        turn_plot = PlotUtility(all_model_df, y_axis=f'{phase}_{metric_name}', x_axis=x_axis)
        if save is True:
            turn_plot.save_figure(base_plot_dir, dataset_name='adult', name=f'{phase}_{metric_name}_vs_{x_axis}')




# class PlotUtilityPlotly(PlotUtility):
#     color_list = px.colors.qualitative.Plotly
#
#     def __init__(self, all_model_df, x_axis='frac', y_axis='time', alphas=[0.05, 0.5, 0.95],
#                  grid_fractions=[0.1, 0.2, 0.5]):
#         self.fig = go.Figure()
#         self.base_plot(all_model_df, x_axis, y_axis, alphas, grid_fractions)
#         self.fig.show()
#
#     def base_plot(self, all_model_df, x_axis, y_axis, alphas, grid_fractions):
#         def_alpha = .5
#         time_aggregated_df = aggregate_phase_time(all_model_df)
#
#         map_df = pd.DataFrame.from_dict(self.model_to_params_map, columns=self.columns, orient='index')
#
#         to_iter = []
#         combined_mask = time_aggregated_df['model_name'] == "combined"
#         to_iter.append(time_aggregated_df[combined_mask].groupby(['model_name', 'alpha', 'grid_frac'], dropna=False))
#         to_iter.append(time_aggregated_df[~combined_mask].groupby(['model_name'], dropna=False))
#         for turn_group in to_iter:
#             for key, turn_df in turn_group:
#                 if isinstance(key, tuple):
#                     model_name, alpha, grid_frac = key
#                     label, color = map_df.loc[model_name, ['label', 'color']].values
#                     if alpha in alphas:
#                         color = self.color_list[alphas.index(alpha)]
#                         label += f' (alpha={alpha})'
#                     if grid_frac in grid_fractions and model_name == 'combined':
#                         if alpha == def_alpha:
#                             color = self.color_list[grid_fractions.index(grid_frac)]
#                         label += f' grid_fr={grid_frac})'
#                 else:
#                     model_name = key
#                     label, color = map_df.loc[model_name, ['label', 'color']].values
#                 if model_name not in self.to_plot_models:
#                     continue
#                 if turn_df['grid_frac'].nunique() > 1:
#                     turn_df = turn_df.query(f'grid_frac == 0.1')
#                 self.add_plot(turn_df, x_axis, y_axis, color, label)
#
#         self.fig.update_layout(
#             title=f'{y_axis} v.s. {x_axis}',
#             xaxis_title=f'{x_axis} (log scale)',
#             yaxis_title=y_axis)
#         self.fig.update_xaxes(type="log")
#         if y_axis == 'time':
#             self.fig.update_yaxes(type="log")
#
#     def add_plot(self, turn_df, x_axis, y_axis, color, label):
#         agg_x_axis = self.groupby_col if x_axis == 'time' else x_axis
#         turn_data = turn_df.pivot(index='random_seed', columns=agg_x_axis, values=y_axis)
#         ci = mean_confidence_interval(turn_data)
#         y_values = ci[0]
#         if x_axis == 'time':
#             time_data = turn_df.pivot(index='random_seed', columns=self.groupby_col, values='time')
#             ci_x = mean_confidence_interval(time_data)
#             xerr = ci_x[2] - ci_x[1]
#             yerr = ci[2] - ci[1]
#             x_values = ci_x[0]
#             if len(y_values) < len(x_values):
#                 y_values = [y_values] * len(x_values)
#             self.fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', name=label,
#                                           error_y=dict(array=yerr), error_x=dict(array=xerr),
#                                           #                           dict(
#                                           #     type='constant',
#                                           #     value=0.2,
#                                           #     color='purple',
#                                           #     thickness=1.5,
#                                           #     width=3,
#                                           # ),
#                                           marker=dict(color=color, size=8)
#                                           ))
#             # ax.errorbar(x_values, y_values, xerr=xerr, yerr=yerr, color=color, label=label, marker="o")
#         else:
#             x_values = turn_data.columns
#             # self.fig.add_trace(go.(x_values, ci[1], ci[2], color=color, alpha=0.3))
#             if len(y_values) < len(x_values):
#                 y_values = [y_values] * len(x_values)
#             self.fig.add_trace(go.Scatter(x_values, y_values, color, label=label, marker="o"))
#             self.fig.add_trace(go.Scatter(x_values, y=ci[1], fill=None, mode=None))
#             self.fig.add_trace(go.Scatter(x=x_values, y=ci[2], fill='tonexty'))
#
#     def save_figure(self, base_dir, dataset_name, name):
#         host_name, current_time_str = get_info()
#         base_dir = os.path.join(base_dir, dataset_name, host_name)
#         path = os.path.join(base_dir, f'{current_time_str}_{name}')
#         last_path = os.path.join(base_dir, f'last_{name}')
#         try:
#             os.makedirs(base_dir)
#         except:
#             pass
#         for full_path in [
#             # path,
#             last_path]:
#             self.fig.write_image(full_path + '.svg')
#             self.fig.write_image(full_path + '.png')
#             self.fig.write_html(full_path + '.html')
