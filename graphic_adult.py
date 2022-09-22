import plotly.express as px
import plotly.graph_objects as go
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns;


sns.set()  # for plot styling
import pandas as pd
from utils import mean_confidence_interval, aggregate_phase_time, get_last_results, add_combined_stats, get_info

sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
sns.set(font_scale=1)
# sns.set(rc={'figure.figsize':(8,6)})
# sns.set_context('notebook')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()


# plt.rcParams["figure.figsize"] = (10,5)
# ax = global_df.pivot_table(index=['id', 'match_code'], columns=['dataset_code'], values=['pearson']).droplevel(0,1).groupby(['match_code']).plot(kind='box')
# ax['match'].get_figure().savefig(os.path.join(...))
# ax['nomatch'].get_figure().savefig(os.path.join(...), bbox_inches='tight')


class PlotUtility():
    color_list = list(mcolors.TABLEAU_COLORS.keys())
    columns = ['label', 'color']
    model_to_params_map = {
        'expgrad_fracs': ['expgrad sample', color_list[3]],
        'hybrid_5': ['hybrid 5 (LP)', 'y'],
        'hybrid_1': ['hybrid 1 (GS only)', 'b'],
        'hybrid_2': ['hybrid 2 (GS + pmf_predict)', 'c'],
        'hybrid_3': ['hybrid 3 (GS + LP)', 'brown'],
        'hybrid_4': ['hybrid 4 (GS + LP+)', 'm'],
        'combined': ['hybrid combined', 'm'],
        'fairlearn_full': ['expgrad full', 'black'],
        'unmitigated': ['unmitigated', 'orange']}

    to_plot_models = [
        'expgrad_fracs',
        'hybrid_5',
        'hybrid_1',
        'hybrid_2',
        'hybrid_3',
        'hybrid_4',
        # 'combined',
        'fairlearn_full',
        'unmitigated'
    ]

    def __init__(self, all_model_df, x_axis='frac', y_axis='time', alphas=[0.05, 0.5, 0.95],
                 grid_fractions=[0.1, 0.2, 0.5]):
        ax = plt.subplot()
        self.base_plot(all_model_df, x_axis, y_axis, alphas, grid_fractions, ax)
        sns.axes_style("whitegrid")
        ax.legend()
        plt.tight_layout()
        plt.show()
        self.fig = plt.gcf()
        self.ax = ax


    def base_plot(self, all_model_df, x_axis, y_axis, alphas, grid_fractions, ax):
        def_alpha = .5
        # Print Time and plot them
        x_values = all_model_df[x_axis].dropna().unique()

        # fr = np.log10(fr)
        # all_model_df.loc[all_model_df['model_name'] == "combined", 'alpha'] = 0.5
        # all_model_df = all_model_df.query(f'grid_frac == {grid_fraction}')
        time_aggregated_df = aggregate_phase_time(all_model_df)

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
        agg_x_axis = 'frac' if x_axis == 'time' else x_axis
        turn_data = turn_df.pivot(index='random_seed', columns=agg_x_axis, values=y_axis)
        ci = mean_confidence_interval(turn_data)
        y_values = ci[0]
        if x_axis == 'time':
            time_data = turn_df.pivot(index='random_seed', columns='frac', values='time')
            ci_x = mean_confidence_interval(time_data)
            xerr = ci_x[2] - ci_x[1]
            yerr = ci[2]-ci[1]
            x_values = ci_x[0]
            if len(y_values) < len(x_values):
                y_values = [y_values] * len(x_values)
            ax.errorbar(x_values, y_values, xerr=xerr, yerr=yerr, color=color, label=label, marker="o")
        else:
            x_values = turn_data.columns
            ax.fill_between(x_values, ci[1], ci[2], color=color, alpha=0.3)
            if len(y_values) < len(x_values):
                y_values = [y_values] * len(x_values)
            ax.plot(x_values, y_values, color, label=label, marker="o")




    def save_figure(self, base_dir, dataset_name, name):
        host_name, current_time_str = get_info()
        base_dir = os.path.join(base_dir, dataset_name, host_name)
        path = os.path.join(base_dir, f'{current_time_str}_{name}')
        last_path = os.path.join(base_dir, f'last_{name}')
        try:
            os.makedirs(base_dir)
        except:
            pass
        for full_path in [path, last_path]:
            self.fig.savefig(full_path + '.svg', format='svg')
            self.fig.savefig(full_path + '.png')


if __name__ == '__main__':
    save = True
    base_dir = os.path.join("results", "sparc20", "adult")
    all_model_df = get_last_results(base_dir)
    all_model_df = add_combined_stats(all_model_df)


    base_plot_dir = os.path.join('results', 'plots')
    turn_plot = PlotUtility(all_model_df, y_axis=f'train_error', x_axis='time')
    if save is True:
        turn_plot.save_figure(base_plot_dir, dataset_name='adult', name=f'error_vs_time')

    pl = PlotUtility(all_model_df)
    if save is True:
        pl.save_figure(base_plot_dir, dataset_name='adult', name=f'frac_vs_time')
    for phase in ['train', 'test']:
        for metric_name in ['violation', 'error']:
            turn_plot = PlotUtility(all_model_df, y_axis=f'{phase}_{metric_name}')
            if save is True:
                turn_plot.save_figure(base_plot_dir, dataset_name='adult', name=f'{phase}_{metric_name}')


    # plot_utility(all_model_df, x_axis='time', y_axis='train_error')
