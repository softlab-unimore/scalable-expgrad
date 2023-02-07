from functools import partial

from graphic_utility import *
from utils_results_data import load_results_single_directory, add_combined_stats

# sns.set_context(font_scale=.9)


if __name__ == "__main__":
    alphas = [0.5]  # , 0.5, 0.95]
    data_sizes = [10 ** x for x in range(4, 8)]
    df_list = []
    for n in data_sizes:
        base_dir = os.path.join("results", "sparc20", f'synth_n{n}_f3_t0.5_t00.3_t10.6_tr0.3_v1')
        turn_df = load_results_single_directory(base_dir)
        turn_df['n'] = n
        df_list.append(turn_df.copy())
    all_model_df = pd.concat(df_list)
    all_model_df = add_combined_stats(all_model_df, alphas)


    save = True
    base_plot_dir = os.path.join('results', 'plots')
    PlotUtility_n = partial(PlotUtility, groupby_col='n')
    pl = PlotUtility_n(all_model_df, x_axis='n', y_axis='time')
    if save is True:
        pl.save_figure(base_plot_dir, dataset_name='synth', name=f'n_vs_time')
    itertools.product()
    to_iter = list(itertools.product(['train', 'test'], ['violation', 'error'], ['n', 'time']))
    for phase, metric_name, x_axis in to_iter:
        turn_plot = PlotUtility_n(all_model_df, y_axis=f'{phase}_{metric_name}', x_axis=x_axis)
        if save is True:
            turn_plot.save_figure(base_plot_dir, dataset_name='synth', name=f'{phase}_{metric_name}_vs_{x_axis}')
    # time_plot(all_model_df)
    # error_plot(results)
    # error_plot(results, dataset_portion='test')
