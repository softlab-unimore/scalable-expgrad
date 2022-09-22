from graphic_adult import *

if __name__ == "__main__":
    alphas = [0.5]  # , 0.5, 0.95]


    data_sizes = [10**x for x in range(4,8)]

    df_list = []

    for n in data_sizes:
        base_dir = os.path.join("results", "sparc20", f'synth_n{n}_f3_t0.5_t00.3_t10.6_tr0.3_v1')
        turn_df = get_last_results(base_dir)
        turn_df['n'] = n
        df_list.append(turn_df.copy())
    all_model_df = pd.concat(df_list)
    all_model_df = add_combined_stats(all_model_df, alphas)
    # sns.set(font_scale=.8)
    fig = PlotUtility(all_model_df, x_axis='n')
    for phase in ['train','test']:
        for metric_name in ['violation','error']:
            PlotUtility(all_model_df, x_axis='n', y_axis=f'{phase}_{metric_name}')
    # time_plot(all_model_df)
    # error_plot(results)
    # error_plot(results, dataset_portion='test')


