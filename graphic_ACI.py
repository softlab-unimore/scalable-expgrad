from functools import partial

from graphic_utility import *
from utils_results_data import load_results_single_directory, add_combined_stats

# sns.set_context(font_scale=.9)


if __name__ == "__main__":
    save = True
    base_plot_dir = os.path.join('results', 'plots')
    for dataset_name in ['ACSPublicCoverage',
                         'ACSIncome', 'ACSMobility', 'ACSEmployment', 'ACSTravelTime',
                         'ACSHealthInsurance', 'ACSEmploymentFiltered' 'ACSIncomePovertyRatio']:
        results_path = os.path.join("results", "fairlearn-2", dataset_name)
        try:
            all_model_df = load_results_single_directory(results_path)
        except Exception as e:
            print(e)
            continue
        eps_mask = all_model_df['model_name'].str.endswith('_eps')
        eps_df = all_model_df[eps_mask]
        frac_df = all_model_df[~eps_mask]
        pl_util = PlotUtility()
        phase, metric_name, x_axis = 'train', 'error', 'eps'
        y_axis = f'{phase}_{metric_name}'
        y_axis = 'time'
        pl_util.to_plot_models = ['fairlearn_full_eps', 'expgrad_fracs_eps']
        pl_util.plot(eps_df, y_axis=y_axis, x_axis='eps')
        if save is True:
            pl_util.save(base_plot_dir, dataset_name=dataset_name, name=f'{y_axis}_vs_{x_axis}')