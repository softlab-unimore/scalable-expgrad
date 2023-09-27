import os

import pandas as pd

from run_experiments import utils_experiment
from run_experiments.utils_prepare_data import load_transform_ACS
from run import ExperimentRun

if __name__ == '__main__':
    er = ExperimentRun()
    descriptions_dir = os.path.join('..', er.base_result_dir, 'ACSDataset_descriptions')
    os.makedirs(descriptions_dir, exist_ok=True)
    dict_list = []
    t_dict = {}
    for dataset_str in set(utils_experiment.dataset_names) - set(['adult']):
        X, y, A, acs_data = load_transform_ACS(dataset_str, return_acs_data=True)
        # acs_data.iloc[:1000].to_csv(f'results/fairlearn-2/ACSDataset_descriptions/ACSPublicCoverage_h1000.csv')
        # acs_data.iloc[:1000].loc[:, ['HISP', 'RAC1P', 'PUBCOV']].to_csv(f'results/fairlearn-2/ACSDataset_descriptions/ACSPublicCoverage_h1000_race.csv')
        mem_usage = X.memory_usage().sum() / (2 ** (10 * 3))
        t_dict.update(dataset_name=dataset_str, size=X.shape[0], columns=X.shape[1], sensitive_attr=A.name,
                      sensitive_attr_nunique=A.nunique(), target_col=y.name, sensitive_attr_unique_values=A.unique(),
                      mem_usage=mem_usage)
        function_list = ['mean', 'std', 'size', ('perc', lambda x: x.size / y.shape[0])]
        target_stats = y.groupby(A).agg(function_list)
        target_stats.loc['mean'] = target_stats.mean()
        target_stats.loc['std'] = target_stats.std()
        target_stats.loc['all'] = target_stats.loc['all'] = y.agg(
            ['mean', 'std', 'size', (lambda x: x.size / y.shape[0])]).rename({'<lambda>': 'perc'})
        target_stats.to_csv(os.path.join(descriptions_dir, dataset_str + '_target_stats.csv'))
        desc = X.describe().join([y.describe(), A.describe()])
        desc.to_csv(os.path.join(descriptions_dir, dataset_str + 'describe.csv'))

        if dataset_str in ['adult', 'ACSPublicCoverage',
                           'ACSEmployment', ]:
            er.dataset_str = dataset_str
            best_params_dict = er.load_best_params(base_model_code='lr', fraction=1.)
            t_dict.update(**best_params_dict)
        dict_list.append(t_dict.copy())
    pd.DataFrame(dict_list).to_csv(os.path.join(descriptions_dir, 'all_df_descriptions_summary.csv'))
