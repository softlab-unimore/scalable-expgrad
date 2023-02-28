import os

import pandas as pd

import folktables
from utils_prepare_data import load_transform_ACS
from run import ExpreimentRun

if __name__ == '__main__':
    er = ExpreimentRun()
    descriptions_dir = os.path.join(er.base_result_dir, 'ACSDataset_descriptions')
    os.makedirs(descriptions_dir, exist_ok=True)
    dict_list = []
    t_dict = {}
    for dataset_str in ['ACSEmploymentFiltered', 'ACSIncomePovertyRatio',
                        'ACSMobility',
                        'ACSPublicCoverage', 'ACSEmployment', 'ACSTravelTime',
                        'ACSHealthInsurance',
                        'ACSIncome',
                        ]:
        loader_method = getattr(folktables, dataset_str)
        X, y, A, acs_data = load_transform_ACS(loader_method=loader_method, return_acs_data=True)
        # acs_data.iloc[:1000].to_csv(f'results/fairlearn-2/ACSDataset_descriptions/ACSPublicCoverage_h1000.csv')
        # acs_data.iloc[:1000].loc[:, ['HISP', 'RAC1P', 'PUBCOV']].to_csv(f'results/fairlearn-2/ACSDataset_descriptions/ACSPublicCoverage_h1000_race.csv')
        mem_usage = X.memory_usage().sum() / (2 ** (10 * 3))
        t_dict.update(dataset_name=dataset_str, size=X.shape[0], columns=X.shape[1], sensitive_attr=A.name,
                      sensitive_attr_nunique=A.nunique(), target_col=y.name, sensitive_attr_unique_values=A.unique(),
                      mem_usage=mem_usage)
        desc = X.describe().join([y.describe(), A.describe()])
        desc.to_csv(os.path.join(descriptions_dir, dataset_str + 'describe.csv'))
        dict_list.append(t_dict.copy())
    pd.DataFrame(dict_list).to_csv(os.path.join(descriptions_dir, 'all_df_descriptions_summary.csv'))
