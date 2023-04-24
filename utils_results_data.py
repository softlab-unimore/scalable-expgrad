import itertools
import os
import re
import socket
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import sem, t
from run import params_initials_map
from utils_experiment import index_cols

suffix_attr_map = {
    'exp': 'exp_frac',
    'eps': 'eps',
    'gri': 'grid_frac',
}




def select_set_expgrad_time(df: pd.DataFrame) -> pd.DataFrame:
    sub_mask = df['model_name'].str.contains('sub')
    lp_off_mask = df['model_name'].str.contains('LP_off')
    no_h7_mask = ~(df['model_name'].str.contains('hybrid_7'))
    for lp_mask in [lp_off_mask, ~lp_off_mask]:
        if lp_mask.any():
            turn_mask = ~sub_mask & lp_mask & no_h7_mask  # not sub is static sampling
            expgrad_time = df[turn_mask & (df['model_name'].str.contains('expgrad_frac'))]['time'].values[0]
            df.loc[turn_mask, 'time'] = expgrad_time

            turn_mask = sub_mask & lp_mask  # sub is adaptive sampling --> take h7 that is expgrad adaptive
            expgrad_time = df[lp_mask & (df['model_name'].str.contains('hybrid_7'))]['time'].values[0]
            df.loc[turn_mask, 'time'] = expgrad_time
    return df


def filter_results(dirs_df, conf: dict = {}):
    for key, value in conf.items():
        if not isinstance(value, list):
            value = [value]
        if key in dirs_df.columns:
            dirs_df = dirs_df[dirs_df[key].isin(value)]
        else:
            if key == 'states':
                pass
            else:
                assert False, f'\'{key}\' is not a valid key for filter. values available are: {dirs_df.columns.values}'
    if dirs_df.empty:
        return dirs_df
    df = pd.concat(dirs_df['df'].values)
    for split in ['train', 'test']:
        df[split + '_di'] = pd.concat([df[split + '_di'], 1 / df[split + '_di']], axis=1).min(axis=1)
        df[split + '_accuracy'] = 1 - df[split + '_error']
        df[split + '_TPRB'] = 1 - df[split + '_TPRB']
        df[split + '_TNRB'] = 1 - df[split + '_TNRB']
    return df.reset_index(drop=True)


def read_experiment_configuration(path):
    config = dict(dir=path)
    model = re.findall(r'(?P<model>^[a-zA-Z]+)_', path)[0]
    g = re.findall(r'(?P<name>[a-z]+)\((?P<value>[a-zA-Z0-9\.]+)\)\_?', path)
    config = {params_initials_map[x[0]]: x[1] for x in g}
    config['model'] = model
    return config



def load_results(dataset_path, dataset_name, prefix='last', read_files=False):
    base_dir = os.path.join(dataset_path, dataset_name)
    dirs = pd.Series([x for x in os.scandir(base_dir) if x.is_dir() and x.name != 'tuned_models'])
    if read_files:
        dirs = pd.Series([x for x in os.scandir(base_dir) if x.is_file()])

    config_list = []
    for turn_dir in dirs:
        config = read_experiment_configuration(turn_dir.name)
        config['dataset_name'] = dataset_name
        if read_files:
            df = pd.read_csv(turn_dir)
        else:
            df = load_results_single_directory(turn_dir.path, prefix=prefix)

        df = calculate_movign_param(turn_dir.path, df)

        if 'grid_fractions' in config.keys() and config['grid_fractions'] == '1.0':
            # mask = ~df['model_code'].str.contains('|'.join(['expgrad_fracs', 'hybrid_7', 'unconstrained_']))
            models_with_gridsearch  = df.query('phase == "grid_frac"')['model_code'].unique()
            mask = df['model_code'].isin(models_with_gridsearch)
            df.loc[mask, 'model_code'] += '_gf_1'
        for key, value in config.items():
            if key not in df.columns:
                df[key] = value
        config['df'] = df
        config_list.append(config)
    return pd.DataFrame(config_list).fillna({'constraint_code': 'dp', 'train_test_split':'0'}).fillna('')


def load_results_single_directory(base_dir, prefix='last'):
    files = pd.Series([x.name for x in os.scandir(base_dir) if x.is_file()])
    if files.shape[0] == 0:
        print(f'empty directory {base_dir}')
    filesToScan = files[files.str.startswith(prefix)]
    # name_df = last_files.str.extract(r'^(last)_([^_]*)_?(.*)\.(.*)$', expand=True)
    # name_df.rename(columns={0: 'last', 1: 'model', 2: 'params', 3: 'extension'}, inplace=True)
    df_list = []
    for turn_file in filesToScan:
        full_path = os.path.join(base_dir, turn_file)
        df_list.append(pd.read_csv(full_path))
    all_df = pd.concat(df_list)
    all_df = calculate_movign_param(base_dir, all_df)

    if 'rls(False)' in base_dir:
        to_drop = all_df[all_df['model_name'].isin(['unconstrained', 'unconstrained_frac'])].index
        all_df.drop(index=to_drop, inplace=True)

    return all_df


def calculate_movign_param(path, df: pd.DataFrame):
    cols_to_check = suffix_attr_map.values()
    if np.intersect1d(list(cols_to_check), df.columns).shape[0] < len(cols_to_check):
        df['frac'] = 1
        df['model_code'] = df['model_name']
        return df
    suffix = ''
    for key, name in suffix_attr_map.items():
        if f'_{key}[' in path:
            suffix += f'{key}'

    if suffix == '':
        for key, name in suffix_attr_map.items():
            if df[name].nunique() > 1:
                suffix = f'{key}'
                break
    df['moving_param'] = suffix
    df['model_code'] = df['model_name'] + '_' + df['moving_param']

    for key, col in suffix_attr_map.items():
        mask = df['moving_param'] == key
        df.loc[mask, 'frac'] = df.loc[mask, col]
    fix_expgrad_times(df)
    return df



def fix_expgrad_times(df: pd.DataFrame) -> pd.DataFrame:
    expgrad_phase_mask = df['phase'] == "expgrad_fracs"
    expgrad_df = df[expgrad_phase_mask]
    expgrad_df = expgrad_df.groupby(index_cols + ['frac']).apply(select_set_expgrad_time)
    df[expgrad_phase_mask] = expgrad_df

def aggregate_phase_time(df):
    results_df = df.groupby(df.columns.drop(['metrics_time', 'phase', 'time', 'grid_oracle_times']).tolist(),
                            as_index=False, dropna=False).agg({'time': 'sum'})
    return results_df


def get_info():
    host_name = socket.gethostname()
    if "." in host_name:
        host_name = host_name.split(".")[-1]

    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return host_name, current_time_str


def get_last_results_datetime(base_dir):
    files = pd.Series(os.listdir(base_dir))
    name_df = files.str.extract(r'^(\d{4}-\d{2}-\d{2})_((?:\d{2}-{0,1}){3})_(.*)\.(.*)$', expand=True)
    name_df.rename(columns={0: 'date', 1: 'time', 2: 'model', 3: 'extension'}, inplace=True)
    name_df['full_name'] = files
    name_df = name_df.query('extension == "csv"')
    last_files = name_df.sort_values(['date', 'time'], ascending=False).groupby('model').head(1)
    df_dict = {model_name: pd.read_csv(os.path.join(base_dir, turn_name))
               for turn_name, model_name in (last_files[['full_name', 'model']].values)}
    all_model_df = pd.concat(df_dict.values())
    return all_model_df


def get_confidence_error(data, confidence: float = 0.95):
    a = np.asarray(data).astype(float)
    n = len(a)
    se = sem(a, nan_policy="omit", ddof=1)
    t_value = t.ppf((1.0 + confidence) / 2., n - 1)
    return 2 * se * t_value


def mean_confidence_interval(data, confidence: float = 0.95):
    """
    Args:
        data:
        confidence:

    Returns:
        mean and confidence limit values for the given data and confidence
    """
    if data is None:
        return [np.nan, np.nan, np.nan]

    a = np.asarray(data).astype(float)
    n = len(a)
    m, se = np.nanmean(a, 0), sem(a, nan_policy="omit", ddof=1)
    t_value = t.ppf((1.0 + confidence) / 2., n - 1)
    h1 = m - se * t_value
    h2 = m + se * t_value
    return np.array([m, h1, h2])


def add_combined_stats(df, alphas=[.05, .5, .95]):
    not_combined_df = df.loc[df['model_name'] != "combined"]
    cols_to_group = ['eps', 'frac', 'random_seed', 'grid_frac', 'n']
    cols_to_group = np.intersect1d(cols_to_group, df.columns).tolist()
    combo_stat_list = []
    for alpha in alphas:
        turn_f = partial(get_combined_groupby, alpha=alpha)
        combined_stats = df.groupby(cols_to_group, as_index=False).apply(turn_f)
        combo_stat_list.append(combined_stats.copy())
    df = pd.concat(combo_stat_list + [not_combined_df]).drop_duplicates().reset_index(drop=True)
    return df


def get_combined_hybrid(train_err_hybrids, train_vio_hybrids, alpha):
    # alpha = importance of error vs. violation
    n = len(train_err_hybrids)
    if len(train_vio_hybrids) != n:
        raise Exception()
    scores = [
        alpha * train_err_hybrids[i] + (1 - alpha) * train_vio_hybrids[i]
        for i in range(n)
    ]
    best_index = scores.index(min(scores))
    return best_index


def get_combined_groupby(x, alpha=0.5):
    hybrid_res = x[x['model_name'].str.startswith('hybrid')]
    composed_metric = hybrid_res['train_violation'] * (1 - alpha) + hybrid_res['train_error'] * alpha
    combo_res = hybrid_res.loc[composed_metric.idxmin()]
    comb_df = x[x['model_name'] == 'combined']
    for col in np.setdiff1d(comb_df.columns,
                            ['eps', 'frac', 'model_name', 'time', 'phase', 'random_seed', 'grid_frac', 'n']):
        comb_df[col] = combo_res[col]
    comb_df['alpha'] = alpha
    return comb_df
