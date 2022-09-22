#!/usr/bin/env python
# coding: utf-8

import os
from datetime import datetime
from functools import partial
import socket

import numpy as np
import pandas as pd
from scipy.stats import sem, t
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

baseline_results_file_name = 'results/baseline_results (yeeha).json'
github_data_url = "https://github.com/slundberg/shap/raw/master/data/"


def adult(display=False):
    """ Return the Adult census data in a nice package. """
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_data = pd.read_csv(
        cache(github_data_url + "adult.data"),
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )
    data = raw_data.drop(["Education"], axis=1)  # redundant with Education-Num
    filt_dtypes = list(filter(lambda x: not (x[0] in ["Target", "Education"]), dtypes))
    data["Target"] = data["Target"] == " >50K"
    rcode = {
        "Not-in-family": 0,
        "Unmarried": 1,
        "Other-relative": 2,
        "Own-child": 3,
        "Husband": 4,
        "Wife": 5
    }
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "Relationship":
                data[k] = np.array([rcode[v.strip()] for v in data[k]])
            else:
                data[k] = data[k].cat.codes

    if display:
        return raw_data.drop(["Education", "Target", "fnlwgt"], axis=1), data["Target"].values
    else:
        return data.drop(["Target", "fnlwgt"], axis=1), data["Target"].values


def cache(url, file_name=None):
    if file_name is None:
        file_name = os.path.basename(url)
    data_dir = os.path.join(os.path.dirname("."), "cached_data")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    file_path = os.path.join(data_dir, file_name)
    if not os.path.isfile(file_path):
        urlretrieve(url, file_path)

    return file_path


# load data
def load_data():
    X, Y = adult()
    X = pd.get_dummies(X)
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    le = LabelEncoder()
    Y = le.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

    sensitive_attribute = 'Sex'
    A_train, A_test = X_train[sensitive_attribute], X_test[sensitive_attribute]

    X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    A_train, A_test = A_train.reset_index(drop=True), A_test.reset_index(drop=True)

    X_train_all, X_test_all = pd.DataFrame(X_train), pd.DataFrame(X_test)
    A_train_all, A_test_all = pd.Series(A_train), pd.Series(A_test)
    y_train_all, y_test_all = pd.Series(Y_train), pd.Series(Y_test)

    return X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all


def aggregate_phase_time(df):
    results_df = df.groupby(df.columns.drop(['phase', 'time']).tolist(), as_index=False, dropna=False).agg(
        {'time': 'sum'})
    return results_df


def get_combined_groupby(x, alpha=0.5):
    hybrid_res = x[x['model_name'].str.startswith('hybrid')]
    composed_metric = hybrid_res['train_violation'] * (1 - alpha) + hybrid_res['train_error'] * alpha
    combo_res = hybrid_res.loc[composed_metric.idxmin()]
    comb_df = x[x['model_name'] == 'combined']
    for col in np.setdiff1d(comb_df.columns, ['eps', 'frac', 'model_name', 'time', 'phase', 'random_seed', 'grid_frac','n']):
        comb_df[col] = combo_res[col]
    comb_df['alpha'] = alpha
    return comb_df


def add_combined_stats(df, alphas=[.05, .5, .95]):
    not_combined_df = df.loc[df['model_name'] != "combined"]
    cols_to_group = ['eps', 'frac', 'random_seed', 'grid_frac']
    if 'n' in df.columns:
        cols_to_group += ['n']
    combo_stat_list = []
    for alpha in alphas:
        turn_f = partial(get_combined_groupby, alpha=alpha)
        combined_stats = df.groupby(cols_to_group, as_index=False).apply(turn_f)
        combo_stat_list.append(combined_stats.copy())
    df = pd.concat(combo_stat_list + [not_combined_df]).drop_duplicates().reset_index(drop=True)
    return df


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


def get_last_results(base_dir):
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


def get_info():

    host_name = socket.gethostname()
    if "." in host_name:
        host_name = host_name.split(".")[-1]

    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return host_name, current_time_str