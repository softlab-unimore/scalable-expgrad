#!/usr/bin/env python
# coding: utf-8

import os

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
