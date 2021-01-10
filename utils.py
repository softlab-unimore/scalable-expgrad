#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd

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
    sensitive_attribute = 'Sex'

    A = X[sensitive_attribute]
    X = pd.get_dummies(X)

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    le = LabelEncoder()
    Y = le.fit_transform(Y)

    X = X.reset_index(drop=True)
    A = A.reset_index(drop=True)

    X_train_all = pd.DataFrame(X)
    A_train_all = pd.Series(A)
    y_train_all = pd.Series(Y)

    return X_train_all, y_train_all, A_train_all


