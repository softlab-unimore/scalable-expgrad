import gc

from folktables import ACSDataSource, generate_categories
import os
import numpy as np
import pandas as pd
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


def load_data(sensitive_attribute='Sex', test_size=0.3, random_state=42):
    # https://archive.ics.uci.edu/ml/datasets/adult
    features = ['Age', 'Workclass', 'Education-Num', 'Marital Status',
                'Occupation', 'Relationship', 'Race', 'Sex',
                'Capital Gain', 'Capital Loss', 'Hours per week', 'Country']
    categorical_cols = ['Workclass',  # 'Education-Num',
                        'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    numerical_cols = np.setdiff1d(features, categorical_cols)

    X, Y = adult()
    Y = pd.Series(LabelEncoder().fit_transform(Y))
    A = X[sensitive_attribute].copy()
    X_transformed = pd.get_dummies(X, dtype=int, columns=categorical_cols)
    X_transformed[numerical_cols] = StandardScaler().fit_transform(X[numerical_cols])
    return X_transformed, Y, A


def load_transform_ACS(loader_method, states=None, fillna_mode='mean'):
    data_source = ACSDataSource(survey_year=2018, horizon='1-Year', survey='person', root_dir='cached_data')
    definition_df = data_source.get_definitions(download=True)
    categories = generate_categories(features=loader_method.features, definition_df=definition_df)
    acs_data = data_source.get_data(
        download=True, states=states)  # TODO # with density 1  random_seed=0 do nothing | join_household=False ???
    df, label, group = loader_method.df_to_pandas(acs_data, categories=categories)
    df, label, group = fix_nan(df, label, group, mode=fillna_mode)
    del acs_data
    categorical_cols = list(categories.keys())
    # See here for data documentation of cols https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/
    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict00_02.pdf
    df = pd.get_dummies(df, dtype=np.uint8, columns=categorical_cols)
    numerical_cols = np.setdiff1d(loader_method.features, categorical_cols)
    df[numerical_cols] = StandardScaler().fit_transform(df[numerical_cols]) # choice of the model todo add possibility to chose preprocessing based on the model
    # df[df.columns] = StandardScaler().fit_transform(df)
    print(f'Loaded data memory used by df: {df.memory_usage().sum()/(2**(10*3)):.3f} GB')
    return df, label.iloc[:, 0].astype(int), group.iloc[:, 0]

def fix_nan(X:pd.DataFrame, y, A, mode='mean'):
    if mode=='mean':
        X.fillna(X.mean())
    if mode=='remove':
        notna_mask = X.notna().all(1)
        X, y, A = X[notna_mask], y[notna_mask], A[notna_mask]

    return X, y, A


def load_split_data(sensitive_attribute='Sex', test_size=0.3, random_state=42):
    X, Y, A = load_data(sensitive_attribute, test_size, random_state)

    train_index, test_index = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=random_state)
    results = []
    for turn_index in [train_index, test_index]:
        for turn_df in [X, Y, A]:
            results.append(turn_df.iloc[turn_index])
    return results


def get_data_from_expgrad(expgrad):
    res_dict = {}
    for key in ['best_iter_', 'best_gap_',
                # 'weights_', '_hs',  'predictors_', 'lambda_vecs_',
                'last_iter_', 'n_oracle_calls_',
                'n_oracle_calls_dummy_returned_', 'oracle_execution_times_', ]:
        res_dict[key] = getattr(expgrad, key)
    return res_dict
