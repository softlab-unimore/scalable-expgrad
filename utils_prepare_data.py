from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, \
    load_preproc_data_compas, load_preproc_data_german
import gc
import requests
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
adult_github_data_url = "https://github.com/slundberg/shap/raw/master/data/"
german_credit_kaggle_url = 'https://www.kaggle.com/datasets/uciml/german-credit/download?datasetVersionNumber=1'
compas_credit_github_maliha_url = 'https://github.com/maliha93/Fairness-Analysis-Code/raw/master/dataset/compas.csv'
compas_raw_data_github_url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'


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


def split_label_sensitive_attr(df: pd.DataFrame, label_name, sensitive_name):
    Y = df[label_name].copy()
    A = df[sensitive_name].copy()
    return df, Y, A


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
        cache(adult_github_data_url + "adult.data"),
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


def check_download_dataset(dataset_name='compas'):
    if dataset_name == 'compas':
        compas_path = '/home/fairlearn/anaconda3/lib/python3.9/site-packages/aif360/data/raw/compas'
        compas_raw_path = compas_path + '/compas-scores-two-years.csv'
        if not os.path.exists(compas_raw_path):
            df = pd.read_csv(compas_raw_data_github_url)
            os.makedirs(compas_path)
            df.to_csv(compas_raw_path, index=False)
    if dataset_name == 'german':
        base_path = '/home/fairlearn/anaconda3/lib/python3.9/site-packages/aif360/data/raw/german/'
        base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/'
        file_names = ['german.data','german.doc']
        for file_name in file_names:
            turn_path = os.path.join(base_path, file_name)
            if not os.path.exists(turn_path):
                response = requests.get(base_url + file_name)
                open(turn_path, "wb").write(response.content)


def convert_to_df_aif360(dataset, dataset_name):
    X = pd.DataFrame(dataset.features, columns=dataset.feature_names)
    y = pd.Series(dataset.labels.flatten(), name=dataset.label_names[0])
    A = pd.Series(dataset.protected_attributes.flatten(), name=dataset.protected_attribute_names[0])
    if dataset_name == 'german':
        y[y==2] = 0
    return X, y, A


def load_dataset_aif360(dataset_name='compas', split=True, train_test_seed=0):
    if dataset_name == 'compas':
        protected = 'race'
        load_function = load_preproc_data_compas
    elif dataset_name == 'german':
        protected = 'sex'
        load_function = load_preproc_data_german
    else:
        raise(Exception(f'dataset_name {dataset_name} not allowed.'))
    check_download_dataset(dataset_name)
    dataset_orig = load_function(protected_attributes=[protected])
    if split:
        if train_test_seed is 0:
            dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
        else:
            dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed=train_test_seed)
        ret_value = convert_to_df_aif360(dataset_orig_train, dataset_name)
        ret_value += convert_to_df_aif360(dataset_orig_test, dataset_name)
    else:
        ret_value = convert_to_df_aif360(dataset_orig, dataset_name)
    return ret_value



def load_transform_Adult(sensitive_attribute='Sex', test_size=0.3, random_state=42):
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


def load_transform_ACS(loader_method, states=None, return_acs_data=False):
    data_source = ACSDataSource(survey_year=2018, horizon='1-Year', survey='person', root_dir='cached_data')
    definition_df = data_source.get_definitions(download=True)
    categories = generate_categories(features=loader_method.features, definition_df=definition_df)
    acs_data = data_source.get_data(download=True,
                                    states=states)  # TODO # with density 1  random_seed=0 do nothing | join_household=False ???

    df, label, group = loader_method.df_to_pandas(acs_data, categories=categories)
    # df, label, group = fix_nan(df, label, group, mode=fillna_mode)

    categorical_cols = list(categories.keys())
    # See here for data documentation of cols https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/
    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict00_02.pdf
    df = pd.get_dummies(df, dtype=np.uint8, columns=categorical_cols)
    numerical_cols = np.setdiff1d(loader_method.features, categorical_cols)
    df[numerical_cols] = StandardScaler().fit_transform(df[numerical_cols])
    # choice of the model todo add possibility to chose preprocessing based on the model
    # df[df.columns] = StandardScaler().fit_transform(df)
    print(f'Loaded data memory used by df: {df.memory_usage().sum() / (2 ** (10 * 3)):.3f} GB')
    ret_value = df, label.iloc[:, 0].astype(int), group.iloc[:, 0]
    if return_acs_data:
        ret_value += tuple([acs_data])
    else:
        del acs_data

    return ret_value


def fix_nan(X: pd.DataFrame, y, A, mode='mean'):
    if mode == 'mean':
        X.fillna(X.mean())
    if mode == 'remove':
        notna_mask = X.notna().all(1)
        X, y, A = X[notna_mask], y[notna_mask], A[notna_mask]

    return X, y, A


def load_split_data(sensitive_attribute='Sex', test_size=0.3, random_state=42):
    X, Y, A = load_transform_Adult(sensitive_attribute, test_size, random_state)

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
