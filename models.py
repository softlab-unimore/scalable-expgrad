from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV


def get_model_parameter_grid(base_model_code=None):
    if base_model_code is None or base_model_code == 'lr':
        # Unmitigated LogRes
        return {'solver': [  # 'newton-cg',
            'lbfgs',
            'liblinear'
        ],
            'penalty': ['l2'],
            'C': [0.01, 0.005, 0.001],
            # [10, 1.0, 0.1, 0.05, 0.01],
            # max-iter': 100,
        }
    elif base_model_code == 'gbm':
        return dict(n_estimators=[10, 100, 500],
                    learning_rate=[0.001, 0.01, 0.1],
                    subsample=[0.5, 0.7, 1.0],
                    max_depth=[3, 7, 9])
    elif base_model_code == 'lgbm':
        return dict(
            l2_regularization=[10, 0.1, 0.01],
            learning_rate=[0.001, 0.01, 0.1],
            max_depth=[3, 7, 9])
    else:
        assert False, f'available model codes are:{["lr", "gbm", "lgbm"]}'


def get_base_model(base_model_code, random_seed=0):
    if base_model_code is None or base_model_code == 'lr':
        # Unmitigated LogRes
        model = LogisticRegression(solver='liblinear', fit_intercept=True, random_state=random_seed)
    elif base_model_code == 'gbm':
        model = GradientBoostingClassifier(random_state=random_seed)
    elif base_model_code == 'lgbm':
        model = HistGradientBoostingClassifier(random_state=random_seed)
    else:
        assert False, f'available model codes are:{["lr", "gbm", "lgbm"]}'
    return model


def finetune_model(base_model_code, X, y, random_seed=0):
    base_model = get_base_model(base_model_code=base_model_code, random_seed=random_seed)
    parameters = get_model_parameter_grid(base_model_code=base_model_code)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_seed)
    clf = GridSearchCV(base_model, parameters, cv=cv, n_jobs=1, scoring=['f1', 'accuracy'], refit='f1')
    clf.fit(X, y)
    return clf

