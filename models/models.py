import tensorflow as tf
from functools import partial
from warnings import warn
from aif360.algorithms.inprocessing import AdversarialDebiasing, MetaFairClassifier, GerryFairClassifier
from aif360.algorithms.preprocessing import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import utils_prepare_data
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV

from models import wrappers


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




def get_model(method_str, base_model, constrain_name, eps, random_state, datasets):
    params = method_str, base_model, constrain_name, eps, random_state, datasets
    methods_name_dict = {'hybrids': 'hybrids',
                         'unmitigated': 'unmitigated',
                         'fairlearn': 'fairlearn',
                         'ThresholdOptimizer': 'ThresholdOptimizer',
                         'MetaFairClassifier': 'MetaFairClassifier',
                         'AdversarialDebiasing': 'AdversarialDebiasing',
                         'Kearns': 'Kearns',
                         'Calmon': 'Calmon',
                         'ZafarDI':'ZafarDI'
                         }
    if method_str == methods_name_dict['ThresholdOptimizer']:
        model = ThresholdOptimizer(
            estimator=base_model,
            constraints=constrain_name,
            objective="accuracy_score",
            prefit=False,
            predict_method='predict_proba', )
        model.predict = partial(model.predict, random_state=random_state)
        if eps is not None:
            warn(f"eps has no effect with {method_str} methos")
    elif method_str == methods_name_dict['AdversarialDebiasing']:

        privileged_groups, unprivileged_groups = utils_prepare_data.find_privileged_unprivileged(**datasets)
        sess = tf.Session()
        # Learn parameters with debias set to True
        model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                              unprivileged_groups=unprivileged_groups,
                                              scope_name='debiased_classifier',
                                              debias=True,
                                              sess=sess)
    elif method_str == methods_name_dict['Kearns']:
        model = wrappers.Kearns(*params)
    elif method_str == methods_name_dict['Calmon']:
        model = wrappers.CalmonWrapper(*params)
    elif method_str == methods_name_dict['ZafarDI']:
        model = wrappers.ZafarDI(*params)
    else:
        raise ValueError(
            f'the method specified ({method_str}) is not allowed. Valid options are {methods_name_dict.values()}')
    return model