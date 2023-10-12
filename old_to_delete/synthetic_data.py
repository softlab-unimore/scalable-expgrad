from copy import deepcopy
from random import random, seed

from scipy.stats._multivariate import random_correlation_gen
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import numpy.random as rg

def get_miro_synthetic_data(num_data_points, num_features, random_seed, ratios=None, theta=.6):
    """
    Args:
        num_data_points: number of data points
        num_features: number of features to generate in addition to the sensitive attribute
        random_seed: random_seed
        ratios: dict containing the set of keys-values:
            group: set of values for the groups,
            group_prob: probability of each group value,
            y_prob: target (Y) probability related to each group value,
            switch_pos: probability to switch target (Y) when Y == 1 for each group value,
            switch_neg: probability to switch target (Y) when Y == 0 for each group value
            eg:{'group': np.arange(3),
                'group_prob': prob,
                'y_prob': [.7, .6, .65],
                'switch_pos':[.1,.2,.15],
                'switch_neg':[.2,.15,.2]}
        theta: probability of the features to be correlated with the target (Y) X = binomial(n=1, p=eps)

    Returns:

    """
    if ratios is None:
        prob = np.array([4, 3, 15])
        prob = prob / prob.sum()
        ratios = {'group': np.arange(3),
                  'group_prob': prob,
                  'y_prob': [.7, .6, .65],
                  'switch_pos':[.1,.2,.15],
                  'switch_neg':[.2,.15,.2]}
    else:
        keys = ['group','group_prob','y_prob','switch_pos','switch_neg']
        missing_keys = np.setdiff1d(keys, list(ratios.keys()))
        assert len(missing_keys) == 0, f'Missing {missing_keys} in ratios dict'
        ngroups = len(ratios['group'])
        for key, val in ratios.items():
            assert len(val) == ngroups, f'{key} has a different size wrt number of groups'
    rng = np.random.default_rng(random_seed)
    sensitive_attribute = rng.choice(a=ratios['group'], size=num_data_points, replace=True, p=ratios['group_prob'],
                                     axis=0, shuffle=True)
    Y = np.zeros_like(sensitive_attribute, dtype=int)
    for group, y_prob, switch_pos, switch_neg in zip(ratios['group'], ratios['y_prob'],
                                 ratios['switch_pos'],
                                 ratios['switch_neg']):
        group_mask = sensitive_attribute == group
        turn_Y = rng.binomial(n=1, p=y_prob, size=group_mask.sum())
        pos_mask = deepcopy(turn_Y == 1)
        for t_mask, t_switch_prob in [[pos_mask, switch_pos],
                                     [~pos_mask, switch_neg]]:
            t_values = turn_Y[t_mask]
            to_switch = rng.binomial(n=1, p=t_switch_prob, size=t_mask.sum()) == 1
            t_values[to_switch] = 1 - t_values[to_switch]
            turn_Y[t_mask] = t_values

        Y[group_mask] = turn_Y

    equal_y = rng.binomial(n=1, p=theta, size=(num_data_points, num_features))
    to_switch_mask = equal_y == 0
    X = Y.reshape(-1,1).repeat(num_features, axis=1)
    X[to_switch_mask] = 1 - X[to_switch_mask]
    X = np.concatenate([sensitive_attribute.reshape(-1,1), X], axis=1)
    X = pd.DataFrame(X, columns=['sensitive_attr']+ list(range(num_features)))
    return X, pd.Series(Y), pd.Series(sensitive_attribute)


def get_synthetic_data(num_data_pts, num_features, type_ratio, t0_ratio, t1_ratio, random_seed):
    """
    Generate synthetic data

    :param num_data_pts: number of datapoints
    :param num_features: number of features ; min=3
    :param type_ratio: ratio of sensitive group (e.g.: male:female)
    :param t0_ratio: ratio of positive labels in group 0 (e.g.: hired_females:unhired_females)
    :param t1_ratio: ratio of positive labels in group 1 (e.g.: hired_males:unhired_males)
    :param random_seed: random seed number
    :return: [X,Y,T]
    """
    random_state = int(random_seed * 99) + 1  # why?
    np.random.seed(random_seed)
    seed(random_state)
    sensitive_attribute = pd.Series(np.random.default_rng(random_state).binomial(1, type_ratio, int(num_data_pts)))
    Y = pd.Series([0] * num_data_pts)

    for group, turn_y_ratio in [(0, t0_ratio), (1, t1_ratio)]:
        mask = sensitive_attribute == group
        num_elements = mask.value_counts()[True]
        Y[mask] = np.random.default_rng(random_state).binomial(1, turn_y_ratio, num_elements)

    random_values = np.array([random() for i in range(num_data_pts)])
    X = Y * random_values
    X1 = pd.DataFrame(np.random.rand(int(num_data_pts), num_features - 2))
    X.name = num_features - 2
    Y.name = num_features - 1
    sensitive_attribute.name = num_features
    X = pd.concat([X1, X], axis=1)
    return pd.concat([X, Y, sensitive_attribute], axis=1)


def get_synthetic_data_old(num_data_pts, num_features, type_ratio, t0_ratio, t1_ratio, random_seed):
    """
    Generate synthetic data

    :param num_data_pts: number of datapoints
    :param num_features: number of features ; min=3
    :param type_ratio: ratio of sensitive group (e.g.: male:female)
    :param t0_ratio: ratio of positive labels in group 0 (e.g.: hired_females:unhired_females)
    :param t1_ratio: ratio of positive labels in group 1 (e.g.: hired_males:unhired_males)
    :param random_seed: random seed number
    :return: [X,Y,T]
    """
    random_state = int(random_seed * 99) + 1  # why?
    np.random.seed(random_seed)
    seed(random_state)
    # sensitive attribute - 0/1
    T = np.random.default_rng(random_state).binomial(1, type_ratio, int(num_data_pts))
    A = np.zeros(T.shape)

    Y = np.zeros(T.shape)
    X = np.zeros(T.shape)

    g0_X = A[T.astype(str) == '0']
    T0_Y = np.random.default_rng(random_state).binomial(1, t0_ratio, g0_X.shape)

    g1_X = A[T.astype(str) == '1']
    T1_Y = np.random.default_rng(random_state).binomial(1, t1_ratio, g1_X.shape)

    j = 0  # for 0
    k = 0  # for 1
    for i in range(T.shape[0]):
        if T[i] == 0:
            # get from T0_Y
            Y[i] = T0_Y[j]
            X[i] = Y[i] * random()
            j += 1
        elif T[i] == 1:
            # get from T1_Y
            Y[i] = T1_Y[k]
            X[i] = Y[i] * random()
            k += 1

    T = pd.Series(T)
    X1 = np.random.rand(int(num_data_pts), num_features - 2)
    X = pd.concat([pd.DataFrame(X), pd.DataFrame(X1), T], axis=1)
    Y = pd.Series(Y)
    return pd.concat([X, Y, T], axis=1)


# split data points
def data_split(All, test_ratio):
    """
    data split for above synthetic data set

    :param All: [X,Y,A]
    :param test_ratio: test ratio
    :return: X_train, Y_train, A_train, X_test, Y_test, A_test
    """
    # We know that All = X, Y, T
    all_train, all_test, Y_train, Y_test = train_test_split(All, All.iloc[:, -2], test_size=test_ratio, random_state=42)
    # test dataset
    A_test = all_test.iloc[:, -1]
    X_test = all_test.iloc[:, :-2]

    # train dataset
    A_train = all_train.iloc[:, -1]
    X_train = all_train.iloc[:, :-2]
    return pd.DataFrame(X_train), pd.Series(Y_train), pd.Series(A_train), \
           pd.DataFrame(X_test), pd.Series(Y_test), pd.Series(A_test)
