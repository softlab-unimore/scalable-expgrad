from random import random, seed
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# This code is meant to generate synthetic data
# to run the experiments.


def get_data(num_data_pts, num_features, type_ratio, t0_ratio, t1_ratio, random_seed):
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
    random_state = int(random_seed * 99) + 1
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

    :param All: [X,Y,T]
    :param test_ratio: test ratio
    :return: X_train, Y_train, A_train, X_test, Y_test, A_test
    """
    # We know that All = X, Y, T
    all_train, all_test, Y_train, Y_test = train_test_split(All, All.iloc[:, -2], test_size=test_ratio, random_state=42)
    # test dataset
    T_test = all_test.iloc[:, -1]
    X_test = all_test.iloc[:, :-2]

    # train dataset
    T_train = all_train.iloc[:, -1]
    X_train = all_train.iloc[:, :-2]
    return pd.DataFrame(X_train), pd.Series(Y_train), pd.Series(T_train), \
           pd.DataFrame(X_test), pd.Series(Y_test), pd.Series(T_test)
