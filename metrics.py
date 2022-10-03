import numpy as np
from fairlearn.reductions import DemographicParity, ErrorRate
from sklearn.metrics import confusion_matrix


def getViolation(X, Y, A, predict_method):
    disparity_moment = DemographicParity()
    disparity_moment.load_data(X, Y, sensitive_features=A)
    return disparity_moment.gamma(predict_method).max()


def getError(X, Y, A, predict_method):
    error = ErrorRate()
    error.load_data(X, Y, sensitive_features=A)
    return error.gamma(predict_method)[0]


def getDI(X, Y, S, predict_method):
    return di(X, Y, S, predict_method(X))

def di(X, Y, S, y_pred):
    s_values = np.unique(S)
    s_values.sort()
    group_0_mask = S == s_values[0]
    group_1_mask = S == s_values[1]
    PrY1_S0 = np.sum(group_0_mask & y_pred == 1) / np.sum(group_0_mask)
    PrY1_S1 = np.sum(group_1_mask & y_pred == 1) / np.sum(group_1_mask)
    disparate_impact = PrY1_S0 / PrY1_S1
    return disparate_impact


def trueRateBalance(X, Y, S, y_pred):
    s_values = np.unique(S)
    s_values.sort()
    mask_0 = S == s_values[0]
    mask_1 = S == s_values[1]
    results = {}
    for turn_mask, group in zip([mask_1, mask_0], [1, 0]):
        TN, FP, FN, TP = confusion_matrix(Y[turn_mask], y_pred[turn_mask] >= 0.5).ravel()
        results[f'TPR_{group}'] = TP / (TP + FN)
        results[f'TNR_{group}'] = TN / (FP + TN)
    return results


def TPRB(X, Y, S, y_pred):
    rates_dict = trueRateBalance(X, Y, S, y_pred)
    return np.abs(rates_dict['TPR_1'] - rates_dict['TPR_0'])

def getTPRB(X, Y, S, predict_method):
    return np.abs(TPRB(X, Y, S, predict_method(X)))

def TNRB(X, Y, S, y_pred):
    rates_dict = trueRateBalance(X, Y, S, y_pred)
    return rates_dict['TNR_1'] - rates_dict['TNR_0']  # TNRB

def getTNRB(X, Y, S, predict_method):
    return TNRB(X, Y, S, predict_method(X))


metrics_dict = {'error': getError,
                'violation': getViolation,
                'di': getDI,
                'TPRB': getTPRB,
                'TNRB': getTNRB
                }