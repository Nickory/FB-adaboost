from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np


def binary_score(y_true, y_hat, metric='acc'):
    c = confusion_matrix(y_true, y_hat)

    if metric == 'fnr':
        return c[1][0] / (c[1][0] + c[1][1])
    elif metric == 'fpr':
        return c[0][1] / (c[0][0] + c[0][1])
    elif metric == 'acc':
        return (c[0][0] + c[1][1]) / len(y_true)


def eval_binary_metrics(pre_model, X_eval, y_eval, sens_attr):
    # For binary labels #
    metrics = {}
    if len(X_eval) != len(sens_attr):
        raise ValueError

    for g_sel in [0, 1]:
        X_eval_g = X_eval[sens_attr == g_sel]
        y_eval_g = y_eval[sens_attr == g_sel]
        y_pre_g = pre_model.predict(X_eval_g)

        c = confusion_matrix(y_eval_g, y_pre_g)
        metrics[g_sel] = {}
        metrics[g_sel]['Size'] = c[0][0] + c[0][1] + c[1][0] + c[1][1]
        metrics[g_sel]['True Parity'] = (c[1][0] + c[1][1]) / (c[0][0] + c[0][1] + c[1][0] + c[1][1])
        metrics[g_sel]['Parity'] = (c[0][1] + c[1][1]) / (c[0][0] + c[0][1] + c[1][0] + c[1][1])
        metrics[g_sel]['FPR'] = c[0][1] / (c[0][0] + c[0][1])
        metrics[g_sel]['FNR'] = c[1][0] / (c[1][0] + c[1][1])
        metrics[g_sel]['PPV'] = c[1][1] / (c[0][1] + c[1][1])
        metrics[g_sel]['NPV'] = c[0][0] / (c[0][0] + c[1][0])
        metrics[g_sel]['Accuracy'] = accuracy_score(y_eval_g, y_pre_g)

    return metrics


def fair_binary_score(y_true, y_hat, S, metric='acc'):
    return np.abs(binary_score(y_true[S == 0], y_hat[S == 0], metric) -
                  binary_score(y_true[S == 1], y_hat[S == 1], metric))
