import numpy as np
import pickle

"""
    Reference:https://github.com/haewon55/FairMIPForest/
"""


def balance_data(df, attr, major=1):
    sample_size = len(df[df[attr] == major]) - len(df[df[attr] == (1 - major)])
    if sample_size < 0:
        raise ValueError

    np.random.seed(0)
    drop_idx = np.random.choice(df[df[attr] == major].index, sample_size, replace=False)
    return df.drop(drop_idx)


def df_to_pickle(df_ms, sens_attr, filename='df_ms.pkl'):
    S = df_ms[sens_attr]
    if sens_attr == 'race':
        S = 1 - S
    X = df_ms.iloc[:, :-1].copy()
    X = X.drop(columns=[sens_attr])
    y = df_ms.iloc[:, -1].copy()
    y[y < 0] = 0
    X, y, S = np.array(X), np.array(y), np.array(S)
    data = {'X': X, 'y': y, 'S': S}

    with open(filename, 'wb') as handle:
        pickle.dump(data, handle)
