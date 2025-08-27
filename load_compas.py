from data_utils import *
import pandas as pd
from collections import defaultdict
from sklearn import preprocessing
from random import seed, shuffle

"""
    Reference:https://github.com/haewon55/FairMIPForest/
"""


def load_compas_data():
    SEED = 1234
    seed(SEED)
    np.random.seed(SEED)

    FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count",
                               "c_charge_degree"]  # features to be used for classification
    CONT_VARIABLES = [
        "priors_count"]  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "two_year_recid"  # the decision variable
    SENSITIVE_ATTRS = ["race"]

    # load the data and get some stats
    df = pd.read_csv('./dataset/Compas/compas-scores-two-years.csv')
    df = df.dropna(subset=["days_b_screening_arrest"])  # dropping missing vals

    # convert to np array
    data = df.to_dict('list')
    for k in list(data.keys()):
        data[k] = np.array(data[k])

    """ Filtering the data """

    # These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
    idx = np.logical_and(data["days_b_screening_arrest"] <= 30, data["days_b_screening_arrest"] >= -30)

    # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O")  # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # we will only consider blacks and whites for this analysis
    idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))

    # select the examples that satisfy this criteria
    for k in list(data.keys()):
        data[k] = data[k][idx]

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    y[y == 0] = -1

    X = np.array([]).reshape(len(y),
                             0)  # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)

    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals)  # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col

        else:  # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)

        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals

        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES:  # continuous feature, just append the name
            feature_names.append(attr)
        else:  # categorical features
            if vals.shape[1] == 1:  # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_:  # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))

    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in list(x_control.keys()):
        assert (x_control[k].shape[1] == 1)  # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()

    """permute the date randomly"""
    perm = list(range(0, X.shape[0]))
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in list(x_control.keys()):
        x_control[k] = x_control[k][perm]

    return X, y, x_control


def load_and_balance_compas(balance=True):
    X, y, x_control = load_compas_data()

    df = pd.DataFrame(X, columns=['age_cat_25 - 45', 'age_cat_Greater than 45', 'age_cat_Less than 25', 'race', 'sex',
                                  'priors_count', 'c_charge_degree'])

    y = pd.Series(y, name="two_year_recid")

    df = pd.concat([df, y], axis=1)

    sens_attr = 'race'

    # Balancing the Data
    if balance:
        df = balance_data(df, 'race', 0)
        df_to_pickle(df, sens_attr, filename='./pkl_data/compas_balance_{}.pkl'.format(sens_attr))
    else:
        df_to_pickle(df, sens_attr, filename='./pkl_data/compas_{}.pkl'.format(sens_attr))

    return


if __name__ == "__main__":
    load_and_balance_compas(balance=True)
