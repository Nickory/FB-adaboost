import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_utils import *

"""
    Reference:https://github.com/haewon55/FairMIPForest/
"""


def load_adult(scaler=True, race=False):
    """
    :param race: if sensitive attribute is 'race'
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    """
    pwd = './dataset/Adult/'
    data = pd.read_csv(
        pwd + 'adult.data',
        names=[
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"]
    )
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(
        pwd + 'adult.test',
        names=[
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"]
    )
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = data["workclass"][4].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    # # Here we apply discretisation on column marital_status
    # data.replace([' Divorced', ' Married-AF-spouse',
    #               ' Married-civ-spouse', ' Married-spouse-absent',
    #               ' Never-married', ' Separated', ' Widowed'],
    #              ['not married', 'married', 'married', 'married',
    #               'not married', 'not married', 'not married'], inplace=True)
    if race:
        data["race"].replace([' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other'],
                             ['Non-white', 'Non-white', 'Non-white', 'Non-white'], inplace=True)
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c

    data, data_test = data.drop(columns=['fnlwgt']), data_test.drop(columns=['fnlwgt'])

    datamat = data.values
    datamat = datamat[:, :-1]

    if scaler:
        scaler = MinMaxScaler()
        scaler.fit(datamat)
        data.iloc[:, :-1] = scaler.fit_transform(datamat)

    return data.iloc[:len_train], data.iloc[len_train:]


def load_and_balance_adult(balance=True, sens_attr='gender'):
    # load data
    df_train, df_test = load_adult(race=(sens_attr == 'race'))

    if balance:
        # balance data
        df = balance_data(df_train, 'income', 0)
        if sens_attr == 'gender':
            df = balance_data(df, sens_attr, 1)
        elif sens_attr == 'race':
            df = balance_data(df_train, sens_attr, 0)

        # save
        df_to_pickle(df, sens_attr, filename='./pkl_data/adult_balance_{}.pkl'.format(sens_attr))
    else:
        df_to_pickle(df_train, sens_attr, filename='./pkl_data/adult_train_{}.pkl'.format(sens_attr))
        df_to_pickle(df_test, sens_attr, filename='./pkl_data/adult_test_{}.pkl'.format(sens_attr))

    return


if __name__ == "__main__":
    load_and_balance_adult(balance=True, sens_attr='gender')
