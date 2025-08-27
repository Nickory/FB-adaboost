from sklearn.model_selection import train_test_split
import Zafar.funcs_disp_mist as fdm
from load_adult import *
from load_compas import *
from load_hsls import *

"""
    Reference:https://github.com/haewon55/FairMIPForest/
"""


def main():
    seed_list = [i + 1 for i in range(20)]
    exp_id = 1

    assert exp_id in [i + 1 for i in range(9)], "Experiment does not exist."

    if exp_id == 1:
        # EXP1: Adult-gender-acc
        tau_list = [0.5, 1, 5, 10, 100]
        exp_data = 'adult'
        fair = 'acc'
        sens_attr = 'gender'

    elif exp_id == 2:
        # EXP2: Adult-gender-fpr
        tau_list = [0.01, 0.1, 1, 10, 100]
        exp_data = 'adult'
        fair = 'fpr'
        sens_attr = 'gender'

    elif exp_id == 3:
        # EXP3: Adult-gender-fnr
        tau_list = [5, 10, 20, 50, 100]
        exp_data = 'adult'
        fair = 'fnr'
        sens_attr = 'gender'

    elif exp_id == 4:
        # EXP4: Compas-race-fnr
        tau_list = [0.01, 0.1, 1, 10, 100]
        exp_data = 'compas'
        fair = 'fnr'
        sens_attr = 'race'

    elif exp_id == 5:
        # EXP5: Compas-race-acc
        tau_list = [0.01, 0.1, 1, 10, 100]
        exp_data = 'compas'
        fair = 'acc'
        sens_attr = 'race'

    elif exp_id == 6:
        # EXP6: Compas-race-fpr
        tau_list = [1, 5, 10, 100]
        exp_data = 'compas'
        fair = 'fpr'
        sens_attr = 'race'

    elif exp_id == 7:
        # EXP7: HSLS-race-fnr
        tau_list = [0.1, 1, 10, 100]
        exp_data = 'hsls'
        fair = 'fnr'
        sens_attr = 'racebin'

    elif exp_id == 8:
        # EXP8: HSLS-race-acc
        tau_list = [0.01, 0.1, 1, 10, 100]
        exp_data = 'hsls'
        fair = 'acc'
        sens_attr = 'racebin'

    else:
        # EXP9: HSLS-race-fpr
        tau_list = [0.01, 0.1, 1, 10, 100]
        exp_data = 'hsls'
        fair = 'fpr'
        sens_attr = 'racebin'

    tau_num = len(tau_list)
    seed_num = len(seed_list)

    train_acc_array = np.ones((tau_num, seed_num))
    train_fair_array = np.ones((tau_num, seed_num))
    test_acc_array = np.ones((tau_num, seed_num))
    test_fair_array = np.ones((tau_num, seed_num))
    test_fair_0_array = np.ones((tau_num, seed_num))
    test_fair_1_array = np.ones((tau_num, seed_num))

    # load and process data
    if exp_data == 'adult':
        df_train, df_test = load_adult()
        df = balance_data(df_train, 'income', 0)
        df = balance_data(df, 'gender', 1)
        df['income'].replace({0: -1}, inplace=True)
    elif exp_data == 'compas':
        X, y, x_control = load_compas_data()

        df = pd.DataFrame(X,
                          columns=['age_cat_25 - 45', 'age_cat_Greater than 45', 'age_cat_Less than 25', 'race', 'sex',
                                   'priors_count', 'c_charge_degree'])

        y = pd.Series(y, name="two_year_recid")

        df = pd.concat([df, y], axis=1)
        df = balance_data(df, 'race', 0)

    elif exp_data == 'hsls':
        df = load_hsls()
        df = balance_data(df, "racebin", 1)
        df['gradebin'].replace({0: -1}, inplace=True)

    if fair == 'acc':
        cons_type = 0
    elif fair == 'fpr':
        cons_type = 1
    elif fair == 'fnr':
        cons_type = 2
    mu = 1.2
    loss_function = "logreg"  # perform the experiments with logistic regression
    EPS = 1e-6

    for j in range(tau_num):
        tau = tau_list[j]
        for k in range(seed_num):
            seed = seed_list[k]
            df_train, df_test = train_test_split(df, test_size=0.3, random_state=seed)

            sensitive_attrs = [sens_attr]

            X_train = df_train.iloc[:, :-1]
            X_train['intercept'] = np.ones(len(X_train))
            x_control_train = dict({sens_attr: np.array([int(x) for x in X_train[sens_attr]]).astype(np.int64)})
            X_train = np.array(X_train.drop(columns=[sens_attr]))
            y_train = np.array(df_train.iloc[:, -1])

            X_test = df_test.iloc[:, :-1]
            X_test['intercept'] = np.ones(len(X_test))
            x_control_test = dict({sens_attr: np.array([int(x) for x in X_test[sens_attr]]).astype(np.int64)})
            X_test = np.array(X_test.drop(columns=[sens_attr]))
            y_test = np.array(df_test.iloc[:, -1])

            print(tau, seed)
            sensitive_attrs_to_cov_thresh = {sens_attr: {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}, 2: {0: 0,
                                                                                               1: 0}}}  # zero covariance threshold, means try to get the fairest solution
            cons_params = {"cons_type": cons_type,
                           "tau": tau,
                           "mu": mu,
                           "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

            w = fdm.train_model_disp_mist(X_train, y_train, x_control_train, loss_function, EPS, cons_params)
            train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test = fdm.get_clf_stats(
                w, X_train, y_train, x_control_train, X_test, y_test, x_control_test, sensitive_attrs)

            train_fr = np.abs(s_attr_to_fp_fn_train[sens_attr][0][fair] - s_attr_to_fp_fn_train[sens_attr][1][fair])
            test_fr = np.abs(s_attr_to_fp_fn_test[sens_attr][0][fair] - s_attr_to_fp_fn_test[sens_attr][1][fair])
            test_fr_0 = s_attr_to_fp_fn_test[sens_attr][0][fair]
            test_fr_1 = s_attr_to_fp_fn_test[sens_attr][1][fair]

            train_acc_array[j][k] = train_score
            train_fair_array[j][k] = train_fr
            test_acc_array[j][k] = test_score
            test_fair_array[j][k] = test_fr
            test_fair_0_array[j][k] = test_fr_0
            test_fair_1_array[j][k] = test_fr_1

            print('train_acc: {}'.format(train_score))
            print('test_acc: {}'.format(test_score))
            print('train_fr: {}'.format(train_fr))
            print('test_fr: {}'.format(test_fr))

    if exp_id == 7 or exp_id == 8 or exp_id == 9:
        sens_attr = 'race'

    np.save('./result/result_{}_{}/{}/train_acc_zafar_{}.npy'.format(exp_data, sens_attr, fair, mu), train_acc_array)
    np.save('./result/result_{}_{}/{}/train_fair_zafar_{}.npy'.format(exp_data, sens_attr, fair, mu), train_fair_array)
    np.save('./result/result_{}_{}/{}/test_acc_zafar_{}.npy'.format(exp_data, sens_attr, fair, mu), test_acc_array)
    np.save('./result/result_{}_{}/{}/test_fair_zafar_{}.npy'.format(exp_data, sens_attr, fair, mu), test_fair_array)
    np.save('./result/result_{}_{}/{}/test_fair_0_zafar_{}.npy'.format(exp_data, sens_attr, fair, mu), test_fair_0_array)
    np.save('./result/result_{}_{}/{}/test_fair_1_zafar_{}.npy'.format(exp_data, sens_attr, fair, mu), test_fair_1_array)

    return


if __name__ == "__main__":
    main()
