import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    base_type = 'CART'
    mu = 1.2
    exp_id = 7  # 1 -- 9

    if exp_id == 1:
        # EXP1: Adult-gender-acc
        exp_data = 'adult'
        sens_attr = 'gender'
        fair = 'acc'
        T = 30
        turn = False
        FAB_id = 6
        Agarwal_id = 0
        Agarwal_grid_id = 3
        Zafar_id = 4
        sens = "Gender"
        fair_name = "Accuracy"
        sens_0 = "Female"
        sens_1 = "Male"

    elif exp_id == 2:
        # EXP2: Adult-gender-fpr
        exp_data = 'adult'
        sens_attr = 'gender'
        fair = 'fpr'
        T = 30
        turn = False
        FAB_id = 5
        Agarwal_id = 0
        Agarwal_grid_id = 3
        Zafar_id = 0
        sens = "Gender"
        fair_name = "FPR"
        sens_0 = "Female"
        sens_1 = "Male"

    elif exp_id == 3:
        # EXP3: Adult-gender-fnr
        exp_data = 'adult'
        sens_attr = 'gender'
        fair = 'fnr'
        T = 30
        turn = True
        FAB_id = 5
        Agarwal_id = 0
        Agarwal_grid_id = 3
        Zafar_id = 3
        sens = "Gender"
        fair_name = "FNR"
        sens_0 = "Female"
        sens_1 = "Male"

    elif exp_id == 4:
        # EXP4: Compas-race-fnr
        exp_data = 'compas'
        sens_attr = 'race'
        fair = 'fnr'
        T = 30
        turn = True
        FAB_id = 5
        Agarwal_id = 0
        Agarwal_grid_id = 4
        Zafar_id = 0
        sens = "Race"
        fair_name = "FNR"
        sens_0 = "Black"
        sens_1 = "White"

    elif exp_id == 5:
        # EXP5: Compas-race-acc
        exp_data = 'compas'
        sens_attr = 'race'
        fair = 'acc'
        T = 15
        turn = True
        FAB_id = 3
        Agarwal_id = 0
        Agarwal_grid_id = 0
        Zafar_id = 3
        sens = "Race"
        fair_name = "Accuracy"
        sens_0 = "Black"
        sens_1 = "White"

    elif exp_id == 6:
        # EXP6: Compas-race-fpr
        exp_data = 'compas'
        sens_attr = 'race'
        fair = 'fpr'
        T = 20
        turn = False
        FAB_id = 5
        Agarwal_id = 0
        Agarwal_grid_id = 4
        Zafar_id = 0
        sens = "Race"
        fair_name = "FPR"
        sens_0 = "Black"
        sens_1 = "White"

    elif exp_id == 7:
        # EXP7: HSLS-race-fnr
        exp_data = 'hsls'
        sens_attr = 'race'
        fair = 'fnr'
        T = 20
        turn = True
        FAB_id = 5
        Agarwal_id = 0
        Agarwal_grid_id = 3
        Zafar_id = 0
        sens = "Race"
        fair_name = "FNR"
        sens_0 = "URM"
        sens_1 = "White/Asian"

    elif exp_id == 8:
        # EXP8: HSLS-race-acc
        exp_data = 'hsls'
        sens_attr = 'race'
        fair = 'acc'
        T = 20
        turn = True
        FAB_id = 2
        Agarwal_id = 0
        Agarwal_grid_id = 3
        Zafar_id = 2
        sens = "Race"
        fair_name = "Accuracy"
        sens_0 = "URM"
        sens_1 = "White/Asian"

    elif exp_id == 9:
        # EXP8: HSLS-race-fpr
        exp_data = 'hsls'
        sens_attr = 'race'
        fair = 'fpr'
        T = 20
        turn = False
        FAB_id = 3
        Agarwal_id = 0
        Agarwal_grid_id = 3
        Zafar_id = 2
        sens = "Race"
        fair_name = "FPR"
        sens_0 = "URM"
        sens_1 = "White/Asian"

    else:
        print("Experiment does not exist.")
        return

    # load_result
    if turn:
        test_boost_fr_1_loss = np.load(
            './result/result_{}_{}/{}/test_fair_0_adaboost_{}_3_{}.npy'.format(exp_data, sens_attr, fair, base_type, T))
        test_boost_fr_0_loss = np.load(
            './result/result_{}_{}/{}/test_fair_1_adaboost_{}_3_{}.npy'.format(exp_data, sens_attr, fair, base_type, T))
    else:
        test_boost_fr_0_loss = np.load(
            './result/result_{}_{}/{}/test_fair_0_adaboost_{}_3_{}.npy'.format(exp_data, sens_attr, fair, base_type, T))
        test_boost_fr_1_loss = np.load(
            './result/result_{}_{}/{}/test_fair_1_adaboost_{}_3_{}.npy'.format(exp_data, sens_attr, fair, base_type, T))

    test_fr_0_hardt = np.load('./result/result_{}_{}/{}/test_fair_0_hardt.npy'.format(exp_data, sens_attr, fair))
    test_fr_1_hardt = np.load('./result/result_{}_{}/{}/test_fair_1_hardt.npy'.format(exp_data, sens_attr, fair))

    test_fr_0_agarwal = np.load('./result/result_{}_{}/{}/test_fair_0_agarwal.npy'.format(exp_data, sens_attr, fair))
    test_fr_1_agarwal = np.load('./result/result_{}_{}/{}/test_fair_1_agarwal.npy'.format(exp_data, sens_attr, fair))

    test_fr_0_agarwal_grid = np.load(
        './result/result_{}_{}/{}/test_fair_0_agarwal_grid.npy'.format(exp_data, sens_attr, fair))
    test_fr_1_agarwal_grid = np.load(
        './result/result_{}_{}/{}/test_fair_1_agarwal_grid.npy'.format(exp_data, sens_attr, fair))

    test_fr_0_zafar = np.load('./result/result_{}_{}/{}/test_fair_0_zafar_{}.npy'.format(exp_data, sens_attr, fair, mu))
    test_fr_1_zafar = np.load('./result/result_{}_{}/{}/test_fair_1_zafar_{}.npy'.format(exp_data, sens_attr, fair, mu))

    # AdaBoost
    n = test_boost_fr_1_loss.shape[1]
    dict_adaboost_0 = {'Method': ["AdaBoost"] * n, 'Fairness indicator ({})'.format(fair_name): test_boost_fr_0_loss[0],
                       sens: [sens_0] * n}
    df_adaboost_0 = pd.DataFrame(data=dict_adaboost_0)
    dict_adaboost_1 = {'Method': ["AdaBoost"] * n, 'Fairness indicator ({})'.format(fair_name): test_boost_fr_1_loss[0],
                       sens: [sens_1] * n}
    df_adaboost_1 = pd.DataFrame(data=dict_adaboost_1)
    df = pd.concat([df_adaboost_0, df_adaboost_1])

    # FAB
    dict_fab_0 = {'Method': ["FAB"] * n, 'Fairness indicator ({})'.format(fair_name): test_boost_fr_0_loss[FAB_id],
                  sens: [sens_0] * n}
    df_fab_0 = pd.DataFrame(data=dict_fab_0)
    dict_fab_1 = {'Method': ["FAB"] * n, 'Fairness indicator ({})'.format(fair_name): test_boost_fr_1_loss[FAB_id],
                  sens: [sens_1] * n}
    df_fab_1 = pd.DataFrame(data=dict_fab_1)
    df = pd.concat([df, df_fab_0])
    df = pd.concat([df, df_fab_1])

    # Hardt
    dict_hardt_0 = {'Method': ["Hardt"] * n, 'Fairness indicator ({})'.format(fair_name): test_fr_0_hardt,
                    sens: [sens_0] * n}
    df_hardt_0 = pd.DataFrame(data=dict_hardt_0)
    dict_hardt_1 = {'Method': ["Hardt"] * n, 'Fairness indicator ({})'.format(fair_name): test_fr_1_hardt,
                    sens: [sens_1] * n}
    df_hardt_1 = pd.DataFrame(data=dict_hardt_1)
    df = pd.concat([df, df_hardt_0])
    df = pd.concat([df, df_hardt_1])

    # Agarwal-Exp
    dict_agarwal_0 = {'Method': ["Agarwal-Exp"] * n,
                      'Fairness indicator ({})'.format(fair_name): test_fr_0_agarwal[Agarwal_id],
                      sens: [sens_0] * n}
    df_agarwal_0 = pd.DataFrame(data=dict_agarwal_0)
    dict_agarwal_1 = {'Method': ["Agarwal-Exp"] * n,
                      'Fairness indicator ({})'.format(fair_name): test_fr_1_agarwal[Agarwal_id],
                      sens: [sens_1] * n}
    df_agarwal_1 = pd.DataFrame(data=dict_agarwal_1)
    df = pd.concat([df, df_agarwal_0])
    df = pd.concat([df, df_agarwal_1])

    # Agarwal-Grid
    dict_agarwal_grid_0 = {'Method': ["Agarwal-Grid"] * n,
                           'Fairness indicator ({})'.format(fair_name): test_fr_0_agarwal_grid[Agarwal_grid_id],
                           sens: [sens_0] * n}
    df_agarwal_grid_0 = pd.DataFrame(data=dict_agarwal_grid_0)
    dict_agarwal_grid_1 = {'Method': ["Agarwal-Grid"] * n,
                           'Fairness indicator ({})'.format(fair_name): test_fr_1_agarwal_grid[Agarwal_grid_id],
                           sens: [sens_1] * n}
    df_agarwal_grid_1 = pd.DataFrame(data=dict_agarwal_grid_1)
    df = pd.concat([df, df_agarwal_grid_0])
    df = pd.concat([df, df_agarwal_grid_1])

    # Zafar
    dict_zafar_0 = {'Method': ["Zafar"] * n, 'Fairness indicator ({})'.format(fair_name): test_fr_0_zafar[Zafar_id],
                    sens: [sens_0] * n}
    df_zafar_0 = pd.DataFrame(data=dict_zafar_0)
    dict_zafar_1 = {'Method': ["Zafar"] * n, 'Fairness indicator ({})'.format(fair_name): test_fr_1_zafar[Zafar_id],
                    sens: [sens_1] * n}
    df_zafar_1 = pd.DataFrame(data=dict_zafar_1)
    df = pd.concat([df, df_zafar_0])
    df = pd.concat([df, df_zafar_1])

    plt.figure()
    sns.set(font_scale=1.5)
    sns.violinplot(x="Method", y="Fairness indicator ({})".format(fair_name), hue=sens,
                   data=df)
    plt.show()

    return


if __name__ == "__main__":
    main()
