import numpy as np
import matplotlib.pyplot as plt


def plot_result(seed_num=20):
    base_type = 'CART'
    mu = 1.2
    exp_id = 6  # 1 -- 9
    result_type = "test"  # "train"/"test"

    if exp_id == 1:
        # EXP1: Adult-gender-acc
        exp_data = 'adult'
        title = 'Adult'
        sens_attr = 'gender'
        fair = 'acc'
        fair_title = 'ACC'
        T = 30

    elif exp_id == 2:
        # EXP2: Adult-gender-fpr
        exp_data = 'adult'
        title = 'Adult'
        sens_attr = 'gender'
        fair = 'fpr'
        fair_title = 'FPR'
        T = 30

    elif exp_id == 3:
        # EXP3: Adult-gender-fnr
        exp_data = 'adult'
        title = 'Adult'
        sens_attr = 'gender'
        fair = 'fnr'
        fair_title = 'FNR'
        T = 30

    elif exp_id == 4:
        # EXP4: Compas-race-fnr
        exp_data = 'compas'
        title = 'Compas'
        sens_attr = 'race'
        fair = 'fnr'
        fair_title = 'FNR'
        T = 30

    elif exp_id == 5:
        # EXP5: Compas-race-acc
        exp_data = 'compas'
        title = 'Compas'
        sens_attr = 'race'
        fair = 'acc'
        fair_title = 'ACC'
        T = 15

    elif exp_id == 6:
        # EXP6: Compas-race-fpr
        exp_data = 'compas'
        title = 'Compas'
        sens_attr = 'race'
        fair = 'fpr'
        fair_title = 'FPR'
        T = 20

    elif exp_id == 7:
        # EXP7: HSLS-race-fnr
        exp_data = 'hsls'
        title = 'HSLS'
        sens_attr = 'race'
        fair = 'fnr'
        fair_title = 'FNR'
        T = 20

    elif exp_id == 8:
        # EXP8: HSLS-race-acc
        exp_data = 'hsls'
        title = 'HSLS'
        sens_attr = 'race'
        fair = 'acc'
        fair_title = 'ACC'
        T = 20

    elif exp_id == 9:
        # EXP8: HSLS-race-fpr
        exp_data = 'hsls'
        title = 'HSLS'
        sens_attr = 'race'
        fair = 'fpr'
        fair_title = 'FPR'
        T = 20

    else:
        print("Experiment does not exist.")
        return

    # load_result
    train_acc = np.load('./result/result_{}_{}/{}/train_acc_adaboost_{}_3_{}.npy'.format(exp_data, sens_attr, fair, base_type, T))
    train_fair_loss = np.load('./result/result_{}_{}/{}/train_fair_adaboost_{}_3_{}.npy'.format(exp_data, sens_attr, fair, base_type, T))
    test_acc = np.load('./result/result_{}_{}/{}/test_acc_adaboost_{}_3_{}.npy'.format(exp_data, sens_attr, fair, base_type, T))
    test_fair_loss = np.load('./result/result_{}_{}/{}/test_fair_adaboost_{}_3_{}.npy'.format(exp_data, sens_attr, fair, base_type, T))

    train_acc_zafar = np.load('./result/result_{}_{}/{}/train_acc_zafar_{}.npy'.format(exp_data, sens_attr, fair, mu))
    train_fr_zafar = np.load('./result/result_{}_{}/{}/train_fair_zafar_{}.npy'.format(exp_data, sens_attr, fair, mu))
    test_acc_zafar = np.load('./result/result_{}_{}/{}/test_acc_zafar_{}.npy'.format(exp_data, sens_attr, fair, mu))
    test_fr_zafar = np.load('./result/result_{}_{}/{}/test_fair_zafar_{}.npy'.format(exp_data, sens_attr, fair, mu))

    test_acc_hardt = np.load('./result/result_{}_{}/{}/test_acc_hardt.npy'.format(exp_data, sens_attr, fair))
    test_fr_hardt = np.load('./result/result_{}_{}/{}/test_fair_hardt.npy'.format(exp_data, sens_attr, fair))

    train_acc_agarwal = np.load('./result/result_{}_{}/{}/train_acc_agarwal.npy'.format(exp_data, sens_attr, fair))
    train_fr_agarwal = np.load('./result/result_{}_{}/{}/train_fair_agarwal.npy'.format(exp_data, sens_attr, fair))
    test_acc_agarwal = np.load('./result/result_{}_{}/{}/test_acc_agarwal.npy'.format(exp_data, sens_attr, fair))
    test_fr_agarwal = np.load('./result/result_{}_{}/{}/test_fair_agarwal.npy'.format(exp_data, sens_attr, fair))

    train_acc_agarwal_grid = np.load('./result/result_{}_{}/{}/train_acc_agarwal_grid.npy'.format(exp_data, sens_attr, fair))
    train_fr_agarwal_grid = np.load('./result/result_{}_{}/{}/train_fair_agarwal_grid.npy'.format(exp_data, sens_attr, fair))
    test_acc_agarwal_grid = np.load('./result/result_{}_{}/{}/test_acc_agarwal_grid.npy'.format(exp_data, sens_attr, fair))
    test_fr_agarwal_grid = np.load('./result/result_{}_{}/{}/test_fair_agarwal_grid.npy'.format(exp_data, sens_attr, fair))

    # calculate
    train_acc_mean = np.mean(train_acc, axis=1)
    train_acc_se = np.std(train_acc, axis=1) / np.sqrt(seed_num)
    train_fair_loss_mean = np.mean(train_fair_loss, axis=1)
    train_fair_loss_se = np.std(train_fair_loss, axis=1) / np.sqrt(seed_num)
    test_acc_mean = np.mean(test_acc, axis=1)
    test_acc_se = np.std(test_acc, axis=1) / np.sqrt(seed_num)
    test_fair_loss_mean = np.mean(test_fair_loss, axis=1)
    test_fair_loss_se = np.std(test_fair_loss, axis=1) / np.sqrt(seed_num)

    train_acc_zafar_mean = np.mean(train_acc_zafar, axis=1)
    train_acc_zafar_se = np.std(train_acc_zafar, axis=1) / np.sqrt(seed_num)
    train_fr_zafar_mean = np.mean(train_fr_zafar, axis=1)
    train_fr_zafar_se = np.std(train_fr_zafar, axis=1) / np.sqrt(seed_num)
    test_acc_zafar_mean = np.mean(test_acc_zafar, axis=1)
    test_acc_zafar_se = np.std(test_acc_zafar, axis=1) / np.sqrt(seed_num)
    test_fr_zafar_mean = np.mean(test_fr_zafar, axis=1)
    test_fr_zafar_se = np.std(test_fr_zafar, axis=1) / np.sqrt(seed_num)

    test_acc_hardt_mean = np.mean(test_acc_hardt)
    test_acc_hardt_se = np.std(test_acc_hardt) / np.sqrt(seed_num)
    test_fr_hardt_mean = np.mean(test_fr_hardt)
    test_fr_hardt_se = np.std(test_fr_hardt) / np.sqrt(seed_num)

    train_acc_agarwal_grid_mean = np.mean(train_acc_agarwal_grid, axis=1)
    train_acc_agarwal_grid_se = np.std(train_acc_agarwal_grid, axis=1) / np.sqrt(seed_num)
    train_fr_agarwal_grid_mean = np.mean(train_fr_agarwal_grid, axis=1)
    train_fr_agarwal_grid_se = np.std(train_fr_agarwal_grid, axis=1) / np.sqrt(seed_num)
    test_acc_agarwal_grid_mean = np.mean(test_acc_agarwal_grid, axis=1)
    test_acc_agarwal_grid_se = np.std(test_acc_agarwal_grid, axis=1) / np.sqrt(seed_num)
    test_fr_agarwal_grid_mean = np.mean(test_fr_agarwal_grid, axis=1)
    test_fr_agarwal_grid_se = np.std(test_fr_agarwal_grid, axis=1) / np.sqrt(seed_num)

    train_acc_agarwal_mean = np.mean(train_acc_agarwal, axis=1)
    train_acc_agarwal_se = np.std(train_acc_agarwal, axis=1) / np.sqrt(seed_num)
    train_fr_agarwal_mean = np.mean(train_fr_agarwal, axis=1)
    train_fr_agarwal_se = np.std(train_fr_agarwal, axis=1) / np.sqrt(seed_num)
    test_acc_agarwal_mean = np.mean(test_acc_agarwal, axis=1)
    test_acc_agarwal_se = np.std(test_acc_agarwal, axis=1) / np.sqrt(seed_num)
    test_fr_agarwal_mean = np.mean(test_fr_agarwal, axis=1)
    test_fr_agarwal_se = np.std(test_fr_agarwal, axis=1) / np.sqrt(seed_num)

    print('Boost:')
    print("Acc: {}".format(test_acc_mean))
    print("Fairness loss: {}".format(test_fair_loss_mean))

    print('Hardt:')
    print("Acc: {}".format(test_acc_hardt_mean))
    print("Fairness loss: {}".format(test_fr_hardt_mean))

    print('Agarwal-Exp:')
    print("Acc: {}".format(test_acc_agarwal_mean))
    print("Fairness loss: {}".format(test_fr_agarwal_mean))

    print('Agarwal-Grid:')
    print("Acc: {}".format(test_acc_agarwal_grid_mean))
    print("Fairness loss: {}".format(test_fr_agarwal_grid_mean))

    print('Zafar:')
    print("Acc: {}".format(test_acc_zafar_mean))
    print("Fairness loss: {}".format(test_fr_zafar_mean))

    plt.figure()
    if result_type == "train":
        plt.title('{}({})-Training Set'.format(title, fair_title), fontsize=15)
        plt.errorbar(train_fair_loss_mean[0], train_acc_mean[0], xerr=train_fair_loss_se[0], yerr=train_acc_se[0],
                     fmt=':s', label='Baseline-AdaBoost', color='darkblue', elinewidth=1, markersize=8, capsize=5)
        plt.errorbar(train_fair_loss_mean[1:], train_acc_mean[1:], xerr=train_fair_loss_se[1:], yerr=train_acc_se[1:],
                     fmt='-o', label='FAB', color='red', elinewidth=1, markersize=8,
                     capsize=5)
        plt.errorbar(train_fr_zafar_mean, train_acc_zafar_mean, xerr=train_fr_zafar_se, yerr=train_acc_zafar_se,
                     fmt='-o', label='Baseline-Zafar', color='darkgreen', elinewidth=1, markersize=8,
                     capsize=5, markerfacecolor='white')
        plt.errorbar(train_fr_agarwal_mean, train_acc_agarwal_mean, xerr=train_fr_agarwal_se, yerr=train_acc_agarwal_se,
                     fmt='-o', label='Baseline-Agarwal-Exp', color='orange', elinewidth=1, markersize=8,
                     capsize=5, markerfacecolor='white')
        plt.errorbar(train_fr_agarwal_grid_mean, train_acc_agarwal_grid_mean, xerr=train_fr_agarwal_grid_se, yerr=train_acc_agarwal_grid_se,
                     fmt='-o', label='Baseline-Agarwal-Grid', color='black', elinewidth=1, markersize=8,
                     capsize=5, markerfacecolor='white')

    else:
        plt.title('{}({})-Testing Set'.format(title, fair_title), fontsize=15)
        plt.errorbar(test_fair_loss_mean[0], test_acc_mean[0], xerr=test_fair_loss_se[0], yerr=test_acc_se[0],
                     fmt=':s', label='Baseline-AdaBoost', color='darkblue', elinewidth=1, markersize=8,
                     capsize=5)
        plt.errorbar(test_fair_loss_mean[1:], test_acc_mean[1:], xerr=test_fair_loss_se[1:], yerr=test_acc_se[1:],
                     fmt='-o', label='FAB', color='red', elinewidth=1, markersize=8, capsize=5)
        plt.errorbar(test_fr_zafar_mean, test_acc_zafar_mean, xerr=test_fr_zafar_se, yerr=test_acc_zafar_se,
                     fmt='-o', label='Baseline-Zafar', color='darkgreen', elinewidth=1, markersize=8,
                     capsize=5, markerfacecolor='white')
        plt.errorbar(test_fr_hardt_mean, test_acc_hardt_mean, xerr=test_fr_hardt_se, yerr=test_acc_hardt_se,
                     fmt='-o', label='Baseline-Hardt', color='purple', elinewidth=1, markersize=8,
                     capsize=5, markerfacecolor='white')
        plt.errorbar(test_fr_agarwal_mean, test_acc_agarwal_mean, xerr=test_fr_agarwal_se, yerr=test_acc_agarwal_se,
                     fmt='-o', label='Baseline-Agarwal-Exp', color='orange', elinewidth=1, markersize=8,
                     capsize=5, markerfacecolor='white')
        plt.errorbar(test_fr_agarwal_grid_mean, test_acc_agarwal_grid_mean, xerr=test_fr_agarwal_grid_se,
                     yerr=test_acc_agarwal_grid_se,
                     fmt='-o', label='Baseline-Agarwal-Grid', color='black', elinewidth=1, markersize=8,
                     capsize=5, markerfacecolor='white')

    # plt.legend(prop={'size': 12.5}, loc='lower right', bbox_to_anchor=(1, 0.1))  # exp1
    # plt.gcf().subplots_adjust(bottom=0.25)
    plt.legend()
    # plt.legend(prop={'size': 17.5}, bbox_to_anchor=(1.08, -0.15), ncol=3)  # train 1.08 test 1.05
    plt.xlabel('Fairness Loss', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xticks(fontsize=12.5)
    plt.yticks(fontsize=12.5)

    # annotations = ['AdaBoost', '0.1', '0.2', '0.3', '0.4', '0.45', '0.5']
    # for i, label in enumerate(annotations):
    #     plt.annotate(label, xy=(train_fair_loss_mean[i], train_acc_mean[i]))

    plt.show()

    return


if __name__ == "__main__":
    plot_result()
