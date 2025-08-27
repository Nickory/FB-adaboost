from fair_adaboost import *


def main():
    D = 3
    base_type = 'CART'

    exp_id = 1

    assert exp_id in [i+1 for i in range(9)], "Experiment does not exist."

    if exp_id == 1:
        # EXP1: Adult-gender-acc
        lambd_list = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
        num_tree = 30
        exp_data = 'adult'
        sens_attr = 'gender'
        fair = 'acc'
        turn = False

    elif exp_id == 2:
        # EXP2: Adult-gender-fpr
        lambd_list = [0, 0.1, 0.15, 0.2, 0.25, 0.3]
        num_tree = 30
        exp_data = 'adult'
        sens_attr = 'gender'
        fair = 'fpr'
        turn = False

    elif exp_id == 3:
        # EXP3: Adult-gender-fnr
        lambd_list = [0, 0.02, 0.04, 0.05, 0.06, 0.07]
        num_tree = 30
        exp_data = 'adult'
        sens_attr = 'gender'
        fair = 'fnr'
        turn = True

    elif exp_id == 4:
        # EXP4: Compas-race-fnr
        lambd_list = [0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45]
        num_tree = 30
        exp_data = 'compas'
        sens_attr = 'race'
        fair = 'fnr'
        turn = True

    elif exp_id == 5:
        # EXP5: Compas-race-acc
        lambd_list = [0, 0.1, 0.15, 0.2]
        num_tree = 15
        exp_data = 'compas'
        sens_attr = 'race'
        fair = 'acc'
        turn = True

    elif exp_id == 6:
        # EXP6: Compas-race-fpr
        lambd_list = [0, 0.1, 0.2, 0.3, 0.35, 0.4]
        num_tree = 20
        exp_data = 'compas'
        sens_attr = 'race'
        fair = 'fpr'
        turn = False

    elif exp_id == 7:
        # EXP7: HSLS-race-fnr
        lambd_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        num_tree = 20
        exp_data = 'hsls'
        sens_attr = 'race'
        fair = 'fnr'
        turn = True

    elif exp_id == 8:
        # EXP8: HSLS-race-acc
        lambd_list = [0, 0.1, 0.2, 0.3]
        num_tree = 20
        exp_data = 'hsls'
        sens_attr = 'race'
        fair = 'acc'
        turn = True

    else:
        # EXP9: HSLS-race-fpr
        lambd_list = [0, 0.05, 0.1, 0.15]
        num_tree = 20
        exp_data = 'hsls'
        sens_attr = 'race'
        fair = 'fpr'
        turn = False

    seed_list = [i + 1 for i in range(20)]
    lambd_num = len(lambd_list)
    seed_num = len(seed_list)
    train_acc_array = np.ones((lambd_num, seed_num))
    train_fair_array = np.ones((lambd_num, seed_num))
    test_acc_array = np.ones((lambd_num, seed_num))
    test_fair_array = np.ones((lambd_num, seed_num))
    test_fair_0_array = np.ones((lambd_num, seed_num))
    test_fair_1_array = np.ones((lambd_num, seed_num))
    train_flag_array = np.zeros(lambd_num)

    for j in range(lambd_num):
        lambd = lambd_list[j]
        for k in range(seed_num):
            seed = seed_list[k]
            fair_adaboost_classifier, train_acc, train_fair, train_flag = train_balance(D=D, num_tree=num_tree,
                                                                                        lambd=lambd,
                                                                                        seed=seed, base_type=base_type,
                                                                                        fair=fair,
                                                                                        turn=turn,
                                                                                        input_file='./pkl_data/{}_balance_{}.pkl'
                                                                                        .format(exp_data, sens_attr))
            if train_flag:
                train_flag_array[j] += 1
            test_acc, test_fair, test_fair_0, test_fair_1 = test_balance(classifier=fair_adaboost_classifier,
                                                                         lambd=lambd,
                                                                         seed=seed, fair=fair, turn=turn,
                                                                         filename='./pkl_data/{}_balance_{}.pkl'.format(
                                                                             exp_data, sens_attr))
            train_acc_array[j][k] = train_acc
            train_fair_array[j][k] = train_fair
            test_acc_array[j][k] = test_acc
            test_fair_array[j][k] = test_fair
            test_fair_0_array[j][k] = test_fair_0
            test_fair_1_array[j][k] = test_fair_1

    train_acc_mean = np.mean(train_acc_array, axis=1)
    train_acc_std = np.std(train_acc_array, axis=1)
    train_fair_mean = np.mean(train_fair_array, axis=1)
    train_fair_std = np.std(train_fair_array, axis=1)
    test_acc_mean = np.mean(test_acc_array, axis=1)
    test_acc_std = np.std(test_acc_array, axis=1)
    test_fair_mean = np.mean(test_fair_array, axis=1)
    test_fair_std = np.std(test_fair_array, axis=1)

    print('train_acc_mean:{}'.format(train_acc_mean))
    print('train_acc_std:{}'.format(train_acc_std))
    print('train_fair_mean:{}'.format(train_fair_mean))
    print('train_fair_std:{}'.format(train_fair_std))
    print('test_acc_mean:{}'.format(test_acc_mean))
    print('test_acc_std:{}'.format(test_acc_std))
    print('test_fair_mean:{}'.format(test_fair_mean))
    print('test_fair_std:{}'.format(test_fair_std))

    np.save('./result/result_{}_{}/{}/train_acc_adaboost_{}_{}_{}.npy'.format(exp_data, sens_attr, fair, base_type, D,
                                                                              num_tree), train_acc_array)
    np.save('./result/result_{}_{}/{}/train_fair_adaboost_{}_{}_{}.npy'.format(exp_data, sens_attr, fair, base_type, D,
                                                                               num_tree), train_fair_array)
    np.save('./result/result_{}_{}/{}/test_acc_adaboost_{}_{}_{}.npy'.format(exp_data, sens_attr, fair, base_type, D,
                                                                             num_tree), test_acc_array)
    np.save('./result/result_{}_{}/{}/test_fair_adaboost_{}_{}_{}.npy'.format(exp_data, sens_attr, fair, base_type, D,
                                                                              num_tree), test_fair_array)
    np.save('./result/result_{}_{}/{}/test_fair_0_adaboost_{}_{}_{}.npy'.format(exp_data, sens_attr, fair, base_type, D,
                                                                                num_tree), test_fair_0_array)
    np.save('./result/result_{}_{}/{}/test_fair_1_adaboost_{}_{}_{}.npy'.format(exp_data, sens_attr, fair, base_type, D,
                                                                                num_tree), test_fair_1_array)

    return


if __name__ == "__main__":
    main()
