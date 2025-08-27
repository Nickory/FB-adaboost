from sklearn.model_selection import train_test_split
from load_adult import *
from aif360.datasets import StandardDataset
from sklearn.tree import DecisionTreeClassifier
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction

"""
    Reference:https://github.com/haewon55/FairMIPForest/
"""


def main():
    seed_list = [i + 1 for i in range(20)]
    eps_list = [0.001, 0.005, 0.01, 0.05, 0.1]

    eps_num = len(eps_list)
    seed_num = len(seed_list)

    fair_constraint = "fpr"  # "acc"/"fpr"/"fnr"

    assert fair_constraint in ["acc", "fpr", "fnr"], "Fair constraint does not exist."

    if fair_constraint == "acc":
        constraint = "ErrorRateParity"
    elif fair_constraint == "fpr":
        constraint = "FalsePositiveRateParity"
    else:
        constraint = "TruePositiveRateParity"

    train_acc_array = np.ones((eps_num, seed_num))
    train_fair_acc_array = np.ones((eps_num, seed_num))
    train_fair_fpr_array = np.ones((eps_num, seed_num))
    train_fair_fnr_array = np.ones((eps_num, seed_num))
    test_acc_array = np.ones((eps_num, seed_num))
    test_fair_acc_array = np.ones((eps_num, seed_num))
    test_fair_fpr_array = np.ones((eps_num, seed_num))
    test_fair_fnr_array = np.ones((eps_num, seed_num))
    test_fair_0_acc_array = np.ones((eps_num, seed_num))
    test_fair_0_fpr_array = np.ones((eps_num, seed_num))
    test_fair_0_fnr_array = np.ones((eps_num, seed_num))
    test_fair_1_acc_array = np.ones((eps_num, seed_num))
    test_fair_1_fpr_array = np.ones((eps_num, seed_num))
    test_fair_1_fnr_array = np.ones((eps_num, seed_num))

    privileged_groups = [{'gender': 1}]
    unprivileged_groups = [{'gender': 0}]

    df_train, df_test = load_adult()
    df = balance_data(df_train, 'income', 0)
    df = balance_data(df, 'gender', 1)

    for j in range(eps_num):
        eps = eps_list[j]
        for k in range(seed_num):
            seed = seed_list[k]
            dataset_orig_train, dataset_orig_test = train_test_split(df, test_size=0.3, random_state=seed)

            ### Converting to AIF360 StandardDataset objects ###
            dataset_orig_train = StandardDataset(dataset_orig_train, label_name='income', favorable_classes=[1],
                                                 protected_attribute_names=['gender'], privileged_classes=[[1]])
            dataset_orig_test = StandardDataset(dataset_orig_test, label_name='income', favorable_classes=[1],
                                                protected_attribute_names=['gender'], privileged_classes=[[1]])

            # idx_wo_protected = list(set(range(13)) - set([8]))
            # X_train = dataset_orig_train.features[:, idx_wo_protected]
            # y_train = dataset_orig_train.labels.ravel()

            # lmod = DecisionTreeClassifier(max_depth=3, criterion="gini")
            # lmod.fit(X_train, y_train, sample_weight=dataset_orig_train.instance_weights)
            #
            # X_test = dataset_orig_test.features[:, idx_wo_protected]
            # y_test = dataset_orig_test.labels.ravel()
            #
            # y_pred = lmod.predict(X_test)
            #
            # dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
            # dataset_orig_test_pred.labels = y_pred
            #
            # cm_pred_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
            #                                     unprivileged_groups=unprivileged_groups,
            #                                     privileged_groups=privileged_groups)
            # acc_orig = cm_pred_test.accuracy()
            # fr_orig = cm_pred_test.difference(cm_pred_test.false_positive_rate)

            estimator = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=seed)

            np.random.seed(0)
            exp_grad_red = ExponentiatedGradientReduction(estimator=estimator, constraints=constraint,
                                                          drop_prot_attr=False, eps=eps)
            exp_grad_red.fit(dataset_orig_train)
            exp_grad_red_pred = exp_grad_red.predict(dataset_orig_test)

            cm_transf_test = ClassificationMetric(dataset_orig_test, exp_grad_red_pred,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)

            test_acc = cm_transf_test.accuracy()
            test_acc_array[j][k] = test_acc
            # print(test_acc)

            test_acc_fr = np.abs(cm_transf_test.difference(cm_transf_test.accuracy))
            test_fair_acc_array[j][k] = test_acc_fr
            test_fair_0_acc_array[j][k] = cm_transf_test.accuracy(privileged=False)
            test_fair_1_acc_array[j][k] = cm_transf_test.accuracy(privileged=True)
            # print(test_acc_fr)

            test_fpr_fr = np.abs(cm_transf_test.difference(cm_transf_test.false_positive_rate))
            test_fair_fpr_array[j][k] = test_fpr_fr
            test_fair_0_fpr_array[j][k] = cm_transf_test.false_positive_rate(privileged=False)
            test_fair_1_fpr_array[j][k] = cm_transf_test.false_positive_rate(privileged=True)
            # print(test_fpr_fr)

            test_fnr_fr = np.abs(cm_transf_test.difference(cm_transf_test.false_negative_rate))
            test_fair_fnr_array[j][k] = test_fnr_fr
            test_fair_0_fnr_array[j][k] = cm_transf_test.false_negative_rate(privileged=False)
            test_fair_1_fnr_array[j][k] = cm_transf_test.false_negative_rate(privileged=True)
            # print(test_fnr_fr)

            exp_grad_red_pred_train = exp_grad_red.predict(dataset_orig_train)
            cm_transf_train = ClassificationMetric(dataset_orig_train, exp_grad_red_pred_train,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)

            train_acc = cm_transf_train.accuracy()
            train_acc_array[j][k] = train_acc
            # print(train_acc)

            train_acc_fr = np.abs(cm_transf_train.difference(cm_transf_train.accuracy))
            train_fair_acc_array[j][k] = train_acc_fr
            # print(train_acc_fr)

            train_fpr_fr = np.abs(cm_transf_train.difference(cm_transf_train.false_positive_rate))
            train_fair_fpr_array[j][k] = train_fpr_fr
            # print(train_fpr_fr)

            train_fnr_fr = np.abs(cm_transf_train.difference(cm_transf_train.false_negative_rate))
            train_fair_fnr_array[j][k] = train_fnr_fr
            # print(train_fnr_fr)

    if fair_constraint == "acc":
        np.save('./result/result_adult_gender/acc/train_acc_agarwal.npy', train_acc_array)
        np.save('./result/result_adult_gender/acc/train_fair_agarwal.npy', train_fair_acc_array)
        np.save('./result/result_adult_gender/acc/test_acc_agarwal.npy', test_acc_array)
        np.save('./result/result_adult_gender/acc/test_fair_agarwal.npy', test_fair_acc_array)
        np.save('./result/result_adult_gender/acc/test_fair_0_agarwal.npy', test_fair_0_acc_array)
        np.save('./result/result_adult_gender/acc/test_fair_1_agarwal.npy', test_fair_1_acc_array)
    elif fair_constraint == "fpr":
        np.save('./result/result_adult_gender/fpr/train_acc_agarwal.npy', train_acc_array)
        np.save('./result/result_adult_gender/fpr/train_fair_agarwal.npy', train_fair_fpr_array)
        np.save('./result/result_adult_gender/fpr/test_acc_agarwal.npy', test_acc_array)
        np.save('./result/result_adult_gender/fpr/test_fair_agarwal.npy', test_fair_fpr_array)
        np.save('./result/result_adult_gender/fpr/test_fair_0_agarwal.npy', test_fair_0_fpr_array)
        np.save('./result/result_adult_gender/fpr/test_fair_1_agarwal.npy', test_fair_1_fpr_array)
    else:
        np.save('./result/result_adult_gender/fnr/train_acc_agarwal.npy', train_acc_array)
        np.save('./result/result_adult_gender/fnr/train_fair_agarwal.npy', train_fair_fnr_array)
        np.save('./result/result_adult_gender/fnr/test_acc_agarwal.npy', test_acc_array)
        np.save('./result/result_adult_gender/fnr/test_fair_agarwal.npy', test_fair_fnr_array)
        np.save('./result/result_adult_gender/fnr/test_fair_0_agarwal.npy', test_fair_0_fnr_array)
        np.save('./result/result_adult_gender/fnr/test_fair_1_agarwal.npy', test_fair_1_fnr_array)

    return


if __name__ == "__main__":
    main()
