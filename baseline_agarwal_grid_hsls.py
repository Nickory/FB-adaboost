from sklearn.model_selection import train_test_split
from load_hsls import *
from aif360.datasets import StandardDataset
from sklearn.tree import DecisionTreeClassifier
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing.grid_search_reduction import GridSearchReduction

"""
    Reference:https://github.com/haewon55/FairMIPForest/
"""


def main():
    seed_list = [i + 1 for i in range(20)]
    constraintweight_list = [0.4, 0.5, 0.6, 0.7]

    constraintweight_num = len(constraintweight_list)
    seed_num = len(seed_list)

    fair_constraint = "fpr"  # "acc"/"fpr"/"fnr"

    assert fair_constraint in ["acc", "fpr", "fnr"], "Fair constraint does not exist."

    if fair_constraint == "acc":
        grid_size = 20
        grid_limit = 2.0
        constraint = "ErrorRateParity"
    elif fair_constraint == "fpr":
        grid_size = 20
        grid_limit = 2.0
        constraint = "FalsePositiveRateParity"
    else:
        grid_size = 20
        grid_limit = 2.0
        constraint = "TruePositiveRateParity"

    train_acc_array = np.ones((constraintweight_num, seed_num))
    train_fair_acc_array = np.ones((constraintweight_num, seed_num))
    train_fair_fpr_array = np.ones((constraintweight_num, seed_num))
    train_fair_fnr_array = np.ones((constraintweight_num, seed_num))
    test_acc_array = np.ones((constraintweight_num, seed_num))
    test_fair_acc_array = np.ones((constraintweight_num, seed_num))
    test_fair_fpr_array = np.ones((constraintweight_num, seed_num))
    test_fair_fnr_array = np.ones((constraintweight_num, seed_num))
    test_fair_0_acc_array = np.ones((constraintweight_num, seed_num))
    test_fair_0_fpr_array = np.ones((constraintweight_num, seed_num))
    test_fair_0_fnr_array = np.ones((constraintweight_num, seed_num))
    test_fair_1_acc_array = np.ones((constraintweight_num, seed_num))
    test_fair_1_fpr_array = np.ones((constraintweight_num, seed_num))
    test_fair_1_fnr_array = np.ones((constraintweight_num, seed_num))

    privileged_groups = [{'racebin': 1}]
    unprivileged_groups = [{'racebin': 0}]

    df = load_hsls()

    df = balance_data(df, "racebin", 1)

    for j in range(constraintweight_num):
        constraintweight = constraintweight_list[j]
        for k in range(seed_num):
            seed = seed_list[k]
            dataset_orig_train, dataset_orig_test = train_test_split(df, test_size=0.3, random_state=seed)

            ### Converting to AIF360 StandardDataset objects ###
            dataset_orig_train = StandardDataset(dataset_orig_train, label_name='gradebin', favorable_classes=[1],
                                                 protected_attribute_names=['racebin'], privileged_classes=[[1]])
            dataset_orig_test = StandardDataset(dataset_orig_test, label_name='gradebin', favorable_classes=[1],
                                                protected_attribute_names=['racebin'], privileged_classes=[[1]])

            estimator = DecisionTreeClassifier(max_depth=3, criterion="gini")

            np.random.seed(0)
            grid_grad_red = GridSearchReduction(estimator=estimator, constraints=constraint, grid_size=grid_size,
                                                grid_limit=grid_limit, drop_prot_attr=False,
                                                constraint_weight=constraintweight)
            grid_grad_red.fit(dataset_orig_train)
            grid_grad_red_pred = grid_grad_red.predict(dataset_orig_test)

            cm_transf_test = ClassificationMetric(dataset_orig_test, grid_grad_red_pred,
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

            exp_grad_red_pred_train = grid_grad_red.predict(dataset_orig_train)
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
        np.save('./result/result_hsls_race/acc/train_acc_agarwal_grid.npy', train_acc_array)
        np.save('./result/result_hsls_race/acc/train_fair_agarwal_grid.npy', train_fair_acc_array)
        np.save('./result/result_hsls_race/acc/test_acc_agarwal_grid.npy', test_acc_array)
        np.save('./result/result_hsls_race/acc/test_fair_agarwal_grid.npy', test_fair_acc_array)
        np.save('./result/result_hsls_race/acc/test_fair_0_agarwal_grid.npy', test_fair_0_acc_array)
        np.save('./result/result_hsls_race/acc/test_fair_1_agarwal_grid.npy', test_fair_1_acc_array)
    elif fair_constraint == "fpr":
        np.save('./result/result_hsls_race/fpr/train_acc_agarwal_grid.npy', train_acc_array)
        np.save('./result/result_hsls_race/fpr/train_fair_agarwal_grid.npy', train_fair_fpr_array)
        np.save('./result/result_hsls_race/fpr/test_acc_agarwal_grid.npy', test_acc_array)
        np.save('./result/result_hsls_race/fpr/test_fair_agarwal_grid.npy', test_fair_fpr_array)
        np.save('./result/result_hsls_race/fpr/test_fair_0_agarwal_grid.npy', test_fair_0_fpr_array)
        np.save('./result/result_hsls_race/fpr/test_fair_1_agarwal_grid.npy', test_fair_1_fpr_array)
    else:
        np.save('./result/result_hsls_race/fnr/train_acc_agarwal_grid.npy', train_acc_array)
        np.save('./result/result_hsls_race/fnr/train_fair_agarwal_grid.npy', train_fair_fnr_array)
        np.save('./result/result_hsls_race/fnr/test_acc_agarwal_grid.npy', test_acc_array)
        np.save('./result/result_hsls_race/fnr/test_fair_agarwal_grid.npy', test_fair_fnr_array)
        np.save('./result/result_hsls_race/fnr/test_fair_0_agarwal_grid.npy', test_fair_0_fnr_array)
        np.save('./result/result_hsls_race/fnr/test_fair_1_agarwal_grid.npy', test_fair_1_fnr_array)

    return


if __name__ == "__main__":
    main()
