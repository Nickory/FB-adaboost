from sklearn.model_selection import train_test_split
from load_adult import *
from aif360.datasets import StandardDataset
from sklearn.tree import DecisionTreeClassifier
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.metrics import ClassificationMetric

"""
    Reference:https://github.com/haewon55/FairMIPForest/
"""


def main():
    seed_list = [i + 1 for i in range(20)]

    seed_num = len(seed_list)

    test_acc_array = np.ones(seed_num)
    test_fair_acc_array = np.ones(seed_num)
    test_fair_fpr_array = np.ones(seed_num)
    test_fair_fnr_array = np.ones(seed_num)
    test_fair_0_acc_array = np.ones(seed_num)
    test_fair_0_fpr_array = np.ones(seed_num)
    test_fair_0_fnr_array = np.ones(seed_num)
    test_fair_1_acc_array = np.ones(seed_num)
    test_fair_1_fpr_array = np.ones(seed_num)
    test_fair_1_fnr_array = np.ones(seed_num)

    # EXP1/2/3: Adult-gender-acc / Adult-gender-fpr / Adult-gender-fnr

    privileged_groups = [{'gender': 1}]
    unprivileged_groups = [{'gender': 0}]

    # load and process data
    df_train, df_test = load_adult()
    df = balance_data(df_train, 'income', 0)
    df = balance_data(df, 'gender', 1)

    for k in range(seed_num):
        seed = seed_list[k]
        dataset_orig_train, dataset_orig_test = train_test_split(df, test_size=0.3, random_state=seed)

        # dataset_orig_train_no_sens = dataset_orig_train.drop(columns=['gender', 'income'])
        # dataset_orig_test_no_sens = dataset_orig_test.drop(columns=['gender', 'income'])
        #
        # dataset_orig_train = pd.concat([dataset_orig_train_no_sens, dataset_orig_train[['gender', 'income']]], axis=1)
        # dataset_orig_test = pd.concat([dataset_orig_test_no_sens, dataset_orig_test[['gender', 'income']]], axis=1)

        dataset_orig_valid, dataset_orig_test = train_test_split(dataset_orig_test, test_size=0.5, random_state=seed)

        ### Converting to AIF360 StandardDataset objects ###
        dataset_orig_train = StandardDataset(dataset_orig_train, label_name='income', favorable_classes=[1],
                                             protected_attribute_names=['gender'], privileged_classes=[[1]])
        dataset_orig_valid = StandardDataset(dataset_orig_valid, label_name='income', favorable_classes=[1],
                                             protected_attribute_names=['gender'], privileged_classes=[[1]])
        dataset_orig_test = StandardDataset(dataset_orig_test, label_name='income', favorable_classes=[1],
                                            protected_attribute_names=['gender'], privileged_classes=[[1]])

        # Placeholder for predicted and transformed datasets
        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

        dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)

        idx_wo_protected = list(set(range(13)) - set([8]))
        X_train = dataset_orig_train.features[:, idx_wo_protected]
        y_train = dataset_orig_train.labels.ravel()

        lmod = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=seed)

        lmod.fit(X_train, y_train)

        fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
        y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]

        # Prediction probs for validation and testing data
        X_valid = dataset_orig_valid.features[:, idx_wo_protected]
        y_valid_pred_prob = lmod.predict_proba(X_valid)[:, fav_idx]

        X_test = dataset_orig_test.features[:, idx_wo_protected]
        y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]

        class_thresh = 0.5
        dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
        dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)
        dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

        y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
        y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
        y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
        dataset_orig_train_pred.labels = y_train_pred

        y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
        y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
        y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
        dataset_orig_valid_pred.labels = y_valid_pred

        y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
        y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
        y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
        dataset_orig_test_pred.labels = y_test_pred

        cpp = EqOddsPostprocessing(privileged_groups=privileged_groups,
                                   unprivileged_groups=unprivileged_groups,
                                   seed=seed)
        cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)

        dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)
        dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

        dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)
        dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

        cm_pred_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)
        cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)

        test_fr_acc = np.abs(cm_transf_test.difference(cm_transf_test.accuracy))
        test_fair_acc_array[k] = test_fr_acc
        test_fair_0_acc_array[k] = cm_transf_test.accuracy(privileged=False)
        test_fair_1_acc_array[k] = cm_transf_test.accuracy(privileged=True)
        # print(test_fr_acc)

        test_fr_fpr = np.abs(cm_transf_test.difference(cm_transf_test.false_positive_rate))
        test_fair_fpr_array[k] = test_fr_fpr
        test_fair_0_fpr_array[k] = cm_transf_test.false_positive_rate(privileged=False)
        test_fair_1_fpr_array[k] = cm_transf_test.false_positive_rate(privileged=True)
        # print(test_fr_fpr)

        test_fr_fnr = np.abs(cm_transf_test.difference(cm_transf_test.false_negative_rate))
        test_fair_fnr_array[k] = test_fr_fnr
        test_fair_0_fnr_array[k] = cm_transf_test.false_negative_rate(privileged=False)
        test_fair_1_fnr_array[k] = cm_transf_test.false_negative_rate(privileged=True)
        # print(test_fr_fnr)

        test_acc = cm_transf_test.accuracy()
        test_acc_array[k] = test_acc
        # print(test_acc)

    np.save('./result/result_adult_gender/acc/test_acc_hardt.npy', test_acc_array)
    np.save('./result/result_adult_gender/acc/test_fair_hardt.npy', test_fair_acc_array)
    np.save('./result/result_adult_gender/acc/test_fair_0_hardt.npy', test_fair_0_acc_array)
    np.save('./result/result_adult_gender/acc/test_fair_1_hardt.npy', test_fair_1_acc_array)

    np.save('./result/result_adult_gender/fpr/test_acc_hardt.npy', test_acc_array)
    np.save('./result/result_adult_gender/fpr/test_fair_hardt.npy', test_fair_fpr_array)
    np.save('./result/result_adult_gender/fpr/test_fair_0_hardt.npy', test_fair_0_fpr_array)
    np.save('./result/result_adult_gender/fpr/test_fair_1_hardt.npy', test_fair_1_fpr_array)

    np.save('./result/result_adult_gender/fnr/test_acc_hardt.npy', test_acc_array)
    np.save('./result/result_adult_gender/fnr/test_fair_hardt.npy', test_fair_fnr_array)
    np.save('./result/result_adult_gender/fnr/test_fair_0_hardt.npy', test_fair_0_fnr_array)
    np.save('./result/result_adult_gender/fnr/test_fair_1_hardt.npy', test_fair_1_fnr_array)

    return


if __name__ == "__main__":
    main()
