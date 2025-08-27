from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.model_selection import train_test_split
from estimate import *

"""
    Reference:https://www.cnblogs.com/lifz-ml/p/15737137.html
"""


class FairAdaBoost:
    def __init__(self, X, y, S, lambd, T=100, fair='acc', random_seed=42):
        # features
        self.X = X
        # labels
        y[y == 0] = -1
        self.y = y.reshape(-1, 1)
        # sensitive attribute
        self.S = S
        # fairness preference
        self.lambd = lambd
        # num of base classifiers
        self.T = T
        # random seeds
        self.random_seed = random_seed
        # num of samples
        self.sample_num = self.X.shape[0]
        # set of base classifiers
        self.classifiers = []
        # weights of base classifiers
        self.classifiers_weights = []
        # weights of samples
        self.sample_weights = np.ones(self.sample_num) / self.sample_num
        if fair == 'acc':
            """
                assume s=0 has lower error
            """
            sample_num_1 = self.S.sum()
            sample_num_0 = self.sample_num - sample_num_1
            if self.lambd / sample_num_0 > 1 / self.sample_num:
                print("Recommended maximum lambda: {}".format(sample_num_0 / self.sample_num))
            self.sample_weights += self.S * self.lambd / sample_num_1
            self.sample_weights -= (1 - self.S) * self.lambd / sample_num_0

        elif fair == 'fpr':
            """
                assume s=0 has lower fpr
            """
            sample_num_1_negative = (self.S * (0.5 - 0.5 * y)).sum()
            sample_num_0_negative = ((1 - self.S) * (0.5 - 0.5 * y)).sum()
            if self.lambd / sample_num_0_negative > 1 / self.sample_num:
                print("Recommended maximum lambda: {}".format(sample_num_0_negative / self.sample_num))
            self.sample_weights += (self.S * (0.5 - 0.5 * y)) * self.lambd / sample_num_1_negative
            self.sample_weights -= ((1 - self.S) * (0.5 - 0.5 * y)) * self.lambd / sample_num_0_negative

        elif fair == 'fnr':
            """
                assume s=0 has lower fnr
            """
            sample_num_1_positive = (self.S * (0.5 + 0.5 * y)).sum()
            sample_num_0_positive = ((1 - self.S) * (0.5 + 0.5 * y)).sum()
            if self.lambd / sample_num_0_positive > 1 / self.sample_num:
                print("Recommended maximum lambda: {}".format(sample_num_0_positive / self.sample_num))
            self.sample_weights += (self.S * (0.5 + 0.5 * y)) * self.lambd / sample_num_1_positive
            self.sample_weights -= ((1 - self.S) * (0.5 + 0.5 * y)) * self.lambd / sample_num_0_positive

    def fit(self, D=3, base_type='entropy'):
        for t in range(self.T):
            # base classifier: CART('gini'),ID3('entropy')
            base_classifier = DecisionTreeClassifier(max_depth=D, criterion=base_type, random_state=self.random_seed). \
                fit(self.X, self.y, self.sample_weights)
            # predict by base classifier
            y_pre = base_classifier.predict(self.X)
            # calculate error
            error = np.sum(self.sample_weights * (np.ravel(y_pre) != np.ravel(self.y)).astype(int)) \
                    / self.sample_weights.sum()
            if error > 0.5:
                print("Stop when t = {}".format(t))
                break
            self.classifiers.append(base_classifier)
            # calculate weight
            current_weight = 0.5 * np.log((1 - error) / error)
            self.classifiers_weights.append(current_weight)
            # update sample weights
            self.sample_weights *= np.exp(- current_weight * np.ravel(y_pre) * np.ravel(self.y))
            self.sample_weights /= self.sample_weights.sum()

        self.classifiers_weights = np.array(self.classifiers_weights)

        return

    def predict(self, X):
        # predict by base classifiers
        y_hats = np.array([tree.predict(X) for tree in self.classifiers])
        # add weights
        y_hats = np.array([np.sign(np.sum(self.classifiers_weights * y_hats[:, i])) for i in range(y_hats.shape[1])])
        # label 1/-1 -→ 1/0
        y_hats[y_hats == -1] = 0
        return y_hats


def train_balance(D=3, num_tree=20, fair='acc', input_file='./pkl_data/adult_balance.pkl', seed=1, lambd=0.1,
                  turn=False, base_type='ID3'):
    with open(input_file, 'rb') as handle:
        data = pickle.load(handle)

    X_orig = data['X']
    y_orig = data['y']
    S_orig = data['S']

    if turn:
        S_orig = 1 - S_orig

    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X_orig, y_orig, S_orig, test_size=0.3,
                                                                         random_state=seed)

    classifier = FairAdaBoost(X=X_train, y=y_train, S=S_train, lambd=lambd, T=num_tree, fair=fair, random_seed=seed)
    if base_type == 'ID3':
        classifier.fit(D=D, base_type='entropy')
    else:
        classifier.fit(D=D, base_type='gini')

    y_hat = classifier.predict(X_train)
    y_train[y_train == -1] = 0
    classifier_acc = binary_score(y_train, y_hat)
    classifier_fair = fair_binary_score(y_train, y_hat, S_train, fair)
    print(" ======== Model Performance for Train(lambda: {}, seed: {}) ======== ".format(lambd, seed))
    print("Accuracy: {:.2f}".format(classifier_acc))
    print("Fairness Metric: {:.2f}".format(classifier_fair))
    metrics = eval_binary_metrics(classifier, X_train, y_train, S_train)
    print(metrics)

    train_flag = (binary_score(y_train[S_train == 0], y_hat[S_train == 0], fair) <
                  binary_score(y_train[S_train == 1], y_hat[S_train == 1], fair))

    return classifier, classifier_acc, classifier_fair, train_flag


def test_balance(classifier, fair='acc', lambd=0.1, seed=1, filename='./pkl_data/adult_balance.pkl', turn=False):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    X_orig = data['X']
    y_orig = data['y']
    S_orig = data['S']

    if turn:
        S_orig = 1 - S_orig

    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X_orig, y_orig, S_orig, test_size=0.3,
                                                                         random_state=seed)

    y_hat = classifier.predict(X_test)
    classifier_acc = binary_score(y_test, y_hat)
    classifier_fair = fair_binary_score(y_test, y_hat, S_test, fair)
    classifier_fair_0 = binary_score(y_test[S_test == 0], y_hat[S_test == 0], fair)
    classifier_fair_1 = binary_score(y_test[S_test == 1], y_hat[S_test == 1], fair)
    print(" ======== Model Performance for Test (lambda: {}, seed: {}) ======== ".format(lambd, seed))
    print("Accuracy: {:.2f}".format(classifier_acc))
    print("Fairness Metric: {:.2f}".format(classifier_fair))

    metrics = eval_binary_metrics(classifier, X_test, y_test, S_test)
    print(metrics)
    print()

    return classifier_acc, classifier_fair, classifier_fair_0, classifier_fair_1
