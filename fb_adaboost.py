from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.model_selection import train_test_split
from estimate import *


class FBAdaBoost:
    """FairBalance AdaBoost classifier.

    Parameters
    ----------
    X : np.ndarray
        Training features.
    y : np.ndarray
        Binary labels encoded as {0,1}. Internally converted to {+1,-1}.
    S : np.ndarray
        Sensitive attribute encoded as {0,1}.
    lambd : float
        Trade-off parameter between accuracy and fairness.
    T : int, optional
        Number of boosting rounds.
    alpha : float, optional
        Weight for balancing sensitive groups when initialising sample weights.
    beta : float, optional
        Weight for balancing class distribution when initialising sample weights.
    random_seed : int, optional
        Random seed for base learners.
    """

    def __init__(self, X, y, S, lambd, T=100, alpha=1.0, beta=1.0, random_seed=42):
        self.X = X
        y[y == 0] = -1
        self.y = y.reshape(-1, 1)
        self.S = S
        self.lambd = lambd
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.random_seed = random_seed

        self.sample_num = self.X.shape[0]
        self.classifiers = []
        self.classifiers_weights = []

        # counts for sensitive groups and classes
        N0 = np.sum(self.S == 0)
        N1 = np.sum(self.S == 1)
        N_pos = np.sum(self.y.ravel() == 1)
        N_neg = np.sum(self.y.ravel() == -1)
        N = float(self.sample_num)

        # initial weights
        w = np.ones(self.sample_num) / N
        if N0 + N1 > 0:
            w += self.alpha * abs(N0 - N1) / N * (
                (self.S == 0) / (N0 + 1) - (self.S == 1) / (N1 + 1)
            )
        if N_pos + N_neg > 0:
            w += self.beta * abs(N_pos - N_neg) / N * (
                (self.y.ravel() == 1) / (N_pos + 1) - (self.y.ravel() == -1) / (N_neg + 1)
            )
        w = np.clip(w, 0, None)
        self.sample_weights = w / np.sum(w)

    def fit(self, D=3, base_type='entropy'):
        for t in range(self.T):
            base_classifier = DecisionTreeClassifier(
                max_depth=D, criterion=base_type, random_state=self.random_seed
            ).fit(self.X, self.y, sample_weight=self.sample_weights)
            y_pred = base_classifier.predict(self.X)
            miscls = (y_pred != self.y.ravel()).astype(float)
            error = np.sum(self.sample_weights * miscls)

            if error <= 0 or error >= 0.5:
                break

            err_s0 = np.sum(self.sample_weights[(self.S == 0)] * miscls[(self.S == 0)])
            err_s1 = np.sum(self.sample_weights[(self.S == 1)] * miscls[(self.S == 1)])
            N0 = np.sum(self.S == 0)
            N1 = np.sum(self.S == 1)
            if N0 == 0 or N1 == 0:
                Delta = 0.0
            else:
                Delta = err_s0 / N0 - err_s1 / N1

            denom = error - self.lambd * abs(Delta)
            if denom <= 0:
                break
            alpha_t = 0.5 * np.log((1 - error + self.lambd * abs(Delta)) / denom)

            self.classifiers.append(base_classifier)
            self.classifiers_weights.append(alpha_t)

            self.sample_weights *= np.exp(-alpha_t * self.y.ravel() * y_pred)
            self.sample_weights /= np.sum(self.sample_weights)

        self.classifiers_weights = np.array(self.classifiers_weights)

    def predict(self, X):
        if not self.classifiers:
            return np.zeros(X.shape[0])
        votes = np.array([clf.predict(X) for clf in self.classifiers])
        agg = np.sign(np.dot(self.classifiers_weights, votes))
        agg[agg == -1] = 0
        return agg


def train_balance(D=3, num_tree=20, input_file='./pkl_data/adult_balance.pkl', seed=1,
                  lambd=0.1, alpha=1.0, beta=1.0, turn=False, base_type='ID3'):
    with open(input_file, 'rb') as handle:
        data = pickle.load(handle)

    X_orig = data['X']
    y_orig = data['y']
    S_orig = data['S']

    if turn:
        S_orig = 1 - S_orig

    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
        X_orig, y_orig, S_orig, test_size=0.3, random_state=seed
    )

    classifier = FBAdaBoost(X=X_train, y=y_train, S=S_train, lambd=lambd,
                           T=num_tree, alpha=alpha, beta=beta, random_seed=seed)
    if base_type == 'ID3':
        classifier.fit(D=D, base_type='entropy')
    else:
        classifier.fit(D=D, base_type='gini')

    y_hat = classifier.predict(X_train)
    y_train[y_train == -1] = 0
    classifier_acc = binary_score(y_train, y_hat)
    classifier_fair = fair_binary_score(y_train, y_hat, S_train, 'acc')

    metrics = eval_binary_metrics(classifier, X_train, y_train, S_train)

    train_flag = (
        binary_score(y_train[S_train == 0], y_hat[S_train == 0], 'acc') <
        binary_score(y_train[S_train == 1], y_hat[S_train == 1], 'acc')
    )

    return classifier, classifier_acc, classifier_fair, train_flag


def test_balance(classifier, lambd=0.1, seed=1, filename='./pkl_data/adult_balance.pkl',
                 alpha=1.0, beta=1.0, turn=False):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    X_orig = data['X']
    y_orig = data['y']
    S_orig = data['S']

    if turn:
        S_orig = 1 - S_orig

    X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
        X_orig, y_orig, S_orig, test_size=0.3, random_state=seed
    )

    y_hat = classifier.predict(X_test)
    classifier_acc = binary_score(y_test, y_hat)
    classifier_fair = fair_binary_score(y_test, y_hat, S_test, 'acc')
    classifier_fair_0 = binary_score(y_test[S_test == 0], y_hat[S_test == 0], 'acc')
    classifier_fair_1 = binary_score(y_test[S_test == 1], y_hat[S_test == 1], 'acc')

    metrics = eval_binary_metrics(classifier, X_test, y_test, S_test)

    return classifier_acc, classifier_fair, classifier_fair_0, classifier_fair_1
