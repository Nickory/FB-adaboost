import matplotlib.pyplot as plt
import numpy as np
import scikit_posthocs as sp

"""
    Reference: https://blog.csdn.net/a1920993165/article/details/124910139
"""


def calculate_p_value():
    # accuracy
    data_1 = [1, 1, 1, 2, 1, 1, 3, 1, 1]  # AdaBoost
    data_2 = [2, 2, 2, 1, 4, 5, 2, 2, 2]  # FAB
    data_3 = [6, 6, 6, 6, 5, 4, 6, 6, 6]  # Hardt
    data_4 = [3, 3, 4, 5, 3, 3, 5, 5, 4]  # Agarwal-Exp
    data_5 = [4, 4, 3, 4, 2, 2, 4, 4, 3]  # Agarwal-Grid
    data_6 = [5, 5, 5, 3, 6, 6, 1, 3, 5]  # Zafar

    # # fairness loss
    # data_1 = [6, 6, 6, 3, 6, 6, 1, 6, 6]  # AdaBoost
    # data_2 = [1, 2, 3, 2, 3, 3, 2, 3, 3]  # FAB
    # data_3 = [5, 5, 4, 6, 4, 2, 5, 5, 5]  # Hardt
    # data_4 = [2, 1, 2, 5, 2, 1, 3, 2, 2]  # Agarwal-Exp
    # data_5 = [3, 4, 5, 4, 5, 5, 4, 4, 4]  # Agarwal-Grid
    # data_6 = [4, 3, 1, 1, 1, 4, 6, 1, 1]  # Zafar

    data = np.array([data_1, data_2, data_3, data_4, data_5, data_6])

    print(np.mean(data, axis=1))

    print(sp.posthoc_nemenyi_friedman(data.T))

    return


def visualize():

    k = 6
    n = 9
    q = [1.960, 2.344, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]

    # avg_rank = [12 / 9, 22 / 9, 51 / 9, 35 / 9, 30 / 9, 39 / 9]  # Accuracy
    avg_rank = [46 / 9, 22 / 9, 41 / 9, 20 / 9, 38 / 9, 22 / 9]  # Fairness loss

    y = [1, 2, 3, 4, 5, 6]

    CD = q[k-2] * (np.sqrt((k * (k + 1) / (6 * n))))
    print("CD = {}".format(CD))
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_rank, y, s=100, c='black')
    for i in range(len(y)):
        yy = [y[i], y[i]]
        xx = [avg_rank[i] - CD / 2, avg_rank[i] + CD / 2]
        plt.plot(xx, yy, linewidth=3.0)

    plt.yticks(range(0, 8, 1), labels=['', 'AdaBoost', 'FAB', 'Hardt', 'Agarwal-Exp', 'Agarwal-Grid', 'Zafar', ''], size=20)
    plt.xticks(range(0, 7, 1), labels=['0', '1', '2', '3', '4', '5', '6'], size=20)

    plt.xlabel("Rank", size=20)
    plt.show()

    return


def main():
    calculate_p_value()
    visualize()
    return


if __name__ == "__main__":
    main()
