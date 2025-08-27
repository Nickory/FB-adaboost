from scipy.stats import distributions


def main():
    N = 9
    k = 6

    # avg_rank = [12 / 9, 22 / 9, 51 / 9, 35 / 9, 30 / 9, 39 / 9]  # Accuracy
    avg_rank = [46 / 9, 22 / 9, 41 / 9, 20 / 9, 38 / 9, 22 / 9]  # Fairness loss

    r = 0

    for i in range(len(avg_rank)):
        r += avg_rank[i] ** 2

    q = (12 * N / (k * (k + 1))) * (r - k * (k + 1) ** 2 / 4)
    print("Q = {}".format(q))
    p_chi = distributions.chi2.sf(q, k - 1)
    print("p_value_chi = {}".format(p_chi))
    f = (N - 1) * q / (N * (k - 1) - q)
    print("F = {}".format(f))
    p_f = distributions.f.sf(f, k - 1, (k - 1) * (N - 1))
    print("p_value_f = {}".format(p_f))

    return f


if __name__ == "__main__":
    main()
