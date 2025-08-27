import pickle
from data_utils import *
import numpy as np

"""
    Reference:https://github.com/haewon55/FairMIPForest/
"""


def load_hsls():
    input_file = "./dataset/HSLS/hsls_orig.pkl"
    with open(input_file, 'rb') as handle:
        data = pickle.load(handle)

    data = data.dropna(axis=0)
    return data


def load_and_balance_hsls(balance=True):
    df = load_hsls()

    if balance:
        # df = balance_data(df, "gradebin", 1)
        df = balance_data(df, "racebin", 1)

    df_to_pickle(df, "racebin", filename='./pkl_data/hsls_balance_race.pkl')

    return


if __name__ == "__main__":
    load_hsls()
