import pandas as pd
import numpy as np
import argparse
import configparser
import statistics 
import math
import matplotlib.pyplot as plt
import pickle
from mir_eval.melody import to_cent_voicing, raw_pitch_accuracy, evaluate



def rpa(true_pitch, pitch, threshold):
    pitch_diff = np.absolute(pitch - true_pitch)
    voiced_mask = np.bitwise_and(true_pitch != 0.0, pitch != 0.0)
    return np.sum(pitch_diff[voiced_mask] < threshold) / np.sum(voiced_mask) * 100


def vrr(true_pitch, confidence, threshold):
    true_voiced = true_pitch != 0.0 
    return np.sum(confidence[true_voiced] > threshold) / np.sum(true_voiced) * 100


def vfa(true_pitch, confidence, threshold):
    true_unvoiced = true_pitch == 0.0
    return np.sum(confidence[true_unvoiced] > threshold) / np.sum(true_unvoiced) * 100


def flatten_col(col):
    return np.concatenate(col).ravel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_name', type=str)
    args = parser.parse_args()
    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read('consts.conf')
    ds_conf = conf[args.ds_name]

    with open(ds_conf["results_path"], 'rb') as f:
        df = pickle.load(f)

    true_pitch = flatten_col(df['true_pitch'])
    pitch = flatten_col(df['pitch'])
    confidence = flatten_col(df['confidence'])

    rpa_res1 = rpa(true_pitch, pitch, 0.5)
    print(rpa_res1)

    for i in np.arange(0.8, 1, 0.005):
        print(i, vrr(true_pitch, confidence, i), vfa(true_pitch, confidence, i))

    
    # df["Pitch difference"] = df[["True pitch", "Pitch"]].apply(lambda x: pitch_diff(x[1], x[0]), axis=1)

    # print(df["Pitch difference"].isna().sum())
    # print(df["Pitch difference"].head(5))

    # ax = pd.DataFrame([i for i in df["Pitch difference"].iloc[1] if i != np.inf]).plot.hist(cumulative=True, density=100, bins=300, histtype='step')
    # plt.show()

if __name__ == "__main__":
    main()
 





        