import pandas as pd
import numpy as np
import argparse
import configparser
import statistics 
import math
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
from utils import get_args_and_config, get_time_series_paths, resample_zeros, resample, get_vocal_paths
from mir_eval.melody import raw_pitch_accuracy, raw_chroma_accuracy, voicing_false_alarm, voicing_recall, overall_accuracy, to_cent_voicing
from collections import defaultdict



def rpa(true_pitch, pitch, threshold):
    pitch_diff = np.absolute(pitch - true_pitch)
    voiced_mask = np.bitwise_and(true_pitch != 0.0, pitch != 0.0)
    return np.sum(pitch_diff[voiced_mask] < threshold) / np.sum(voiced_mask) * 100


def vrr(true_pitch, confidence, threshold):
    true_voiced = true_pitch > 0
    # print(len(true_pitch), len(true_pitch[true_voiced]))
    return (np.sum(confidence[true_voiced] > threshold) / np.sum(true_voiced)) * 100


def vfa(true_pitch, confidence, threshold):
    true_unvoiced = true_pitch  == 0
    # print(len(true_pitch), len(true_pitch[true_unvoiced]))
    return (np.sum(confidence[true_unvoiced] > threshold) / np.sum(true_unvoiced)) * 100


def flatten_col(col):
    return np.concatenate(col).ravel()


def add_voicing_and_cents(results_df, detector):
    ref_voicing, ref_cent, est_voicing, est_cent = to_cent_voicing(
            results_df['time'], results_df['pitch'], 
            results_df[f'time_{detector}'], results_df[f'pitch_{detector}']
        )
    return pd.Series(
            [results_df['file'], ref_voicing, ref_cent, est_voicing, est_cent], 
            index=['file', 'ref_voicing', 'ref_cent', 'est_voicing', 'est_cent']
        )


def main():
    _, conf = get_args_and_config()

    # Load labels
    labels = []

    for path in get_time_series_paths(conf['output_dir_label']):
        ts = np.loadtxt(path, delimiter=",")
        f_name = os.path.splitext(os.path.basename(path))[0]
        labels.append([f_name, ts[:,0], ts[:,1]])

    results_df = pd.DataFrame(labels, columns=['file', 'time', 'pitch'])

    # Load results
    detectors = ["spice", "pyin"]

    for detector in detectors:
        path = conf[f'{detector}_results_path']
        with open(path, 'rb') as f:
            df = pickle.load(f)

        results_df = results_df.join(df.set_index('file'), on='file')

    # Convert to cents
    dfs = []

    for df, detector in zip(df, detectors):
        df_cents = results_df.apply(lambda x: add_voicing_and_cents(x, detector), axis=1)
        dfs.append(df_cents)

    # Meauseres
    metrics_df = pd.DataFrame()
    sns.set_theme(style="ticks", palette="pastel")

    for df_cents, detector in zip(dfs, detectors):
        rwa = df_cents.apply(lambda row:
                raw_pitch_accuracy(row['ref_voicing'], row['ref_cent'], row['est_voicing'], row['est_cent']), axis=1)

        rwc = df_cents.apply(lambda row:
                raw_chroma_accuracy(row['ref_voicing'], row['ref_cent'], row['est_voicing'], row['est_cent']), axis=1)

        n = len(df_cents)
        algorithm = pd.Series([detector] * 2 * n, name='algorithm')
        metric = pd.Series(['RWA']*n + ['RWC']*n, name='metric')
        value = pd.Series(pd.concat([rwa, rwc]).to_list(), name='value')
        metrics_df = pd.concat([metrics_df, pd.concat([algorithm, metric, value], axis=1)])


    sns.boxplot(x="metric", y="value",
            hue="algorithm", palette=["m", "g"],
            data=metrics_df)
    sns.despine(offset=10, trim=True)
    plt.show()


        

if __name__ == "__main__":
    main()
 





        