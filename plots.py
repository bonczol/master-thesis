import pandas as pd
import numpy as np
import argparse
import configparser
import statistics 
import math


def pitch_diff(pitch1, pitch2):
    return np.absolute(12 * np.log2(pitch2 / pitch1))


def rpa(true_pitch, pitch, threshold):
    voiced_mask = true_pitch != 0.0
    tpc_fnc = np.sum((pitch[voiced_mask] < threshold))
    return tpc_fnc / np.sum(voiced_mask)


def vrr(true_pitch, confidence, threshold):
    true_voiced = true_pitch != 0.0
    return np.sum(confidence[true_voiced] > threshold) / np.sum(true_voiced)


def vfa(true_pitch, confidence, threshold):
    true_unvoiced = true_pitch == 0.0
    return np.sum(confidence[true_unvoiced] > threshold) / np.sum(true_unvoiced)


def calc_mean_and_interval(data):
    mean = 100 * np.sum(data) / len(data)
    interval = 100 * 2 * 1.960 * (np.std(data) / np.sqrt(len(data)))
    return mean, interval


def rpa_on_dataset(df, threshold):
    rpas = df[["True pitch", "Pitch difference"]].apply(lambda x: rpa(x[0], x[1], threshold), axis=1).to_numpy()
    return calc_mean_and_interval(rpas)


def vrr_vfa_on_dataset(df, threshold):
    vrrs = df[["True pitch", "Confidence"]].apply(lambda x: vrr(x[0], x[1], threshold), axis=1).to_numpy()
    vfas = df[["True pitch", "Confidence"]].apply(lambda x: vfa(x[0], x[1], threshold), axis=1).to_numpy()
    return calc_mean_and_interval(vrrs), calc_mean_and_interval(vfas)


def str_to_numpy(string):
    return np.array(eval(string), dtype=np.float)


def vrr_new(df):
    total_gv = 0
    total_tpc_fnc = 0

    for _, row in df.iterrows():
        voiced_mask = row["True pitch"] != 0.0
        gv = sum(voiced_mask)
        tpc_fnc = np.sum(row["Confidence"][voiced_mask] > 0.90)
        total_gv += gv
        total_tpc_fnc += tpc_fnc

    return total_tpc_fnc / total_gv * 100


# def cumulative_density(df):



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_name', type=str)
    args = parser.parse_args()
    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read('consts.conf')
    ds_conf = conf[args.ds_name]

    df = pd.read_csv(ds_conf["results_path"], converters={'True pitch': str_to_numpy, 'Pitch': str_to_numpy, 'Confidence': str_to_numpy})
    
    df["Pitch difference"] = df[["True pitch", "Pitch"]].apply(lambda x: pitch_diff(x[0], x[1]), axis=1)
    # print(df["Pitch difference"].head(5))

#    df["Pitch difference"].to_numpy()



    # VRR and VFA test
    # for i in np.arange(0.8, 1, 0.005):
    #     print(i, vrr_vfa_on_dataset(df, i))



if __name__ == "__main__":
    main()






        