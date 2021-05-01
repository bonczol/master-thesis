import numpy as np
import os
import configparser
import pandas as pd
import pickle 
from plots import read_label, add_voicing_and_cents
from utils import get_time_series_paths
from mir_eval.melody import raw_pitch_accuracy
from evaluate import evaluate


def main():
    dataset = 'MIR-1k'
    tracker = 'YIN'

    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read('consts.conf')
    conf = conf[dataset]

    labels = map(read_label, get_time_series_paths(conf['output_dir_label']))
    labels_df = pd.DataFrame(labels, columns=['file', 'label_time', 'label_pitch'])

    thresholds = np.arange(0.1, 1, 0.05)
    rpa = []

    for t in thresholds:
        evaluate(dataset, tracker, conf['output_dir_wav'], conf['root_results_dir'], t)

        with open(os.path.join(conf['root_results_dir'], f'{dataset}_{tracker}.pkl'), 'rb') as f:
            df = pickle.load(f)

        df['method'] = tracker 
        df = labels_df.join(df.set_index('file'), on='file')

        df = df.apply(lambda x: add_voicing_and_cents(x, conf), axis=1)
        df["RPA"] = df[['ref_voicing', 'ref_cent', 'est_voicing', 'est_cent']].apply(
                lambda r: raw_pitch_accuracy(r[0], r[1], r[2], r[3]), raw=True, axis=1)
        m = df['RPA'].mean()
        rpa.append(m)
        print(t, m)
    
    print(thresholds[np.argmax(rpa)], np.max(rpa))

    
        




main()