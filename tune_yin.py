import numpy as np
import os
import configparser
import pandas as pd
import pickle 
from plots import add_voicing_and_cents
from mir_eval.melody import raw_pitch_accuracy
from evaluate import evaluate


def main():
    dataset = 'MDB-stem-synth'
    tracker = 'YIN'

    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read('consts.conf')
    conf = conf[dataset]

    with open(conf['processed_label_binary'], 'rb') as f:
        labels_df = pickle.load(f)

    thresholds = np.arange(0.1, 1, 0.05)
    rpa = []

    for t in thresholds:
        evaluate(tracker, conf['processed_wav_dir'], conf['results_dir'], t)

        with open(os.path.join(conf['results_dir'], f'{dataset}_{tracker}.pkl'), 'rb') as f:
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