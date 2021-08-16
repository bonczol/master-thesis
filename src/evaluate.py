import os
import numpy as np
import pandas as pd
import pickle
import consts
import time
from method import Tracker
from multiprocessing import Pool
from scipy.io import wavfile
from tqdm import tqdm
from itertools import repeat


def get_waveform(path):
    sr, waveform = wavfile.read(path, 'rb')
    waveform = waveform / float(consts.MAX_ABS_INT16)
    waveform = waveform.astype(dtype=np.float32)
    return sr, waveform


def evaluate(tracker, wav_path):
    if tracker.method == Tracker.PYIN:
        time, pitch_pred, confidence_pred, evaluation_time = tracker.predict(wav_path)
    else:
        read_sr, waveform = get_waveform(wav_path)
        if consts.SR != read_sr:
            raise Exception('Sampling rate missmatch')
        time, pitch_pred, confidence_pred, evaluation_time = tracker.predict(waveform)

    f_name = os.path.splitext(os.path.basename(wav_path))[0]
    return [f_name, time, pitch_pred, confidence_pred, evaluation_time]


def evaluate_dir(tracker, wav_paths):
    if tracker.is_multicore:
        with Pool(processes=1) as pool:
            return pool.starmap(evaluate, zip(repeat(tracker), wav_paths))
    else:
        return [evaluate(tracker, wav_path) for wav_path in tqdm(wav_paths)]


def run_evaluation(tracker, dataset, noise=None, snr=None):
    print(f'Tracker={tracker.method.value} Dataset={dataset.name} Noise={noise} SNR={snr}')
    results = evaluate_dir(tracker, dataset.get_wavs(noise, snr))
    save_path = dataset.get_result(tracker.method.value, noise, snr)

    with open(save_path, 'wb') as f:
        df = pd.DataFrame(results, columns=["file", "time", "pitch", "confidence", "evaluation_time"])
        pickle.dump(df, f)
