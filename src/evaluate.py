import numpy as np
import pandas as pd
import pickle
import consts
from multiprocessing import Pool
from scipy.io import wavfile
from tqdm import tqdm
from itertools import repeat
from method import Method


def get_waveform(path):
    sr, waveform = wavfile.read(path, 'rb')
    if consts.SR != sr:
        raise Exception('Sampling rate missmatch')
    waveform = waveform / float(consts.MAX_ABS_INT16)
    waveform = waveform.astype(dtype=np.float32)
    return sr, waveform


def evaluate(method, wav_path):
    if method.method in [Method.PYIN, Method.PYIN_MIDI]:
        result = method.predict(wav_path)
    else:
        _, waveform = get_waveform(wav_path)
        result = method.predict(waveform)
    return [wav_path.name] + list(result)


def evaluate_dir(tracker, wav_paths):
    if tracker.is_multicore:
        with Pool() as pool:
            return pool.starmap(evaluate, zip(repeat(tracker), wav_paths))
    else:
        return [evaluate(tracker, wav_path) for wav_path in tqdm(wav_paths)]


def run_evaluation(method, dataset, noise=None, snr=None, notes=False):
    print(f'Method={method.method.value} Dataset={dataset.name} Noise={noise} SNR={snr} Notes={notes}')
    results = evaluate_dir(method, dataset.get_wavs(noise, snr)[:3])

    if notes:
        save_path = dataset.get_result_notes(method.method.value, noise, snr)
        columns=["file", "interval", "pitch"]
    else:
        save_path = dataset.get_result(method.method.value, noise, snr)
        columns=["file", "time", "pitch", "confidence", "evaluation_time"]

    with open(save_path, 'wb') as f:
        df = pd.DataFrame(results, columns=columns)
        pickle.dump(df, f)
