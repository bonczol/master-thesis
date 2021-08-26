import numpy as np
import pandas as pd
import pickle
import consts
import mir_eval.transcription as mir_trans
from multiprocessing import Pool
from scipy.io import wavfile
from tqdm import tqdm
from itertools import repeat, product
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
    return [wav_path.stem] + list(result)


def evaluate_dir(method, wav_paths):
    if method.is_multicore:
        with Pool() as pool:
            return pool.starmap(evaluate, zip(repeat(method), wav_paths))
    else:
        return [evaluate(method, wav_path) for wav_path in tqdm(wav_paths)]


# def tune(method, dataset, noise=None, snr=None):
#     amplitude_t = [0.01, 0.03, 0.05, 0.08, 0.1]
#     confidence_t = np.arange(0.5, 0.72, 0.02)
#     pitch_dist_t = [50, 60, 80, 100]
#     rms_t = [-1.5, -2, -2.5]
    
#     with open(dataset.get_result(method.method.value, noise, snr), 'rb') as f:
#             results_df = pickle.load(f).set_index('file', drop=False)

#     results = []
#     best_score = 0
#     best_params = None

#     for at, ct, pt, rT in product(amplitude_t, confidence_t, pitch_dist_t, rms_t):
#         for wav_path in wav_paths:
#             row = results_df.loc[wav_path.stem, :]
#             _, waveform = get_waveform(wav_path)
#             result = method.transcribe(row.pitch, row.confidence, waveform)
#             results.append([wav_path.stem] + list(result))
        
#         results_df = pd.DataFrame(results, columns=["file", "est_note_interval", "est_note_pitch"])
#         score = mir_trans.evaluate()


def evaluate_dir_seg(method, dataset, noise=None, snr=None):
    wav_paths = dataset.get_wavs(noise, snr)
    results = []

    if method.method == Method.CREPE_MIDI:
        with open(dataset.get_result(method.method.value, noise, snr), 'rb') as f:
            results_df = pickle.load(f).set_index('file', drop=False)
        for wav_path in wav_paths:
            row = results_df.loc[wav_path.stem, :]
            _, waveform = get_waveform(wav_path)
            result = method.transcribe(row.pitch, row.confidence, waveform)
            results.append([wav_path.stem] + list(result))
    elif method.method == Method.PYIN_MIDI:
        for wav_path in wav_paths:
            results.append([wav_path.stem] + list(method.transcribe(wav_path)))

    return results 


def run_evaluation(method, dataset, noise=None, snr=None, notes=False):
    print(f'Method={method.method.value} Dataset={dataset.name} Noise={noise} SNR={snr} Notes={notes}')

    if notes:
        results = evaluate_dir_seg(method, dataset, noise, snr)
        save_path = dataset.get_result_notes(method.method.value, noise, snr)
        columns=["file", "est_note_interval", "est_note_pitch"]
    else:
        results = evaluate_dir(method, dataset.get_wavs(noise, snr))
        save_path = dataset.get_result(method.method.value, noise, snr)
        columns=["file", "time", "pitch", "confidence", "evaluation_time"]

    with open(save_path, 'wb') as f:
        df = pd.DataFrame(results, columns=columns)
        pickle.dump(df, f)
