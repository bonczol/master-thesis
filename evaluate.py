import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import argparse
import configparser
from pydub import AudioSegment
from utils import get_wav_paths, get_args_and_config, semitones2hz
from multiprocessing import Pool
from scipy.io import wavfile
import  pickle

MAX_ABS_INT16 = 32768.0

def predict(model, audio):
    model_output = model.signatures["serving_default"](tf.constant(audio, tf.float32))
    return output2hz(model_output["pitch"]), 1 - model_output["uncertainty"]


def output2hz(pitch_output):
    PT_OFFSET = 25.58
    PT_SLOPE = 63.07
    cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
    return semitones2hz(cqt_bin)


def get_waveform(path):
    rate, waveform = wavfile.read(path, 'rb')
    duration = len(waveform) / float(rate)
    waveform = waveform / float(MAX_ABS_INT16)
    return duration, waveform


def main():
    # tf.config.set_visible_devices([], 'GPU')
    _, conf = get_args_and_config()

    model = hub.load("https://tfhub.dev/google/spice/2")
    wav_paths = get_wav_paths(conf["output_dir_wav"])
    rows = []

    for path in wav_paths:
        duration, waveform = get_waveform(path)
        time = np.arange(0, duration + 0.0001, 0.032)

        pitch_pred, confidence_pred = predict(model, waveform)

        f_name = os.path.splitext(os.path.basename(path))[0]
        rows.append([f_name, time, pitch_pred.numpy(), confidence_pred.numpy()])   

    with open(conf["spice_results_path"], 'wb') as f:
        df = pd.DataFrame(rows, columns=["file", "time", "pitch", "confidence"])
        pickle.dump(df, f)


if __name__ == "__main__": 
    main()

    

    




    



