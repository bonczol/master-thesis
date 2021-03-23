import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import argparse
import configparser
from pydub import AudioSegment
from utils import load_waveforms_and_labels
from scipy.io import wavfile

MAX_ABS_INT16 = 32768.0

def predict(model, audio):
    model_output = model.signatures["serving_default"](tf.constant(audio, tf.float32))
    return output2hz(model_output["pitch"]), 1 - model_output["uncertainty"]

def output2hz(pitch_output):
    PT_OFFSET = 25.58
    PT_SLOPE = 63.07
    FMIN = 10.0
    BINS_PER_OCTAVE = 12.0
    cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
    return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)

def get_waveform(path):
    _, waveform = wavfile.read(path, 'rb')
    return waveform / float(MAX_ABS_INT16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_name', type=str)
    args = parser.parse_args()
    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read('consts.conf')
    ds_conf = conf[args.ds_name]

    model = hub.load("https://tfhub.dev/google/spice/2")
    wav_paths, label_paths = load_waveforms_and_labels(ds_conf["output_dir_wav"], ds_conf["output_dir_label"])
    results = []

    for wav_path, label_path in zip(wav_paths, label_paths):
        waveform = get_waveform(wav_path)
        label = np.loadtxt(label_path, delimiter=",")
        pitch_pred, confidence_pred = predict(model, waveform)
        f_name = os.path.splitext(os.path.basename(wav_path))[0]
            
        results.append([f_name, list(label[:,1]), list(pitch_pred.numpy()), list(confidence_pred.numpy())])
    
    results_pd = pd.DataFrame(results, columns=["File", "True pitch", "Pitch", "Confidence"]) 
    results_pd.to_csv(ds_conf["results_path"])


if __name__ == "__main__": 
    main()

    

    




    



