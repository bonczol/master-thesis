import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import math
import argparse
from pydub import AudioSegment
from scipy.io import wavfile

FILE_DIR = "Wavfile"
LABEL_DIR = "PitchLabel"


def evaluate(wav_ds, ds_name):
    model = hub.load("https://tfhub.dev/google/spice/2")
    results = pd.DataFrame(columns=["Pitch", "Confidence", "Label"]) 

    for audio, label in wav_ds.take(1):
        pitch, conf = predict(model, audio)
        results.append(pitch, conf, label)
    
    results.to_csv(f"./Results/{ds_name}")

def predict(model, audio):
    model_output = model.signatures["serving_default"](audio)
    return model_output["pitch"], 1 - model_output["uncertainty"]
    

def get_waveform_and_label(file_path, label_path):
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  label = np.loadtxt(label_path)
  return waveform, label


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label_path(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    parts[-2] = LABEL_DIR
    parts[-1] = parts[-1].split(".")[0] + ".txt"
    return os.path.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, required=True)
    args = parser.parse_args()

    files_dir = os.path.join("dataset", args.dataset, FILE_DIR)

    file_names = tf.io.gfile.glob(files_dir)
    labels_names = [get_label_path(fn) for fn  in file_names]

    wav_ds = tf.data.Dataset.from_tensor_slices(zip(file_names, labels_names))
    wav_ds = wav_ds.map(
        lambda x: get_waveform_and_label(x[0], x[1]), 
        num_parallel_calls=tf.data.AUTOTUNE
        )

    evaluate(wav_ds, args.dataset)

    

    




    



