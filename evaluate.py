import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pickle
import aubio
import crepe.crepe as crepe
import matlab.engine
import time
from utils import get_wav_paths, get_parser_and_config, semitones2hz
from scipy.io import wavfile
from tqdm import tqdm


MAX_ABS_INT16 = 32768.0

def predict(model, audio):
    model_output = model.signatures["serving_default"](tf.constant(audio, tf.float32))
    return output2hz(model_output["pitch"]), 1 - model_output["uncertainty"]


def predict_yin(yin, audio, n):
    pitch_pred = np.zeros(n)
    confidence_pred = np.zeros(n)

    for i in range(0, len(audio) // 512):
        sample = audio[(i*512):(i*512)+512]
        pitch_pred[i] = yin(sample)[0]
        confidence_pred[i] = yin.get_confidence()

    cmin = confidence_pred.min()
    cmax = confidence_pred.max()

    confidence_pred = 1 - (confidence_pred - cmin) / (cmax - cmin + 0.0001)
    return pitch_pred, confidence_pred


def output2hz(pitch_output):
    PT_OFFSET = 25.58
    PT_SLOPE = 63.07
    cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
    return semitones2hz(cqt_bin)


def get_waveform(path):
    sr, waveform = wavfile.read(path, 'rb')
    duration = len(waveform) / float(sr)
    waveform = waveform / float(MAX_ABS_INT16)
    waveform = waveform.astype(dtype=np.float32)
    return sr, waveform


def evaluate(dataset, tracker_name, wav_dir_path, results_dir_path, yin_thresh=0.8):
    sr = 16000
    wav_paths = get_wav_paths(wav_dir_path)
    rows = []

    if tracker_name == 'SPICE':
        tracker = hub.load("https://tfhub.dev/google/spice/2")
    elif tracker_name == 'CREPE_TINY':
        tracker = crepe.build_and_load_model('tiny')
    elif tracker_name == 'YIN':
        tracker = aubio.pitch("yinfft", 1024, 512, sr)
        tracker.set_tolerance(yin_thresh)
    elif tracker_name == 'hf0':
        eng = matlab.engine.start_matlab()
        eng.cd('./hf0')
        eng.mirverbose(0)
        convnet = eng.getfield(eng.load('convModel.mat'), 'convnet')

    for path in tqdm(wav_paths):
        _, waveform = get_waveform(path)
        duration = len(waveform) / float(sr)
        model_time = np.arange(0, duration + 0.0001, 0.032)

        # Measure time of evaluation
        t_start = time.perf_counter()

        if tracker_name == 'SPICE':
            pitch_pred, confidence_pred = predict(tracker, waveform)
            pitch_pred, confidence_pred = pitch_pred.numpy(), confidence_pred.numpy()
        elif tracker_name == 'CREPE_TINY':
            _, pitch_pred, confidence_pred, _ = crepe.predict(waveform, sr, tracker, step_size=32, verbose=0)
        elif tracker_name == 'hf0':
            pitch_pred, hf0_time, hf0_load_time = eng.PitchExtraction(os.path.abspath(path), convnet, nargout=3)
        elif tracker_name == 'YIN':
            pitch_pred, confidence_pred = predict_yin(tracker, waveform, len(model_time))

        t_end = time.perf_counter()
        elapsed_time = t_end - t_start

        if tracker_name == 'hf0':
            elapsed_time -= hf0_load_time
            pitch_pred = np.array(pitch_pred).ravel()
            hf0_time = np.array(hf0_time).ravel() - 0.032
            pitch_pred = np.interp(model_time, hf0_time, pitch_pred)
            confidence_pred = np.array(pitch_pred > 0, dtype=np.int)
        
        f_name = os.path.splitext(os.path.basename(path))[0]
        rows.append([f_name, model_time, pitch_pred, confidence_pred])   

    save_path = os.path.join(results_dir_path, f'{dataset}_{tracker_name}.pkl')
    with open(save_path, 'wb') as f:
        df = pd.DataFrame(rows, columns=["file", "time", "pitch", "confidence"])
        pickle.dump(df, f)


def main():
    # tf.config.set_visible_devices([], 'GPU')
    parser, conf = get_parser_and_config()
    parser.add_argument('net', type=str)
    args = parser.parse_args()
    conf = conf[args.ds_name]
    evaluate(args.ds_name, args.net, conf["output_dir_wav"], conf['root_results_dir'])



if __name__ == "__main__": 
    main()

    

    




    



