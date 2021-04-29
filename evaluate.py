import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from utils import get_wav_paths, get_parser_and_config, semitones2hz
from multiprocessing import Pool
from scipy.io import wavfile
import  pickle
import crepe.crepe as crepe
import librosa
from tqdm import tqdm
import aubio


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
    sr, waveform = wavfile.read(path, 'rb')
    duration = len(waveform) / float(sr)
    waveform = waveform / float(MAX_ABS_INT16)
    waveform = waveform.astype(dtype=np.float32)
    return sr, waveform


def main():
    # tf.config.set_visible_devices([], 'GPU')
    parser, conf = get_parser_and_config()
    parser.add_argument('net', type=str)
    args = parser.parse_args()
    conf = conf[args.ds_name]

    if args.net == 'SPICE':
        model = hub.load("https://tfhub.dev/google/spice/2")
    elif args.net == 'CREPE_TINY':
        model = crepe.build_and_load_model('tiny')

    wav_paths = get_wav_paths(conf["output_dir_wav"])
    rows = []

    for path in tqdm(wav_paths):
        sr, waveform = get_waveform(path)
        duration = len(waveform) / float(sr)
        time = np.arange(0, duration + 0.0001, 0.032)

        if args.net == 'SPICE':
            pitch_pred, confidence_pred = predict(model, waveform)
            pitch_pred, confidence_pred = pitch_pred.numpy(), confidence_pred.numpy()
        elif args.net == 'CREPE_TINY':
            _, pitch_pred, confidence_pred, _ = crepe.predict(waveform, sr, model, step_size=32, verbose=0)
        elif args.net == 'YIN':
            pitch_o = aubio.pitch("yinfft", 2048, 512, sr)
            pitch_pred = np.zeros(time.shape)
            confidence_pred = np.ones(time.shape)

            for i in range(len(waveform) // 512):
                sample = waveform[(i*512):(i*512)+512]
                pitch_pred[i] = pitch_o(sample)[0]
                confidence_pred[i] = pitch_o.get_confidence()
            cmin = confidence_pred.min()
            cmax = confidence_pred.max()

            confidence_pred = 1 - (confidence_pred - cmin) / (cmax - cmin)
        

        f_name = os.path.splitext(os.path.basename(path))[0]
        rows.append([f_name, time, pitch_pred, confidence_pred])   

    save_path = os.path.join(conf['root_results_dir'], f'{args.ds_name}_{args.net}.pkl')
    with open(save_path, 'wb') as f:
        df = pd.DataFrame(rows, columns=["file", "time", "pitch", "confidence"])
        pickle.dump(df, f)


if __name__ == "__main__": 
    main()

    

    




    



