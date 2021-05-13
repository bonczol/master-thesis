import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import pickle
import math
from utils import get_wav_paths, get_parser_and_config
from scipy.io import wavfile
from tqdm import tqdm
from trackers import Spice, Crepe, Yin, InverseTracker


SR = 16000
MAX_ABS_INT16 = 32768.0


def get_waveform(path):
    sr, waveform = wavfile.read(path, 'rb')
    waveform = waveform / float(MAX_ABS_INT16)
    waveform = waveform.astype(dtype=np.float32)
    return sr, waveform


def evaluate(tracker_name, wav_dir_path, results_dir_path, yin_thresh=0.8, hf0_offset=0, wavs_num=None):
    wav_paths = get_wav_paths(wav_dir_path)
    rows = []

    # Init
    if tracker_name == 'SPICE':
        tracker = Spice()
    elif tracker_name == 'CREPETINY':
        tracker = Crepe('tiny', SR)
    elif tracker_name == 'YIN':
        tracker = Yin(SR, yin_thresh)
    elif tracker_name == "DDSPINV":
        tracker = InverseTracker('mdb_stem_synth_ckpt')
    else:
        raise Exception(f'{tracker_name} not found')


    if wavs_num is None:
        wavs_num = len(wav_paths)

    # Eval
    for path in tqdm(wav_paths):
        read_sr, waveform = get_waveform(path)

        if SR != read_sr:
            raise Exception('Sampling rate missmatch')
    
        time, pitch_pred, confidence_pred, evaluation_time = tracker.predict(waveform)

        f_name = os.path.splitext(os.path.basename(path))[0]
        rows.append([f_name, time, pitch_pred, confidence_pred, evaluation_time])   
    
    dataset = os.path.basename(os.path.split(wav_dir_path)[0])
    folder_parts = os.path.basename(wav_dir_path).split("_")

    if len(folder_parts) > 1:
        _, version, snr = folder_parts
        filename = f'{dataset}_{tracker_name}_{version}_{snr}.pkl'
    else:
        filename = f'{dataset}_{tracker_name}.pkl'

    save_path = os.path.join(results_dir_path, filename)
    with open(save_path, 'wb') as f:
        df = pd.DataFrame(rows, columns=["file", "time", "pitch", "confidence", "evaluation_time"])
        pickle.dump(df, f)


def main():
    # tf.config.set_visible_devices([], 'GPU')
    parser, conf = get_parser_and_config()
    parser.add_argument('net', type=str)
    parser.add_argument('--noise', type=str)
    parser.add_argument('--snr', type=int)
    args = parser.parse_args()
    conf = conf[args.ds_name]
    
    input_dir = conf['processed_wav_dir']
    if args.noise and args.snr:
        input_dir = f'{input_dir}_{args.noise}_{args.snr}'

    evaluate(args.net, input_dir, conf['results_dir'])



if __name__ == "__main__": 
    main()

    

    




    



