import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import pickle
import utils
from method import Method
from scipy.io import wavfile
from tqdm import tqdm
from trackers import Spice, Crepe, Yin, InverseTracker


SR = 16000
MAX_ABS_INT16 = 32768.0


def parse_tracker(method):
    if method == Method.SPICE:
        return Spice()
    elif method == Method.CREPE_TINY:
        return Crepe('tiny', SR)
    elif method == Method.YIN:
        return Yin(SR, 0.8)
    elif method == Method.DDSP_INV:
        return InverseTracker('mdb_stem_synth_ckpt')
    else:
        raise Exception(f'{method} not found')


def get_waveform(path):
    sr, waveform = wavfile.read(path, 'rb')
    waveform = waveform / float(MAX_ABS_INT16)
    waveform = waveform.astype(dtype=np.float32)
    return sr, waveform


def evaluate(tracker, wav_path):
    read_sr, waveform = get_waveform(wav_path)

    if SR != read_sr:
        raise Exception('Sampling rate missmatch')
    
    time, pitch_pred, confidence_pred, evaluation_time = tracker.predict(waveform)
    f_name = os.path.splitext(os.path.basename(wav_path))[0]

    return [f_name, time, pitch_pred, confidence_pred, evaluation_time]


def evaluate_dir(tracker, wav_dir_path):
    wav_paths = utils.get_wav_paths(wav_dir_path)
    return [evaluate(tracker, path) for path in tqdm(wav_paths)]


def run_evaluation(tracker , wav_dir_path, results_dir_path):
    results = evaluate_dir(tracker, wav_dir_path)

    dataset = os.path.basename(os.path.split(wav_dir_path)[0])
    folder_parts = os.path.basename(wav_dir_path).split("_")

    if len(folder_parts) > 1:
        _, version, snr = folder_parts
        filename = f'{dataset}_{tracker.method.value}_{version}_{snr}.pkl'
    else:
        filename = f'{dataset}_{tracker.method.value}.pkl'

    save_path = os.path.join(results_dir_path, filename)
    with open(save_path, 'wb') as f:
        df = pd.DataFrame(results, columns=["file", "time", "pitch", "confidence", "evaluation_time"])
        pickle.dump(df, f)


def main():
    # tf.config.set_visible_devices([], 'GPU')
    parser, conf = utils.get_parser_and_config()
    parser.add_argument('net', type=str)
    parser.add_argument('--noise', type=str)
    parser.add_argument('--snr', type=int)
    args = parser.parse_args()
    conf = conf[args.ds_name]
    
    input_dir = conf['processed_wav_dir']
    if args.noise and args.snr:
        input_dir = f'{input_dir}_{args.noise}_{args.snr}'

    tracker = parse_tracker(Method(args.net))

    run_evaluation(tracker, input_dir, conf['results_dir'])



if __name__ == "__main__": 
    main()

    

    




    



