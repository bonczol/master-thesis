import numpy as np
import os
import argparse
import configparser
import re
from utils import load_waveforms_and_labels
from converters import MirConverter, MdbConverter
from multiprocessing import Pool
from pydub import AudioSegment
import time


def convert_example(wav_name, labels_name, converter, conf):
    audio = AudioSegment.from_file(wav_name)
    labels = np.loadtxt(labels_name, delimiter=',')
    
    labels = converter.convert_label(labels, audio.duration_seconds)
    np.savetxt(
        os.path.join(conf['output_dir_label'], os.path.splitext(os.path.basename(labels_name))[0] + ".csv"), 
        labels, delimiter=',', fmt='%1.6f'
    )

    audio = converter.convert_audio(audio)
    audio.export(os.path.join(conf['output_dir_wav'], os.path.basename(wav_name)), format="wav")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_name', type=str)
    args = parser.parse_args()
    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read('consts.conf')
    ds_conf = conf[args.ds_name]

    if args.ds_name == "MIR-1k":
        converter = MirConverter()
    elif args.ds_name == "MDB-stem-synth":
        converter = MdbConverter()
    else:
        converter = MirConverter()

    print(type(converter).__name__)
    wav_paths, labels_paths = load_waveforms_and_labels(ds_conf['input_dir_wav'], ds_conf['input_dir_label'])

    with Pool() as pool:
        n = len(wav_paths)
        pool.starmap(convert_example, zip(wav_paths, labels_paths, [converter] * n, [ds_conf] * n))


if __name__ == '__main__':
    main()