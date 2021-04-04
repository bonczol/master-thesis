import os
import re
import argparse
import configparser


def load_waveforms_and_labels(wav_dir, labels_dir):
    wav_files = sorted([os.path.join(wav_dir, f) for f in  os.listdir(wav_dir) if not f.startswith(".") and f.endswith(".wav")])
    labels_files = sorted([os.path.join(labels_dir, f) for f in  os.listdir(labels_dir) if not f.startswith(".") and re.match(r".*\.(pv|csv)$", f)])

    if len(wav_files) != len(labels_files):
        raise Exception("Number of .wav files different than number of label files")

    return wav_files, labels_files


def get_wav_paths(wav_dir):
    return sorted([os.path.join(wav_dir, f) for f in  os.listdir(wav_dir) 
                        if not f.startswith(".") and f.endswith(".wav")])


def get_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_name', type=str)
    args = parser.parse_args()
    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read('consts.conf')
    ds_conf = conf[args.ds_name]
    return args, ds_conf