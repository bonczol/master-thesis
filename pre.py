import numpy as np
import os
import argparse
import configparser
import re
from converters import MirConverter, MdbConverter
from pydub import AudioSegment


def convert_dataset(converter, conf):
    wav_files = sorted([f for f in  os.listdir(conf['input_dir_wav']) if not f.startswith(".") and f.endswith(".wav")])
    labels_files = sorted([f for f in  os.listdir(conf['input_dir_label']) if not f.startswith(".") and re.match(r".*\.(pv|csv)$", f)])

    if len(wav_files) != len(labels_files):
        raise Exception("Number of .wav files different than number of label files")

    for wav_name, labels_name in zip(wav_files, labels_files):
        audio = AudioSegment.from_file(os.path.join(conf['input_dir_wav'], wav_name))
        labels = np.loadtxt(os.path.join(conf['input_dir_label'], labels_name), delimiter=',')
        
        labels = converter.convert_label(labels, audio.duration_seconds)
        np.savetxt(
            os.path.join(conf['output_dir_label'], os.path.splitext(labels_name)[0] + ".csv"), 
            labels, delimiter=',', fmt='%1.6f'
        )

        audio = converter.convert_audio(audio)
        audio.export(os.path.join(conf['output_dir_wav'], wav_name), format="wav")


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

    convert_dataset(converter, ds_conf)


main()