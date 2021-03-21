import numpy as np
import os
import argparse
import configparser
import re
from pydub import AudioSegment

EXPECTED_SAMPLE_RATE = 16000


def convert_audio(audio):
    audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE)
    return audio.split_to_mono()[1]


def convert_labels(labels, duration, start_time, time_step):
    t_model = np.arange(0, duration + 0.0001, 0.032)
    t = np.arange(start_time, duration - (time_step - 0.00001),  time_step)
    return  np.interp(t_model, t, labels)


def convert_dataset(conf):
    wav_files = sorted([f for f in  os.listdir(conf['input_dir_wav']) if f.endswith(".wav")])
    labels_files = sorted([f for f in  os.listdir(conf['input_dir_label']) if re.match(r'^\w+.(pv|csv)$', f)])

    if len(wav_files) != len(labels_files):
        raise Exception("Number of .wav files different than number of label files")

    for wav_name, labels_name in zip(wav_files, labels_files):
        audio = AudioSegment.from_file(os.path.join(conf['input_dir_wav'], wav_name))
        labels = np.loadtxt(os.path.join(conf['input_dir_label'], labels_name))

        # if labels.shape[1] == 0:
            
        labels = convert_labels(labels, audio.duration_seconds, conf.getfloat("start_time"), conf.getfloat("time_step"))
        np.savetxt(os.path.join(conf['output_dir_label'], labels_name), labels)

        audio = convert_audio(audio)
        audio.export(os.path.join(conf['output_dir_wav'], wav_name), format="wav")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_name', type=str)
    args = parser.parse_args()

    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.sections()
    conf.read('consts.conf')

    ds_conf = conf[args.ds_name]
    convert_dataset(ds_conf)


main()