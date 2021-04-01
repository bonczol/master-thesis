import time
import os
import numpy as np
import os
import argparse
import configparser
from converters import MirConverter, MdbConverter
from multiprocessing import Pool


def convert_example(converter, out_wav_path, out_label_path):
    audio = converter.convert_audio()
    labels = converter.convert_label(audio.duration_seconds)
    audio.export(out_wav_path, format='wav')
    np.savetxt(out_label_path, labels, delimiter=',', fmt='%1.6f')


def get_paths(dir, file_names, extension):
    return [os.path.join(dir, fn + extension) for fn in file_names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_name', type=str)
    args = parser.parse_args()
    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read('consts.conf')
    ds_conf = conf[args.ds_name]

    file_names = [os.path.splitext(f)[0] for f in  os.listdir(ds_conf['input_dir_wav']) 
                  if not f.startswith('.') and f.endswith('.wav')] 

    wav_paths = get_paths(ds_conf['input_dir_wav'], file_names, ".wav")
    label_paths = get_paths(ds_conf['input_dir_label'], file_names, ds_conf['label_ext'])

    if args.ds_name == 'MIR-1k':
        voicing_paths = get_paths(ds_conf['dir_voicing'], file_names, ".vocal")
        converters = [MirConverter(w, l, v) for w, l, v in zip(wav_paths, label_paths, voicing_paths)]
    elif args.ds_name == 'MDB-stem-synth':
        converters = [MdbConverter(w, l) for w, l in zip(wav_paths, label_paths)]
    else:
        raise Exception('No dataset')

    print(type(converters[0]).__name__)

    out_wav_paths = get_paths(ds_conf['output_dir_wav'], file_names, '.wav')
    out_label_paths = get_paths(ds_conf['output_dir_label'], file_names, '.csv')

    with Pool() as pool:
        pool.starmap(convert_example, zip(converters, out_wav_paths, out_label_paths))


if __name__ == '__main__':
    main()