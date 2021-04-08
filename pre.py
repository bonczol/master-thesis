import time
import os
import numpy as np
import os
import argparse
import configparser
from utils import get_args_and_config
from converters import MirConverter, DummyConverter
from multiprocessing import Pool


def convert_example(converter):
    converter.convert()


def get_paths(dir, file_names, extension):
    return [os.path.join(dir, fn + extension) for fn in file_names]


def main():
    args, conf = get_args_and_config()

    file_names = [os.path.splitext(f)[0] for f in  os.listdir(conf['input_dir_wav']) 
                  if not f.startswith('.') and f.endswith('.wav')] 

    wav_paths = get_paths(conf['input_dir_wav'], file_names, ".wav")
    label_paths = get_paths(conf['input_dir_label'], file_names, conf['label_ext'])
    out_wav_paths = get_paths(conf['output_dir_wav'], file_names, '.wav')
    out_label_paths = get_paths(conf['output_dir_label'], file_names, '.csv')

    if args.ds_name == 'MIR-1k':
        converters = [MirConverter(w, l, ow, ol) 
                      for w, l, ow, ol in zip(wav_paths, label_paths, out_wav_paths, out_label_paths)]
    elif args.ds_name == 'MDB-stem-synth':
        converters = [DummyConverter(w, l, ow, ol) 
                      for w, l, ow, ol in zip(wav_paths, label_paths, out_wav_paths, out_label_paths)]
    else:
        raise Exception('No dataset')

    print(type(converters[0]).__name__)

    with Pool() as pool:
        pool.map(convert_example, converters)


if __name__ == '__main__':
    main()