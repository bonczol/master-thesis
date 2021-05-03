import os
import utils
import numpy as np
import pandas as pd
import pickle
from converters import MirConverter, MDBConverter
from multiprocessing import Pool


def convert_example(converter):
    converter.convert()


def get_paths(dir, file_names, extension):
    return [os.path.join(dir, fn + extension) for fn in file_names]


def read_label(path):
    ts = np.loadtxt(path, delimiter=",")
    f_name = os.path.splitext(os.path.basename(path))[0]
    return [f_name, ts[:,0], ts[:,1]]


def main():
    parser, conf = utils.get_parser_and_config()
    args = parser.parse_args()
    conf = conf[args.ds_name]

    file_names = [os.path.splitext(f)[0] for f in  os.listdir(conf['unprocessed_wav_dir']) 
                  if not f.startswith('.') and f.endswith('.wav')] 

    wav_paths = get_paths(conf['unprocessed_wav_dir'], file_names, ".wav")
    label_paths = get_paths(conf['unprocessed_label_dir'], file_names, conf['label_ext'])
    out_wav_paths = get_paths(conf['processed_wav_dir'], file_names, '.wav')
    out_label_paths = get_paths(conf['processed_label_dir'], file_names, '.csv')

    if args.ds_name == 'MIR-1k':
        out_wav_background_paths = get_paths(conf['processed_wav_bg_dir'], file_names, '.wav')
        converters = [MirConverter(w, l, ow, ol, owb) 
                      for w, l, ow, ol, owb in zip(wav_paths, label_paths, out_wav_paths, out_label_paths, out_wav_background_paths)]
    elif args.ds_name == 'MDB-stem-synth':
        converters = [MDBConverter(w, l, ow, ol) 
                      for w, l, ow, ol in zip(wav_paths, label_paths, out_wav_paths, out_label_paths)]
    else:
        raise Exception('No dataset')

    print(type(converters[0]).__name__)

    with Pool() as pool:
        pool.map(convert_example, converters)
        labels = pool.map(read_label, out_label_paths)

    labels_df = pd.DataFrame(labels, columns=['file', 'label_time', 'label_pitch'])
    with open(conf['processed_label_binary'], 'wb') as f:
        pickle.dump(labels_df, f)


if __name__ == '__main__':
    main()