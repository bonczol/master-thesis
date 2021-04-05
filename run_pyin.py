import subprocess
import pandas as pd
import numpy as np
import os
import pickle
from utils import get_args_and_config


def main():
    _, conf = get_args_and_config()

    subprocess.call(['sonic-annotator',
                     '-t', conf['pyin_trans'],
                     conf['input_dir_wav'],
                     '-r',
                     '-w', 'csv',
                     '--csv-basedir', 'results',
                     '--csv-force'])

    rows = []

    for f_name in os.listdir(conf['tmp_results_pyin']):
        path = os.path.join(conf['tmp_results_pyin'], f_name)
        f_name_no_vamp = f_name.replace('_vamp_pyin_pyin_smoothedpitchtrack', '')
        time, freq_est = list(np.transpose(np.loadtxt(path, delimiter=',')))
        rows.append([f_name_no_vamp, time, freq_est])

    with open(conf["pyin_results_path"], 'wb') as f:
        pickle.dump(pd.DataFrame(rows), f)
        

if __name__ == "__main__":
    main()