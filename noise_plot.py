import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import utils
import os
import matplotlib.pyplot as plt
from plots import add_voicing_and_cents, calc_metrics


def plot_noise(results, output_path):
    lineplot = sns.lineplot(data=results, x='snr', y='RPA', hue='method')
    lineplot.set_title('Accompaniment')
    fig = lineplot.get_figure()
    fig.savefig(output_path)
    fig.set


def main():
    parser, conf = utils.get_parser_and_config()
    args = parser.parse_args()
    conf = conf[args.ds_name]

    noise_types = ['acco']
    snrs = ['clean', '20', '10', '0']
    detectors = [
        "SPICE", 
        "CREPETINY", 
        'YIN', 
        # 'hf0'
    ]

    with open(conf['processed_label_binary'], 'rb') as f:
        labels_df = pickle.load(f)

    dfs = []

    for detector in detectors:
        for noise_type in noise_types:
            for snr in snrs:
                if snr != 'clean':
                    path = os.path.join(conf['results_dir'], f'{args.ds_name}_{detector}_{noise_type}_{snr}.pkl')
                else:
                    path = os.path.join(conf['results_dir'], f'{args.ds_name}_{detector}.pkl')
                with open(path, 'rb') as f:
                    df = pickle.load(f)

                df['method'] = detector
                df['noise_type'] = noise_type
                df['snr'] = snr
                dfs.append(labels_df.join(df.set_index('file'), on='file'))

    results_df = pd.concat(dfs)
    print(results_df)

    results_df = results_df.apply(lambda x: add_voicing_and_cents(x, conf), axis=1)
    results_df = calc_metrics(results_df, args.ds_name)

    for noise_type in noise_types:
        plot_noise(results_df, f'{conf["results_dir"]}/{args.ds_name}_noise_{noise_type}.png')


if __name__ == "__main__":
    main()