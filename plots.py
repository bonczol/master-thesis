import pandas as pd
import numpy as np
import argparse
import configparser
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
import yaml
import glob
import itertools
from multiprocessing import Pool
from utils import get_parser_and_config, get_time_series_paths, resample_zeros, resample, get_vocal_paths, rpa_multi_tolerance
from mir_eval.melody import raw_pitch_accuracy, raw_chroma_accuracy, voicing_false_alarm, voicing_recall, overall_accuracy, to_cent_voicing


def add_voicing_and_cents(df, threshold):
    if df['confidence'] is not np.NAN:
        est_voicing = df['confidence'] > threshold
    else:
        est_voicing = df['pitch'] > 0
    
    df['ref_voicing'], df['ref_cent'], df['est_voicing'], df['est_cent'] = to_cent_voicing(
            df['label_time'], df['label_pitch'], df['time'], df['pitch'], est_voicing)

    return df


def cumulative(results, dataset):
    grouped_res = results.groupby('method')
    r = pd.DataFrame(np.arange(5, 102, 5), columns=['pitch_diff'])

    for name, group in grouped_res:
        r[name] = rpa_multi_tolerance(group.ref_voicing, group.ref_cent, group.est_voicing, group.est_cent, r['pitch_diff'])

    r = pd.melt(r, id_vars=['pitch_diff'], value_vars=list(grouped_res.groups.keys()), var_name='method', value_name='RPA')
    
    line_plot = sns.lineplot(data=r, x='pitch_diff', y="RPA", hue="method")
    line_plot.get_figure().savefig(f'plots/cumulative_{dataset}.png')


def get_instruments(metadata_dir):
    instruments = dict()

    for path in glob.glob(f'{metadata_dir}/*'):
        with open(path) as f:
            meta = yaml.full_load(f)
            for info in meta['stems'].values():
                filename = info['filename']
                name, _ = os.path.splitext(filename)
                filename = name + ".RESYN"
                instruments[filename] = info['instrument']
                 
    return instruments


def calc_metrics(results, dataset):
    all_cols = ['ref_voicing', 'ref_cent', 'est_voicing', 'est_cent']

    results["RPA"] = results[all_cols].apply(
        lambda r: raw_pitch_accuracy(r[0], r[1], r[2], r[3]), raw=True, axis=1)

    print("RPA ...")

    results["RWC"]  = results[all_cols].apply(lambda r:
        raw_chroma_accuracy(r[0], r[1], r[2], r[3]), raw=True, axis=1)

    print("RWC ...")

    if dataset != "MDB-stem-synth": # Skip voicing for synth
        results["VRR"]  = results.apply(lambda r: 
            voicing_recall(r['ref_voicing'], r['est_voicing']), axis=1)

        results["VRF"]  = results.apply(lambda r: 
            voicing_false_alarm(r['ref_voicing'], r['est_voicing']), axis=1)

        results["OA"] = results.apply(lambda r:
            overall_accuracy(r['ref_voicing'], r['ref_cent'], r['est_voicing'], r['est_cent']), axis=1)

    return results


def box_plot(results, dataset):
    sns.set_theme(style="ticks", palette="pastel")
    sns.boxplot(x="metric", y="value",
            hue="method", palette=["m", "g"],
            data=results)
    sns.despine(offset=10, trim=True)
    plt.savefig(f'plots/box_plot_{dataset}.png')
    plt.clf()


def instruments_plot(results):
    results = results.sort_values(by=['avg_pitch'])
    grouped_res = results.groupby('method')

    for name, group in grouped_res:
        fig, ax = plt.subplots()
        sc = ax.scatter(group['instrument'], group['avg_pitch'], c=group['RPA'], vmin=0, vmax=1, cmap=plt.get_cmap('RdBu'), alpha=0.5, edgecolors='black')
        fig.set_size_inches(14, 7)
        fig.colorbar(sc)
        plt.xticks(rotation=90)
        fig.tight_layout()
        fig.savefig(f'plots/instuments_{name}.png')


def read_label(path):
    ts = np.loadtxt(path, delimiter=",")
    f_name = os.path.splitext(os.path.basename(path))[0]
    return [f_name, ts[:,0], ts[:,1]]


def main():
    parser, conf = get_parser_and_config()
    args = parser.parse_args()
    conf = conf[args.ds_name]

    # Load labels
    pool = Pool()
    labels = pool.map(read_label, get_time_series_paths(conf['output_dir_label']))
    labels_df = pd.DataFrame(labels, columns=['file', 'label_time', 'label_pitch'])
    print('Labels ready ...')

    # Load results
    detectors = ["SPICE", "CREPE_TINY"]
    dfs = []

    for detector in detectors:
        path = os.path.join(conf['root_results_dir'], f'{args.ds_name}_{detector}.pkl')
        with open(path, 'rb') as f:
            df = pickle.load(f)

        df['method'] = detector 
        dfs.append(labels_df.join(df.set_index('file'), on='file'))
    
    results_df = pd.concat(dfs)
    print('Results ready ...')

    # Convert to cents
    if args.ds_name == "MIR-1k":
        threshold = 0.922
    else:
        threshold = 0.85

    results_df = results_df.apply(lambda x: add_voicing_and_cents(x, threshold), axis=1)
    print("Converted to cents ...")


    # Flat version
    # flat_df = pd.concat([results_df[['method', col]].explode(col) 
    #                      for col in ['ref_voicing', 'ref_cent', 'est_voicing', 'est_cent']], axis=1)

    # flat_df = flat_df.loc[:,~flat_df.columns.duplicated()]


    # Cumulative density function
    # cumulative(flat_df, args.ds_name)


    # Metrics
    results_df = calc_metrics(results_df, args.ds_name)
    mean = results_df.groupby(['method']).mean() * 100
    std = results_df.groupby(['method']).std() * 100
    print(mean.head(10))
    print(std.head(10))
    

    # Box plots
    # results_melt = pd.melt(results_df, id_vars=['file', 'method'],
    #     value_vars=['RPA', 'RWC', 'VRR', 'VRF', 'OA'], var_name='metric')

    # box_plot(results_melt, args.ds_name)
    
    # Insturments
    if args.ds_name == "MDB-stem-synth":
        instruments = get_instruments(conf['meta_data_dir'])
        results_df['instrument'] = results_df['file'].map(instruments)
        results_df['avg_pitch']  = results_df['label_pitch'].apply(lambda pitch: np.sum(pitch) / np.count_nonzero(pitch))
        instruments_plot(results_df)




if __name__ == "__main__":
    main()
 





        