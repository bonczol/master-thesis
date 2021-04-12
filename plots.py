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
from utils import get_args_and_config, get_time_series_paths, resample_zeros, resample, get_vocal_paths
from mir_eval.melody import raw_pitch_accuracy, raw_chroma_accuracy, voicing_false_alarm, voicing_recall, overall_accuracy, to_cent_voicing


def add_voicing_and_cents(df, threshold):
    if df['confidence'] is not np.NAN:
        est_voicing = df['confidence'] > threshold
    else:
        est_voicing = df['pitch'] > 0
    
    df['ref_voicing'], df['ref_cent'], df['est_voicing'], df['est_cent'] = to_cent_voicing(
            df['label_time'], df['label_pitch'], df['time'], df['pitch'], est_voicing)

    return df


def pitch_diff(ref_voicing, ref_cent, est_voicing, est_cent):
    v = ref_voicing.astype(np.bool)
    x = np.where(est_cent <= 0.0, 500, est_cent)
    return np.abs(ref_cent[v] - x[v])


def cumulative(results):
    grouped_res = results.groupby('method')
    r = []
    
    for i in range(1, 202, 5):
        for name, group in grouped_res:
            rwa = raw_pitch_accuracy(group.ref_voicing, group.ref_cent, group.est_voicing, group.est_cent, i)
            r.append([i, name, rwa])
    
    r = pd.DataFrame(r, columns=['threshold', 'method', 'rwa'])

    sns.lineplot(data=r, x="threshold", y="rwa", hue="method")
    plt.show()


def get_instruments(metadata_dir):
    instruments = dict()

    for path in glob.glob(f'{metadata_dir}/*'):
        with open(path) as f:
            meta = f.full_load(f)
            for info in meta['stems'].values():
                filename = info['filename']
                name, ext = os.path.splitext(filename)
                filename = name + "_RESYN" + ext
                instruments[filename] = info['instrument']
                
    return instruments
            








def main():
    args, conf = get_args_and_config()

    # Load labels
    labels = []

    for path in get_time_series_paths(conf['output_dir_label']):
        ts = np.loadtxt(path, delimiter=",")
        f_name = os.path.splitext(os.path.basename(path))[0]
        labels.append([f_name, ts[:,0], ts[:,1]])

    labels_df = pd.DataFrame(labels, columns=['file', 'label_time', 'label_pitch'])


    # Load results
    detectors = ["spice", "pyin"]
    dfs = []

    for detector in detectors:
        path = conf[f'{detector}_results_path']
        with open(path, 'rb') as f:
            df = pickle.load(f)

        df['method'] = detector
        dfs.append(labels_df.join(df.set_index('file'), on='file'))
    
    results_df = pd.concat(dfs)

    # Convert to cents
    results_df = results_df.apply(lambda x: add_voicing_and_cents(x, 0.922), axis=1)

    # Flat version
    flat_df = pd.concat([results_df[['method', col]].explode(col) 
                         for col in ['ref_voicing', 'ref_cent', 'est_voicing', 'est_cent']], axis=1)

    flat_df = flat_df.loc[:,~flat_df.columns.duplicated()]

    cumulative(flat_df)



    # # Meauseres
    # results_df["RPA"] = results_df.apply(
    #     lambda r: raw_pitch_accuracy(r['ref_voicing'], r['ref_cent'], r['est_voicing'], r['est_cent']), axis=1)

    # results_df["RWC"]  = results_df.apply(lambda r:
    #     raw_chroma_accuracy(r['ref_voicing'], r['ref_cent'], r['est_voicing'], r['est_cent']), axis=1)

    # if args.ds_name != "MDB-stem-synth": # Skip voicing for synth
    #     results_df["VRR"]  = results_df.apply(lambda r: 
    #         voicing_recall(r['ref_voicing'], r['est_voicing']), axis=1)

    #     results_df["VRF"]  = results_df.apply(lambda r: 
    #         voicing_false_alarm(r['ref_voicing'], r['est_voicing']), axis=1)

    #     results_df["OA"] = results_df.apply(lambda r:
    #         overall_accuracy(r['ref_voicing'], r['ref_cent'], r['est_voicing'], r['est_cent']), axis=1)


    # results_melt_df = pd.melt(results_df, id_vars=['file', 'method'],
    #     value_vars=['RPA', 'RWC', 'VRR', 'VRF', 'OA'], var_name='metric')
    
    # mean = results_df.groupby(['method']).mean() * 100
    # std = results_df.groupby(['method']).std() * 100
    # print(mean.head(10))
    # print(std.head(10))
    
    #  # Box plot
    # sns.set_theme(style="ticks", palette="pastel")
    # sns.boxplot(x="metric", y="value",
    #         hue="method", palette=["m", "g"],
    #         data=results_melt_df)
    # sns.despine(offset=10, trim=True)

    # plt.savefig("plots/test1.png")
    # plt.show()
    # plt.clf()



        

if __name__ == "__main__":
    main()
 





        