import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
import yaml
import glob
import params
from method import Method
from collections import defaultdict
from utils import get_parser_and_config, rpa_multi_tolerance
from mir_eval.melody import raw_pitch_accuracy, raw_chroma_accuracy, voicing_false_alarm, voicing_recall, overall_accuracy, to_cent_voicing


def add_voicing_and_cents(df):
    est_voicing = df['confidence'] > params.THRESHOLDS[Method(df["method"])]

    df['ref_voicing'], df['ref_cent'], df['est_voicing'], df['est_cent'] = to_cent_voicing(
            df['label_time'], df['label_pitch'], df['time'], df['pitch'], est_voicing, hop=0.032)

    return df


def cumulative(results, dataset):
    sns.set_style("whitegrid")
    grouped_res = results.groupby('method')
    r = pd.DataFrame(np.arange(5, 102, 5), columns=['pitch_diff'])

    for name, group in grouped_res:
        r[name] = rpa_multi_tolerance(group.ref_voicing, group.ref_cent, group.est_voicing, group.est_cent, r['pitch_diff'])

    r = pd.melt(r, id_vars=['pitch_diff'], value_vars=list(grouped_res.groups.keys()), var_name='method', value_name='RPA')
    
    line_plot = sns.lineplot(data=r, x='pitch_diff', y="RPA", hue="method",
        palette=params.COLORS, hue_order=[m.value for m in list(Method)])
    line_plot.get_figure().savefig(f'plots/cumulative_{dataset}.png')
    plt.clf()


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


# How much time (ms) does it take to evaluate 1s of audio
def calc_latency(results, output_path, frame_size=1024, sample_rate=16000):
    latency = results.groupby(['method']).apply(
            lambda group: group['evaluation_time'].sum() / (group['duration'].sum())
        ).apply(lambda x: x * 100)
    latency.to_csv(output_path)
    print(latency.head())


def metrics_summary(results, metrics, output_path):
    summary = results.groupby(['method'])[metrics] \
        .agg(['mean', 'std']) \
        .apply(lambda x: x * 100) \
        .round(decimals=2)

    summary.to_csv(output_path)
    print(summary.head())


def box_plot(results, dataset):
    box_plot = sns.boxplot(x="metric", y="value", hue="method", data=results, 
        palette=params.COLORS, hue_order=[m.value for m in list(Method)])
    box_plot.legend(loc='lower left')
    fig = box_plot.get_figure()
    fig.set_size_inches(10, 6)
    fig.savefig(f'plots/box_plot_{dataset}.png')
    plt.clf()


def instruments_plot(results):
    results = results.sort_values(by=['avg_pitch'])
    grouped_res = results.groupby('method')

    for name, group in grouped_res:
        fig, ax = plt.subplots()
        sc = ax.scatter(group['instrument'], group['avg_pitch'], c=group['RPA'], vmin=0, vmax=1, cmap=plt.get_cmap('RdBu'), alpha=0.5, edgecolors='black')
        fig.set_size_inches(10, 6)
        fig.colorbar(sc)
        plt.xticks(rotation=90)
        fig.tight_layout()
        fig.savefig(f'plots/instuments_{name}.png')
    plt.clf()


def instruments_comparsion(results, dataset):
    avg_pitch_by_instrument = results.groupby(['instrument'])['avg_pitch'].agg('mean').rename('avg_instrument_pitch')
    results_instument_pitch =  results.join(avg_pitch_by_instrument, on="instrument")
    results_instument_pitch.sort_values(by=['avg_instrument_pitch'], inplace=True)

    rpa_by_instrument = results_instument_pitch.groupby(['method', 'instrument'], sort=False)['RPA', 'avg_instrument_pitch'].mean()
    rpa_by_instrument = rpa_by_instrument.reset_index()

    fig, ax1 = plt.subplots()
    ax1 = sns.scatterplot(y="RPA", x="instrument", hue="method", style="method", data=rpa_by_instrument,  s=50)
    plt.xticks(rotation=90)
    ax1.set_xlabel("Instruments")
    ax2 = ax1.twinx()
    ax2 = sns.lineplot(x="instrument", y="avg_instrument_pitch", data=rpa_by_instrument,  style=True, dashes=[(2,2)], color='red')
    ax2.set_ylabel("Average frequency of instrument", color='red')
    ax2.get_legend().remove()
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    fig.savefig(f'plots/instruments_comp_{dataset}.png')
    plt.clf()


# def pitch_contour(results):
#     files = ['abjones_2_07']
#     starts = [None]
#     ends = [None]

#     for f in zip(files, starts, ends):
#         fig, axs = plt.subplots(4)
#         # fdata = results.loc[f]
#         results_indexed = results.set_index(['file', 'method'])
#         print(results_indexed.loc[f, 'SPICE', :])
#         # print(results_indexed.loc[f])
#         # for 





def grid_search(results_raw, conf):
    thresholds = np.arange(0.5, 1, 0.01)
    params = defaultdict(list)

    for t in thresholds:
        print(f'Threshold = {t}')
        results_cents = results_raw.apply(lambda x: add_voicing_and_cents(x, conf, t), axis=1)
        results_cents["OA"] = results_cents.apply(lambda r:
                overall_accuracy(r['ref_voicing'], r['ref_cent'], r['est_voicing'], r['est_cent']), axis=1)
     
        means = results_cents.groupby(['method'])["OA"].mean()
        for method, mean_oa in means.iteritems():
            params[method].append(mean_oa)

    for key in params.keys():
        best_treshold_idx = np.argmax(params[key])
        best_treshold = thresholds[best_treshold_idx]
        print(key, best_treshold, f'oa = {np.max(params[key])}')




def main():
    parser, conf = get_parser_and_config()
    args = parser.parse_args()
    conf = conf[args.ds_name]

    # Load labels
    with open(conf['processed_label_binary'], 'rb') as f:
        labels_df = pickle.load(f)
        
    print('Labels ready ...')

    # Load results
    detectors = list(Method)

    dfs = []

    for detector in detectors:
        path = os.path.join(conf['results_dir'], f'{args.ds_name}_{detector.value}.pkl')
        with open(path, 'rb') as f:
            df = pickle.load(f)

        df['method'] = detector.value
        dfs.append(labels_df.join(df.set_index('file'), on='file'))
    
    results_df = pd.concat(dfs)
    results_df = results_df.dropna()
    print('Results ready ...')

    # grid_search(results_df, conf)

    # pitch_contour(results_df)


    # Convert to cents
    results_df = results_df.apply(add_voicing_and_cents, axis=1)
    print("Converted to cents ...")


    # Flat version
    flat_df = pd.concat([results_df[['method', col]].explode(col) 
                         for col in ['ref_voicing', 'ref_cent', 'est_voicing', 'est_cent']], axis=1)

    flat_df = flat_df.loc[:,~flat_df.columns.duplicated()]

    
    # Cumulative density function
    cumulative(flat_df, args.ds_name)


    # Metrics
    if args.ds_name == "MIR-1k":
        # metrics = ['RPA', 'RWC', 'VRR', 'VRF', 'OA']
        metrics = ['RPA', 'RWC']
    else:
        metrics = ['RPA', 'RWC']

    results_df = calc_metrics(results_df, args.ds_name)
    metrics_summary(results_df, metrics, f'{conf["results_dir"]}/{args.ds_name}_summary.csv')
    print('\n')
    calc_latency(results_df, f'{conf["results_dir"]}/{args.ds_name}_latency.csv')



    # print('\VRF')
    # x = results_df.sort_values(by=['VRF'], ascending=False)
    # # results_df.hist(column=['VRF'], bins=24)
    # # plt.show()
    # print(x.head(10))
    

    # Box plots
    results_melt = pd.melt(results_df, id_vars=['file', 'method'],
        value_vars=metrics, var_name='metric')

    box_plot(results_melt, args.ds_name)
    
    # Insturments
    if args.ds_name == "MDB-stem-synth":
        instruments = get_instruments(conf['metadata_dir'])
        results_df['instrument'] = results_df['file'].map(instruments)
        results_df['avg_pitch']  = results_df['label_pitch'].apply(lambda pitch: np.sum(pitch) / np.count_nonzero(pitch))
        instruments_comparsion(results_df, args.ds_name)





if __name__ == "__main__":
    main()
 





        