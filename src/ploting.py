import numpy as np
import pickle
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import utils
import consts
import mir_eval.melody as mir
import seaborn as sns
pd.options.display.max_columns = 50
pd.options.display.max_rows = 50
from method import Tracker


def add_voicing_and_cents(row):
    tracker = Tracker(row["method"])
    est_voicing = row['confidence'] > consts.THRESHOLDS[tracker]

    row['ref_voicing'], row['ref_cent'], row['est_voicing'], row['est_cent'] = \
        mir.to_cent_voicing(row['label_time'], row['label_pitch'], row['time'], row['pitch'], est_voicing, hop=0.032)

    return row


def flatten_samples(data):
    flat_df = pd.concat([data[['method', 'dataset', col]].explode(col) 
                         for col in ['ref_voicing', 'ref_cent', 'est_voicing', 'est_cent']], axis=1)
    return flat_df.loc[:,~flat_df.columns.duplicated()]


def calc_metrics(row):
    row["RPA"] = mir.raw_pitch_accuracy(row['ref_voicing'], row['ref_cent'], row['est_voicing'], row['est_cent'])
    row["RWC"]  = mir.raw_chroma_accuracy(row['ref_voicing'], row['ref_cent'], row['est_voicing'], row['est_cent'])

    if row['method'] == "MIR-1k":
        row["VRR"]  = mir.voicing_recall(row['ref_voicing'], row['est_voicing'])
        row["VRF"]  = mir.voicing_false_alarm(row['ref_voicing'], row['est_voicing'])
        row["OA"] = mir.overall_accuracy(row['ref_voicing'], row['ref_cent'], row['est_voicing'], row['est_cent'])
    else:
        row["VRR"], row["VRF"], row["OA"] = pd.NA, pd.NA, pd.NA

    return row


def metrics_summary(data, metrics, output_path):
    summary = data.groupby(['method', 'dataset'])[metrics] \
        .agg(['mean', 'std']) \
        .apply(lambda x: x * 100) \
        .round(decimals=2)

    summary.to_csv(output_path)
    print(summary)


def cumulative_grid(data, output_path):
    max_pitch_diff = np.arange(5, 102, 30)
    rows = []

    for (method, dataset), g in data.groupby(['method', 'dataset']):
        rpa_arr = utils.rpa_multi_tolerance(g.ref_voicing, g.ref_cent, g.est_voicing, g.est_cent, max_pitch_diff)
        rows.extend([[method, dataset, pitch_diff, rpa] for rpa, pitch_diff in zip(rpa_arr, max_pitch_diff)])
 
    cumulative_df = pd.DataFrame(rows, columns=['method', 'dataset', 'pitch_diff', 'rpa'])
    print(cumulative_df.head(10))
    
    g = sns.FacetGrid(cumulative_df, col="dataset", col_wrap=2)
    g.map_dataframe(sns.lineplot, x='pitch_diff',  y="rpa", hue="method", palette=consts.COLORS, hue_order=[m.value for m in list(Tracker)])
    g.set_titles(col_template="{col_name}")
    g.add_legend()
    g.savefig(output_path)
    plt.clf()


def box_plot_grid(data, output_path):
    g = sns.FacetGrid(data, col="dataset", col_wrap=2)
    g.map_dataframe(sns.boxplot, x="metric", y="value", hue="method", data=data, 
        palette=consts.COLORS, hue_order=[m.value for m in list(Tracker)], showfliers = False
        )
    g.add_legend()
    g.savefig(output_path)
    plt.clf()


"""
Labels / results loading
"""

def load_result(dataset, tracker):
    with open(dataset.get_result(tracker.value), 'rb') as f:
        return pickle.load(f)


def load_all_results(datasets, trackers):
    results = []

    for dataset, tracker in itertools.product(datasets, trackers):
        result = load_result(dataset, tracker)
        result['dataset'] = dataset.name
        result['method'] = tracker.value
        results.append(result)

    return pd.concat(results).dropna()


def load_label(dataset):
    with open(dataset.label_bin_path, 'rb') as f:
        return pickle.load(f)


def load_all_labels(datasets):
    labels = []

    for dataset in datasets:
        label = load_label(dataset)
        label['dataset'] = dataset.name
        labels.append(label)
    
    return pd.concat(labels)


def load_merged_data(datasets, trackers):
    labels = load_all_labels(datasets)
    results = load_all_results(datasets, trackers)
    return results.join(labels.set_index(['file', 'dataset']), on=['file','dataset'])


"""
API
"""

def plot(dataset, trackers):
    data = load_merged_data([dataset], trackers)

    data['ref_voicing'], data['ref_cent'], \
    data['est_voicing'], data['est_cent'] = zip(*data.apply(add_voicing_and_cents, axis=1))



def subplot(datasets, trackers):
    sns.set_style("whitegrid")
    sns.set_context("paper")

    data = load_merged_data(datasets, trackers)
    data = data.apply(add_voicing_and_cents, axis=1)

    data = data.apply(calc_metrics, axis=1)

    metrics = ['RPA', 'RWC', 'VRR', 'VRF', 'OA']
    metrics_summary(data, metrics, consts.RESULTS_PATH / 'summary.csv')

    results_melt = pd.melt(data, id_vars=['file', 'method', 'dataset'],
        value_vars=['RPA', 'RWC'], var_name='metric')
    box_plot_grid(results_melt, consts.PLOTS_PATH / 'boxplot_pitch.pdf')

    







    

    # flat_data = flatten_samples(data)

    # cumulative_grid(flat_data, consts.PLOTS_PATH / 'cumulative.pdf')





    # Boxplot - pitch
    # Boxplot - voicing
    



    






