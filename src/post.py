import numpy as np
import pickle
import pandas as pd
from itertools import product
import consts
import mir_eval.melody as mir
from method import Tracker


def read_result(dataset, tracker, noise, snr):
    with open(dataset.get_result(tracker.value, noise, snr), 'rb') as f:
        return pickle.load(f)


def load_all_results(datasets, trackers, noises, snrs):
    results = []

    for dataset, tracker in product(datasets, trackers):
        result_path = dataset.get_result(tracker.value, None, None)
        if result_path.exists():
            result = read_result(dataset, tracker, None, None)
            result['dataset'] = dataset.name
            result['method'] = tracker.value
            result['noise'] = 'clean'
            result['snr'] = 'clean'
            results.append(result)

    for dataset, tracker, noise, snr in product(datasets, trackers, noises, snrs):
        result_path = dataset.get_result(tracker.value, noise, snr)
        if result_path.exists():
            result = read_result(dataset, tracker, noise, snr)
            result['dataset'] = dataset.name
            result['method'] = tracker.value
            result['noise'] = noise 
            result['snr'] = str(snr)
            results.append(result)

    return pd.concat(results).dropna()


def load_all_labels(datasets):
    labels = []

    for dataset in datasets:
        with open(dataset.label_bin_path, 'rb') as f:
            label = pickle.load(f)
        label['dataset'] = dataset.name
        labels.append(label) 
    
    return pd.concat(labels)


def add_voicing_and_cents(row):
    tracker = Tracker(row["method"])
    est_voicing = row['confidence'] > consts.THRESHOLDS[tracker]

    row['ref_voicing'], row['ref_cent'], row['est_voicing'], row['est_cent'] = \
        mir.to_cent_voicing(row['label_time'], row['label_pitch'], row['time'], row['pitch'], est_voicing, hop=0.032)

    return row


def calc_metrics(row):
    row["RPA"] = mir.raw_pitch_accuracy(row['ref_voicing'], row['ref_cent'], row['est_voicing'], row['est_cent'])
    row["RWC"]  = mir.raw_chroma_accuracy(row['ref_voicing'], row['ref_cent'], row['est_voicing'], row['est_cent'])

    if row['dataset'] == "MIR-1k":
        row["VRR"]  = mir.voicing_recall(row['ref_voicing'], row['est_voicing'])
        row["VRF"]  = mir.voicing_false_alarm(row['ref_voicing'], row['est_voicing'])
        row["OA"] = mir.overall_accuracy(row['ref_voicing'], row['ref_cent'], row['est_voicing'], row['est_cent'])
    else:
        row["VRR"], row["VRF"], row["OA"] = np.nan, np.nan, np.nan

    return row


def calc_metric_flatten(data):
    rows = []

    for (method, dataset, noise, snr), group in data.groupby(['method', 'dataset', 'noise', 'snr']):
        rpa = mir.raw_pitch_accuracy(group['ref_voicing'].values, group['ref_cent'].values, group['est_voicing'].values, group['est_cent'].values)
        rwc = mir.raw_chroma_accuracy(group['ref_voicing'].values, group['ref_cent'].values, group['est_voicing'].values, group['est_cent'].values)
        if dataset == "MIR-1k":
            vrr  = mir.voicing_recall(group['ref_voicing'].values, group['est_voicing'].values)
            vrf  = mir.voicing_false_alarm(group['ref_voicing'].values, group['est_voicing'].values)
            oa = mir.overall_accuracy(group['ref_voicing'].values, group['ref_cent'].values, group['est_voicing'].values, group['est_cent'].values)
        else:
            vrr, vrf, oa = np.nan, np.nan, np.nan
        rows.append([method, dataset, noise, snr, rpa, rwc, vrr, vrf, oa])

    return pd.DataFrame(rows, columns=['method', 'dataset', 'noise', 'snr', 'RPA', 'RCA', 'VRR', 'VRF', 'OA'])


def flatten_samples(data):
    flat_df = pd.concat([data[['file', 'method', 'dataset', 'noise', 'snr', 'instrument', col]].explode(col) 
                         for col in ['ref_voicing', 'ref_cent', 'est_voicing', 'est_cent']], axis=1)
    return flat_df.loc[:,~flat_df.columns.duplicated()]


def transform(datasets, trackers, noises, snrs):
    labels = load_all_labels(datasets)
    results = load_all_results(datasets, trackers, noises, snrs)
    data = results.join(labels.set_index(['file', 'dataset']), on=['file','dataset'])

    data = data.apply(add_voicing_and_cents, axis=1)
    data = data.apply(calc_metrics, axis=1)

    with open(consts.POST_RESULTS_PATH / 'labels.pkl', 'wb') as f:
        pickle.dump(labels, f)

    with open(consts.POST_RESULTS_PATH / 'data.pkl', 'wb') as f:
        pickle.dump(data, f)

    flat_data = flatten_samples(data)
    types2 = {
        'ref_voicing': 'int8', 
        'ref_cent': 'float32', 
        'est_voicing': 'int8', 
        'est_cent': 'float32', 
    }
    flat_data = flat_data.astype(types2)
    
    with open(consts.POST_RESULTS_PATH / 'flat_data.pkl', 'wb') as f:
        pickle.dump(flat_data, f)


    flat_metrics = calc_metric_flatten(flat_data)
    with open(consts.POST_RESULTS_PATH / 'flat_metrics.pkl', 'wb') as f:
        pickle.dump(flat_metrics, f)



    



