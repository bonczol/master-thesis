import numpy as np
import pickle
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import utils
import post
import librosa
from matplotlib import colors
import consts
import pretty_midi
import mir_eval.melody as mir
import seaborn as sns
from method import Method
from itertools import product
pd.options.display.max_columns = 50
pd.options.display.max_rows = 100


def add_hlines(latex_code):
    return latex_code.replace("\\\n", "\\ \hline\n")


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def semtiones_diff(hz1, hz2):
    return np.abs(12 * np.log2(hz1 / hz2))


def metrics_summary(data, metrics, output_path):
    summary = data.groupby(['dataset', 'method', 'noise', 'snr'])[metrics] \
        .agg(['mean', 'std']) \
        .apply(lambda x: x * 100) \
        .round(decimals=2)

    summary.to_csv(output_path)
    print(summary.head(100))


def cumulative_grid(data, output_path):
    max_pitch_diff = np.arange(5, 101, 1)
    rows = []

    for (method, dataset), g in data.groupby(['method', 'dataset']):
        rpa_arr = utils.rpa_multi_tolerance(g.ref_voicing, g.ref_cent, g.est_voicing, g.est_cent, max_pitch_diff)
        rca_arr = utils.rca_multi_tolerance(g.ref_voicing, g.ref_cent, g.est_voicing, g.est_cent, max_pitch_diff)
        rows.extend([[method, dataset, pitch_diff, rpa, rca] for rpa, rca, pitch_diff in zip(rpa_arr, rca_arr, max_pitch_diff)])
 
    cumulative_df = pd.DataFrame(rows, columns=['method', 'dataset', 'pitch_diff', 'RPA', 'RCA'])
    
    g = sns.FacetGrid(cumulative_df, col="dataset", col_wrap=2, col_order=consts.DS_ORDER)
    g.map_dataframe(sns.lineplot, x='pitch_diff',  y="RPA", hue="method", palette=consts.COLORS, hue_order=[m.value for m in list(Method)])
    g.set_titles(col_template="{col_name}")
    g.add_legend(ncol=2, bbox_to_anchor=(0.7, 0.25))
    g.savefig(output_path)
    plt.clf()

    cumulative_df = cumulative_df.astype({'dataset': consts.DS_CAT, 'method': consts.METHOD_CAT})

    cumulative_df[['RPA', 'RCA']] = np.round(cumulative_df[['RPA', 'RCA']] * 100, 1)
    latex_code = (cumulative_df.loc[cumulative_df.pitch_diff.isin([50])]
                  .set_index(['dataset', 'method'])
                  .sort_index()
                  .loc[:, ['RPA', 'RCA']]
                  .to_latex(multirow=True, sparsify=True))

    print(latex_code)



def box_plot_grid(data, output_path):
    g = sns.FacetGrid(data, col="dataset", col_wrap=2, col_order=consts.DS_ORDER)
    g.map_dataframe(sns.boxplot, x="metric", y="value", hue="method", data=data, 
        palette=consts.COLORS, hue_order=[m.value for m in list(Method)], linewidth=0.5, fliersize=0.25)
    g.set_titles(col_template="{col_name}")
    g.set_ylabels('Score')
    g.add_legend(ncol=2, bbox_to_anchor=(0.7, 0.25))
    g.savefig(output_path)
    plt.clf()


def instruments(data, output_path):
    mdb_data = data[data['dataset'] == 'MDB-stem-synth']

    inst_data = mdb_data[(mdb_data['method'] == 'CREPE') & (mdb_data['ref_voicing'] == 1)]
    inst_data['ref_hz'] = utils.semitones2hz(inst_data['ref_cent'] / 100)
    avg_inst_f0 = inst_data.groupby(['instrument'])['ref_hz'].agg('mean').reset_index().sort_values(by=['ref_hz'])

    calc_rpa = lambda g: mir.raw_pitch_accuracy(g['ref_voicing'], g['ref_cent'], g['est_voicing'], g['est_cent'])
    avg_inst_rpa = mdb_data.groupby(['method', 'instrument']) \
                              .apply(calc_rpa) \
                              .rename('RPA') \
                              .to_frame() \
                              .reset_index() \
                              .join(avg_inst_f0.set_index('instrument'), on='instrument') \
                              .sort_values(by=['ref_hz'])
                
    fig, ax1 = plt.subplots()
    ax1 = sns.scatterplot(y="RPA", x="instrument", hue="method", style="method", data=avg_inst_rpa, palette=consts.COLORS, hue_order=[m.value for m in list(Method)])
    plt.xticks(rotation=90)
    ax1.set_xlabel('')
    ax1.set_ylabel("Raw pitch accuracy")
    ax1.legend(loc='lower right')
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2 = sns.lineplot(x="instrument", y="ref_hz", data=avg_inst_f0,  style=True, dashes=[(2,2)], color='red')
    ax2.set_ylabel("Average frequency of an instrument", color='red')
    ax2.get_legend().remove()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.clf()



def noise_plot(data, output_path):
    noise_types = ['brown', 'pink', 'white', 'blue', 'violet']
    urmp_data = data.loc[(data.dataset == 'URMP') & (data.noise.isin(noise_types)), :]
    urmp_data.sort_values(by=['snr'], ascending=False, inplace=True)

    g = sns.FacetGrid(urmp_data, col="noise", col_wrap=2, col_order=noise_types, aspect=1.33)
    g.map_dataframe(sns.lineplot, x='snr', y="RPA", hue="method", data=urmp_data,  sort=False,
        palette=consts.COLORS, hue_order=[m.value for m in list(Method)])
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("SNR [dB]", "Raw Pitch Accuracy")
    g.add_legend(ncol=2, bbox_to_anchor=(0.7, 0.25))
    g.savefig(output_path)
    plt.clf()


def acco_plot(data, output_path):
    mir_data = data.loc[(data.dataset == 'MIR-1k') & (data.noise.isin(['acco'])), :]
    mir_data.sort_values(by=['snr'], ascending=False, inplace=True)

    plot = sns.lineplot(x='snr', y="RPA", hue="method", data=mir_data,  sort=False,
        palette=consts.COLORS, hue_order=[m.value for m in list(Method)])
    plot.set_xlabel("SNR [dB]")
    plot.set_ylabel("Raw Pitch Accuracy")

    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.clf()



def box_plot_voicing(data, output_path):
    voicing_data = data[(data.noise == 'clean') & (data.dataset == 'MIR-1k') & (~data.method.isin(['DDSPINV', 'SWIPE']))]
    data_melt = pd.melt(voicing_data, id_vars=['file', 'method', 'dataset'],
        value_vars=['VRR', 'VRF'], var_name='metric')

    palette = consts.COLORS.copy()
    palette.pop('DDSPINV')
    palette.pop('SWIPE')

    hues = [m for m in consts.METHODS_VAL if m not in ['DDSPINV', 'SWIPE']] 

    plot = sns.boxplot(x="metric", y="value", hue="method", data=data_melt, palette=palette, hue_order=hues,
         linewidth=0.5, fliersize=0.25)
    plot.set_xlabel('')
    plot.set_ylabel('Score')
    plt.tight_layout()
    plot.get_figure().savefig(output_path)
    plt.clf()


def voicing_table(data):
    metrics = ['VRR', 'VRF', 'RPA', 'OA']
    voicing_df = data.loc[(data.noise == 'clean') & (data.dataset == 'MIR-1k') & (~data.method.isin(['DDSPINV', 'SWIPE'])), :].copy()
    voicing_df[metrics] = np.round(voicing_df[metrics] * 100, 1)

    latex_code = (voicing_df.set_index('method')
                            .sort_index()
                            .loc[:, metrics]
                            .to_latex(multirow=True, sparsify=True)
        )

    print(latex_code)


def plot_datasets(data, output_path):
    ds_data = data[['dataset', 'label_pitch']].copy()
    ds_data_flat = (ds_data.set_index('dataset') 
                     .apply(lambda x: x.apply(pd.Series).stack()) 
                     .reset_index() 
                     .drop('level_1', 1))
                     

    ds_data_flat_voiced = ds_data_flat[ds_data_flat.label_pitch > 0]

    plot = sns.histplot(ds_data_flat_voiced, x='label_pitch', hue='dataset', stat="density", 
                        common_norm=False, color='blue', bins=24, log_scale=2, element="step")
    
    plot.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}'))
    plot.set_xlabel('Pitch(Hz)')
    plot.get_figure().tight_layout()
    plot.get_figure().savefig(output_path)


def datasets_info(data, output_path):
    info = data.groupby('dataset').agg(
        **{
            "Files" : ('file', lambda x: np.unique(x).shape[0]),
            "Duration" : ('duration', lambda x: np.sum(x) / 3600),
            "Voiced frames" : ('label_pitch', lambda x: np.sum(np.concatenate(x) > 0)),
            "Frames" : ('label_pitch', lambda x: np.concatenate(x).shape[0]),
            "Min f0" : ('label_pitch',  lambda x: np.min(np.concatenate(x)[np.concatenate(x) > 0])),
            "Max f0" : ('label_pitch', lambda x: np.concatenate(x).max()),
            "Instruments" : ('instrument', lambda x: np.unique(x).shape[0])
        }
    )

    info = info.reset_index()
    info['Range'] = info.apply(lambda row: semtiones_diff(row['Max f0'], row['Min f0']) / 12, axis=1)

    numeric = ['Duration', 'Min f0', 'Max f0', 'Range']
    info[numeric] = info[numeric].transform(lambda x: np.round(x, 1))

    frames = ['Voiced frames', 'Frames']
    info[frames] = info[frames].applymap(lambda x: human_format(x))

    info = info.T
    info.to_csv(output_path)
    latex_code = info.to_latex(columns=[1,0,2], header=False)
    latex_code = latex_code.replace("\\\n", "\\ \hline\n")
    print(latex_code)


def calc_latency(data, output_path):
    latency = (data.loc[(data.noise == 'clean') & data.dataset.isin(['URMP', 'MDB-stem-synth'])]
                   .groupby(['method']).apply(lambda g: g.evaluation_time.sum() / (g.duration.sum()))
                   .apply(lambda x: x * 1000) # to miliseconds
    )
    latency.to_csv(output_path)
    latex_code = latency.to_latex()
    print(latex_code)


def grid_search(data):
    thresholds = np.arange(0.5, 1, 0.01)
    consts = defaultdict(list)

    for t in thresholds:
        results_cents = data.apply(post.add_voicing_and_cents, axis=1)
        results_cents["OA"] = results_cents.apply(lambda r:
                mir.overall_accuracy(r['ref_voicing'], r['ref_cent'], r['est_voicing'], r['est_cent']), axis=1)
     
        means = results_cents.groupby(['method'])["OA"].mean()
        for method, mean_oa in means.iteritems():
            consts[method].append(mean_oa)

    for key in consts.keys():
        best_treshold_idx = np.argmax(consts[key])
        best_treshold = thresholds[best_treshold_idx]
        print(key, best_treshold, f'oa = {np.max(consts[key])}')


def notes_box_plot():
    with open(consts.POST_RESULTS_PATH / 'data_trans.pkl', 'rb') as f:
        data = pickle.load(f)

    data['method'] = data['method'].map({'PYINMIDI': 'PYIN/HMM', 'CREPEMIDI': 'CREPE/Heur.'})

    data = data.melt(id_vars=['file', 'method'], value_vars=['COn', 'COnP', 'COnPOff'],
        var_name='metric', value_name='metric_value').reset_index()

    print(data)

    boxplot = sns.boxplot(x="metric", y="metric_value", hue="method", data=data, 
        showmeans=True, linewidth=0.5, fliersize=0.25)

    output_path = consts.PLOTS_PATH / 'box_plot_notes.pdf'
    boxplot.set_xlabel('')
    boxplot.set_ylabel('Score')
    boxplot.get_figure().savefig(output_path)

    stats = data.groupby(['method', 'metric'])['metric_value'].agg(['mean', 'std'])
    stats[['mean', 'std']] = stats[['mean', 'std']].transform(lambda x: np.round(x*100, 1))
    stats =  stats.reset_index().pivot(index='method', columns='metric', values=['mean', 'std'])
    print(stats)
    print(stats.to_latex())
    plt.clf()



def plot_piano_roll(pm, start_pitch, end_pitch, ax, color, fs=100):
    cmap = colors.ListedColormap(['white', color])
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch), ax=ax, cmap=cmap)
    ax.grid(True)
    plt.xlim(4, 24)


def midi():
    sns.reset_orig()
    # plt.rcParams["axes.axisbelow"] = False
    with open(consts.POST_RESULTS_PATH / 'data_trans.pkl', 'rb') as f:
        data = pickle.load(f)

    methods = ['CREPEMIDI', 'PYINMIDI']
    labels = ['CREPE/Heur.', 'PYIN/HMM ']
    # print(methods)
    # tracks = ['AuSep_2_vn_08_Spring', 'AuSep_1_vn_01_Jupiter']
    tracks = ['AuSep_1_vn_01_Jupiter']
    
    for track in tracks:
        fig, axes = plt.subplots(len(methods)+1, 1, figsize=(8, 10), sharex=True)
        ground_truth = data.loc[(data.file == track), :].iloc[0,:]
        melody = utils.intervals_to_midi(ground_truth.ref_note_interval, ground_truth.ref_note_pitch)
        plot_piano_roll(melody, 59, 76, axes[0], sns.color_palette()[0])
        axes[0].set_title('Ground-truth')
        for i, method in enumerate(methods):
            row = data.loc[(data.file == track) & (data.method == method), :].iloc[0,:]
            melody = utils.intervals_to_midi(row.est_note_interval, row.est_note_pitch)
            plot_piano_roll(melody, 59, 76, axes[i+1], sns.color_palette()[i+1])
            axes[i+1].set_title(labels[i])

        for ax in axes:
            ax.label_outer()

        plt.tight_layout()
        fig.savefig(consts.PLOTS_PATH / f'midi_{track}.pdf')


def subplot():
    sns.set_theme()
    # with open(consts.POST_RESULTS_PATH / 'labels.pkl', 'rb') as f:
    #     labels =  pickle.load(f)

    # with open(consts.POST_RESULTS_PATH / 'data.pkl', 'rb') as f:
    #     data = pickle.load(f)

    '''
    Grid search - threshold
    '''
    # grid_search(data[(data.dataset == 'MIR-1k') & (data.method == 'CREPE') & (data.noise == 'clean')])

    # with open(consts.POST_RESULTS_PATH / 'flat_data.pkl', 'rb') as f:
    #     flat_data = pickle.load(f)

    # with open(consts.POST_RESULTS_PATH / 'flat_metrics.pkl', 'rb') as f:
    #     flat_metrics = pickle.load(f)

    # flat_metrics = flat_metrics.astype({'dataset': consts.DS_CAT, 'method': consts.METHOD_CAT})

    # metrics = ['RPA', 'RWC', 'VRR', 'VRF', 'OA']
    # metrics_summary(data, metrics, consts.RESULTS_PATH / 'summary.csv')

    """
    Datasets 
    """
    # plot_datasets(labels, consts.PLOTS_PATH / 'datasets.pdf')
    # datasets_info(labels, consts.PLOTS_PATH / 'datasets_info.csv')


    # Box plot RPA RWC
    # clean_data = data[data['noise'] == 'clean']
    # clean_data_melt = pd.melt(clean_data, id_vars=['file', 'method', 'dataset'],
    #     value_vars=['RPA', 'RWC'], var_name='metric')
    # box_plot_grid(clean_data_melt, consts.PLOTS_PATH / 'boxplot_pitch.pdf')

    # Box plot VRR VRF 
    # voicing_table(flat_metrics)
    # box_plot_voicing(data, consts.PLOTS_PATH / 'boxplot_vocing.pdf')


    # # Cumulative RPA
    # clean_flat_data = flat_data[flat_data['noise'] == 'clean']
    # cumulative_grid(clean_flat_data, consts.PLOTS_PATH / 'cumulative.pdf')

    # Instruments MDB
    # instruments(flat_data, consts.PLOTS_PATH / 'instruments.pdf')

    # Noise urmp
    # noise_metrics = flat_metrics.copy()
    # for _, row in noise_metrics[noise_metrics.noise == 'clean'].iterrows():  
    #     for noise in ['white', 'pink', 'brown', 'acco', 'blue', 'violet']:
    #         new_row = row.copy()
    #         new_row['noise'] = noise
    #         new_row['snr'] = 'inf'
    #         noise_metrics = noise_metrics.append(new_row)
    # noise_metrics = noise_metrics.loc[noise_metrics.noise != 'clean', :]

    # noise_plot(noise_metrics, consts.PLOTS_PATH / 'noise.pdf')
    # acco_plot(noise_metrics, consts.PLOTS_PATH / 'acco.pdf')


    """
    Latency
    """
    # calc_latency(data, consts.PLOTS_PATH / 'latency.csv')


    """
    Notes
    """
    notes_box_plot()
    # midi()

    






