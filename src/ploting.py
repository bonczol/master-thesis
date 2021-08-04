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
pd.options.display.max_rows = 100
from method import Tracker



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
        rows.extend([[method, dataset, pitch_diff, rpa] for rpa, pitch_diff in zip(rpa_arr, max_pitch_diff)])
 
    cumulative_df = pd.DataFrame(rows, columns=['method', 'dataset', 'pitch_diff', 'rpa'])
    print(cumulative_df.head(10))
    
    g = sns.FacetGrid(cumulative_df, col="dataset", col_wrap=3, col_order=consts.DS_ORDER)
    g.map_dataframe(sns.lineplot, x='pitch_diff',  y="rpa", hue="method", palette=consts.COLORS, hue_order=[m.value for m in list(Tracker)])
    g.set_titles(col_template="{col_name}")
    g.add_legend()
    g.savefig(output_path)
    plt.clf()


def box_plot_grid(data, output_path):
    g = sns.FacetGrid(data, col="dataset", col_wrap=3, aspect=0.65, col_order=consts.DS_ORDER)
    g.map_dataframe(sns.boxplot, x="metric", y="value", hue="method", data=data, 
        palette=consts.COLORS, hue_order=[m.value for m in list(Tracker)], linewidth=0.5, fliersize=0.25)
    g.set_titles(col_template="{col_name}")
    g.add_legend()
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
    ax1 = sns.scatterplot(y="RPA", x="instrument", hue="method", style="method", data=avg_inst_rpa,  s=50, palette=consts.COLORS, hue_order=[m.value for m in list(Tracker)])
    plt.xticks(rotation=90)
    ax1.set_xlabel("Instruments")
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2 = sns.lineplot(x="instrument", y="ref_hz", data=avg_inst_f0,  style=True, dashes=[(2,2)], color='red')
    ax2.set_ylabel("Average frequency of instrument", color='red')
    ax2.get_legend().remove()
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.clf()


def subplot():
    sns.set_theme()
    sns.set_context("paper")

    # with open(consts.POST_RESULTS_PATH / 'data.pkl', 'rb') as f:
    #     data = pickle.load(f)

    with open(consts.POST_RESULTS_PATH / 'flat_data.pkl', 'rb') as f:
        flat_data = pickle.load(f)
    
    # with open(consts.POST_RESULTS_PATH / 'flat_metrics.pkl', 'rb') as f:
    #     flat_metrics = pickle.load(f)

    metrics = ['RPA', 'RWC', 'VRR', 'VRF', 'OA']
    # metrics_summary(data, metrics, consts.RESULTS_PATH / 'summary.csv')

    # Box plot RPA RWC
    # clean_data = data[data['noise'] == 'clean']
    # clean_data_melt = pd.melt(clean_data, id_vars=['file', 'method', 'dataset'],
    #     value_vars=['RPA', 'RWC'], var_name='metric')
    # box_plot_grid(clean_data_melt, consts.PLOTS_PATH / 'boxplot_pitch.pdf')

    # Box plot 


    # # Cumulative RPA
    # clean_flat_data = flat_data[flat_data['noise'] == 'clean']
    # cumulative_grid(clean_flat_data, consts.PLOTS_PATH / 'cumulative.pdf')

    # Instruments MDB
    # instruments(flat_data, consts.PLOTS_PATH / 'instruments.pdf')

    # 
    



    






