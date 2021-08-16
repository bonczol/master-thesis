import numpy as np
import matplotlib.pyplot as plt
import librosa
import consts
import evaluate
import pandas as pd
import seaborn as sns
import sox
import utils
pd.options.display.max_columns = 50
pd.options.display.max_rows = 100
from librosa import display as librosadisplay


def plot_stft(x, sample_rate, color):
  x_stft = np.abs(librosa.stft(x, hop_length=32, n_fft=2048,  win_length=1024))
  x_stft_db = librosa.amplitude_to_db(x_stft, ref=np.max)
  librosadisplay.specshow(data=x_stft_db, x_axis="s", y_axis='log', sr=sample_rate, hop_length=32, cmap='coolwarm')
  plt.ylim(40, 200)
  plt.yticks([40, 80, 120, 160 , 200])
  # plt.ylim(350, 750)
  # plt.yticks(np.arange(350, 750, 100))


    
def generate(trackers):
  sns.set_theme()
  for wav_path in (consts.SPECTROGRAMS_WAV_PATH / "raw").glob('*.wav'):
    proc_wav_path = consts.SPECTROGRAMS_WAV_PATH / "processed" / wav_path.name
    (sox.Transformer().convert(consts.SR, 1, 16) 
                      .build(str(wav_path), str(proc_wav_path)))

    y, _ = librosa.load(proc_wav_path, sr=consts.SR, duration=2)
    results = []
    for tracker in trackers:
      _, time, pitch, confidence, _ = evaluate.evaluate(tracker, proc_wav_path)
      t = consts.THRESHOLDS[tracker.method]
      pitch = np.where(confidence > t, pitch, np.inf)
      results.append([tracker.method.value, time, pitch])

    results_df = pd.DataFrame(results, columns=["method", "time", "pitch"])
    results_df = (results_df.astype({'method': consts.METHOD_CAT})
                            .sort_values(by=['method']))

    flat_df = utils.explode_custom(results_df, ["time", "pitch"])
    flat_df = flat_df.astype({"time": np.float32, "pitch": np.float32})
    flat_df[["time", "pitch"]] = flat_df[["time", "pitch"]].applymap(lambda x:  np.inf if x <= 0 else x)
    
    g = sns.FacetGrid(flat_df, col='method',col_wrap=2, aspect=1.33)
    g.map(plot_stft, x=y, sample_rate=consts.SR)
    g.map_dataframe(sns.lineplot, x='time', y='pitch', color='green', linestyle='--')
    g.set_titles(col_template="{col_name}")
    for ax in g.axes.flat:
      ax.set_xlabel("Time [s]")
      ax.set_ylabel("Frequency [Hz]")

    spec_path = consts.SPECTROGRAMS_PATH / (str(wav_path.stem) + ".png")
    g.savefig(spec_path)
 