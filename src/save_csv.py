import consts
import pickle
import numpy as np
import utils
import librosa
import crepe_mod.crepe as crepe
import matplotlib.pyplot as plt
from librosa import display as librosadisplay
from evaluate import get_waveform
from dataset import DatasetOutput
from trackers import OrignalPYin


TRACKERS = ['CREPE']
DATASET = DatasetOutput('IAPAS')
RES_DIR = consts.OUT_DIR / 'extra'

THRESHOLDS = {'CREPE': 0.6, 'PYIN': 0.5}


def plot_stft(audio, time, pitch, sample_rate):
  x_stft = np.abs(librosa.stft(audio, hop_length=256, n_fft=2048,  win_length=1024))
  x_stft_db = librosa.amplitude_to_db(x_stft, ref=np.max)
  librosadisplay.specshow(data=x_stft_db, x_axis="s", y_axis='log', sr=sample_rate, hop_length=256, cmap='coolwarm')
  plt.ylim(52, 1000)
  plt.plot(time, pitch)


def save_f0():
    methods = ['CREPE', 'PYIN']
    dfs = {method: pickle.load(open(DATASET.get_result(method), 'rb')) 
           for method in methods}

    for wav_path in DATASET.get_wavs():
        _, waveform = get_waveform(wav_path)
        file_ = wav_path.stem
        dir_ = RES_DIR / file_
        if not dir_.exists():
            dir_.mkdir()

        fig, axes = plt.subplots(len(methods), 1)

        for i, method in enumerate(methods):
            df = dfs[method]
            row = df[df.file == wav_path.stem].iloc[0]
            ts = np.vstack((row.time, row.pitch, row.confidence)).T
            np.savetxt(dir_ / f'{method}_F0.csv', ts, delimiter=',')

            pitch_voiced = np.where((row.pitch > 0) & (row.confidence > THRESHOLDS[method]), row.pitch, np.inf)

            axes[i].set_title(method)
            plt.sca(axes[i])
            plot_stft(waveform, row.time, pitch_voiced, consts.SR)
        
        fig.set_size_inches(50, 10)
        fig.savefig(dir_ / 'f0_spectrogram.jpg', dpi=400)
        plt.clf()


save_f0()

