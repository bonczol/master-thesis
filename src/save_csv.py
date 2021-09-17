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


TRACKERS = ['CREPE']
DATASET = DatasetOutput('IAPAS')
RES_DIR = consts.OUT_DIR / 'extra'


def plot_stft(audio, time, pitch, sample_rate):
  x_stft = np.abs(librosa.stft(audio, hop_length=256, n_fft=2048,  win_length=1024))
  x_stft_db = librosa.amplitude_to_db(x_stft, ref=np.max)
  librosadisplay.specshow(data=x_stft_db, x_axis="s", y_axis='log', sr=sample_rate, hop_length=256, cmap='coolwarm')
  plt.ylim(52, 1000)
  plt.plot(time, pitch)


def save_f0_crepe():
    df = pickle.load(open(DATASET.get_result('CREPE'), 'rb'))

    # f0 with confidence
    for _, row in df.iterrows():

        dir_ = RES_DIR / row.file
        if not dir_.exists():
            dir_.mkdir()

        ts = np.vstack((row.time, row.pitch, row.confidence)).T
        np.savetxt(dir_ / 'f0_confidence.csv', ts, delimiter=',')

    # f0
    for wav_path in DATASET.get_wavs():
        _, waveform = get_waveform(wav_path)
        audio = waveform

        frames = crepe.get_frames(audio, 10)
        frames_rms = np.apply_along_axis(utils.rms, 1, frames)
        frames_rms = (frames_rms - frames_rms.mean()) / frames_rms.std()

        row = df[df.file == wav_path.stem].iloc[0]

        dir_ = RES_DIR / row.file
        if not dir_.exists():
            dir_.mkdir()

        pitch = row.pitch * ((row.confidence > 0.6) & (frames_rms > -1) & (row.pitch > 52) & (row.pitch < 3000))
        # pitch = row.pitch
        

        ts = np.vstack((row.time, pitch)).T
        np.savetxt(dir_ / 'f0.csv', ts, delimiter=',')
        
        pitch = np.where(pitch > 0, pitch, np.inf)
        plot_stft(audio, row.time, pitch, consts.SR)
        fig = plt.gcf()
        fig.set_size_inches(50, 10)
        fig.savefig(dir_ / 'f0_spectrogram.jpg', dpi=400)
        plt.clf()



save_f0_crepe()



    # results_note_dfs = [pickle.load(open(dataset.get_result_notes(t), 'rb'))
    #                     for t in trackers]


# MIDI
# for path in Path(consts.RESULTS_NOTES_PATH / 'IAPAS').glob("*.pkl"):
#     with open(path, 'rb') as f:
#         df = pickle.load(f)

#     for i, row in df.iterrows():
#         # pitch = row.pitch * (row.confidence > 0.6)
#         ts = np.vstack((row.est_note_interval, row.est_note_pitch)).T
#         ts_path = consts.OUT_DIR / 'extra' / f'{row.file}.csv'
#         np.savetxt(ts_path, ts, delimiter=',')



