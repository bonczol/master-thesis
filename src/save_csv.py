import consts
import pickle
import numpy as np
import utils
import crepe_mod.crepe as crepe
from evaluate import get_waveform
from dataset import DatasetOutput

trackers = ['CREPE']
dataset = DatasetOutput('IAPAS')

results_dfs = [pickle.load(open(dataset.get_result(t), 'rb'))
               for t in trackers]

# results_note_dfs = [pickle.load(open(dataset.get_result_notes(t), 'rb'))
#                     for t in trackers]


# f0 with confidence
for df in results_dfs:
    for i, row in df.iterrows():
        # pitch = row.pitch * (row.confidence > 0.6)
        ts = np.vstack((row.time, row.pitch, row.confidence)).T
        np.savetxt(consts.OUT_DIR / 'extra' / 'f0_confidence' /f'{row.file}.csv', ts, delimiter=',')

# f0
for wav_path in dataset.get_wavs():
    _, waveform = get_waveform(wav_path)
    audio = utils.normalize_peak(waveform)
    frames = crepe.get_frames(audio, 10)
    frames_rms = np.apply_along_axis(utils.rms, 1, frames)

    for df in results_dfs:
        row = df[df.file == wav_path.stem].iloc[0]

        pitch = row.pitch * ((row.confidence > 0.6) & (frames_rms > 0.1) & (row.pitch > 52))

        ts = np.vstack((row.time, pitch)).T
        np.savetxt(consts.OUT_DIR / 'extra' / 'f0' / f'{wav_path.stem}.csv', ts, delimiter=',')


# MIDI
# for path in Path(consts.RESULTS_NOTES_PATH / 'IAPAS').glob("*.pkl"):
#     with open(path, 'rb') as f:
#         df = pickle.load(f)

#     for i, row in df.iterrows():
#         # pitch = row.pitch * (row.confidence > 0.6)
#         ts = np.vstack((row.est_note_interval, row.est_note_pitch)).T
#         ts_path = consts.OUT_DIR / 'extra' / f'{row.file}.csv'
#         np.savetxt(ts_path, ts, delimiter=',')



