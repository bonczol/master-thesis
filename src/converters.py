import numpy as np
import shutil
import array
import pickle
import utils
import pandas as pd
from multiprocessing import Pool
from pydub import AudioSegment
from utils import semitones2hz
from pathlib import Path
import consts
import mir_eval


class Converter:
    def __init__(self, raw, proc):
        self.raw = raw        
        self.proc = proc

    def _convert_args(self):
        return zip(self.raw.get_wavs(), self.raw.get_labels(), 
                   self.proc.get_wavs(), self.proc.get_labels())

    def _convert_func(self):
        pass

    def _convert(self):
        with Pool() as pool:
            pool.starmap(self._convert_func(), self._convert_args())

    def _save_labels_binary(self):
        with Pool() as pool:
            labels = pool.map(utils.read_label, self.proc.get_labels())
   
        labels_df = pd.DataFrame(labels, columns=['file', 'label_time', 'label_pitch'])
        labels_df['duration'] = labels_df['label_time'].map(lambda x: x[-1])
        with open(self.proc.label_bin_path, 'wb') as f:
            pickle.dump(labels_df, f)


    def prepare(self):
        self._convert()
        self._save_labels_binary()



class MirConverter(Converter):
    def __init__(self, raw, proc):
        self.label_t0 = 0.02
        self.label_ts = 0.02
        super().__init__(raw, proc)

    def _convert_args(self):
        return zip(self.raw.get_wavs(), self.raw.get_labels(), self.proc.get_wavs(), 
                   self.proc.get_labels(), self.proc.get_background())

    def _convert_func(self):
        return self._convert_example

    def _convert_example(self, wav_path, label_path, out_wav_path, out_label_path, out_wav_background_path):
        audio = AudioSegment.from_file(wav_path)
        background, audio = audio.split_to_mono()

        freq_est = np.loadtxt(label_path, delimiter=',')
        freq_est  = np.where(freq_est <= 0.0, freq_est, semitones2hz(freq_est - 3.5))
        t = np.arange(self.label_t0, audio.duration_seconds - (self.label_ts - 0.00001),  self.label_ts)

        time_series = np.transpose(np.vstack((t, freq_est)))

        audio.export(out_wav_path, format='wav')
        background.export(out_wav_background_path, format='wav')
        np.savetxt(out_label_path, time_series, delimiter=',', fmt='%1.6f')


class MdbConverter(Converter):
    def __init__(self, dataset_raw, datset_proc):
        super().__init__(dataset_raw, datset_proc)

    def _convert_func(self):
        return self._convert_example

    def _convert_example(self, wav_path, label_path, out_wav_path, out_label_path):
        audio = AudioSegment.from_file(wav_path)
        audio = audio.set_frame_rate(consts.SR)
        audio.export(out_wav_path, format='wav')
        shutil.copyfile(label_path, out_label_path)


class UrmpConverter(Converter):
    def __init__(self, raw, proc):
        super().__init__(raw, proc)

    def _convert_func(self):
        return self._convert_example

    def _convert_example(self, wav_path, label_path, out_wav_path, out_label_path):
        audio = AudioSegment.from_file(wav_path)
        audio = audio.set_frame_rate(consts.SR)
        audio.export(out_wav_path, format='wav')
    
        label = np.loadtxt(label_path, delimiter='\t')
        np.savetxt(out_label_path, label, delimiter=',', fmt='%1.6f')



class PtdbConverter(Converter):
    def __init__(self, raw, proc):
        super().__init__(raw, proc)


    def _convert_func(self):
        return self._convert_example

    def _convert_example(self, wav_path, label_path, out_wav_path, out_label_path):
        audio = AudioSegment.from_file(wav_path)
        audio = audio.set_frame_rate(consts.SR)
    
        label = np.loadtxt(label_path, delimiter=' ')
        pitch = label[:, 0]
        time = np.arange(pitch.shape[0]) * 0.01
        timeseries = np.transpose(np.vstack((time, pitch)))

        audio.export(out_wav_path, format='wav')
        np.savetxt(out_label_path, timeseries, delimiter=',', fmt='%1.6f')

