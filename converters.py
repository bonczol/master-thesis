import numpy as np
import os
import shutil
from pydub import AudioSegment
from mir_eval.melody import hz2cents
from pydub.utils import mediainfo
from utils import semitones2hz, resample_zeros
import scipy


class Converter:
    def __init__(self, wav_path, label_path, out_wav_path, out_label_path):
        self.wav_path = wav_path
        self.label_path = label_path
        self.out_wav_path = out_wav_path
        self.out_label_path = out_label_path
        self.sample_rate = 16000

    def get_audio(self):
        return AudioSegment.from_file(self.wav_path)

    def get_label(self):
        return np.loadtxt(self.label_path)
        

class MirConverter(Converter):
    def __init__(self, wav_path, label_path, out_wav_path, out_label_path):
        self.label_t0 = 0.02
        self.label_ts = 0.02
        super().__init__(wav_path, label_path, out_wav_path, out_label_path)

    def convert(self):
        audio = self.get_audio()
        audio = audio.split_to_mono()[1]

        freq_est = self.get_label()
        freq_est  = np.where(freq_est <= 0.0, freq_est, semitones2hz(freq_est - 3.5))
        t = np.arange(self.label_t0, audio.duration_seconds - (self.label_ts - 0.00001),  self.label_ts)

        time_series = np.transpose(np.vstack((t, freq_est)))

        audio.export(self.out_wav_path, format='wav')
        np.savetxt(self.out_label_path, time_series, delimiter=',', fmt='%1.6f')


class MDBConverter(Converter):
    def __init__(self, wav_path, label_path, out_wav_path, out_label_path):
        super().__init__(wav_path, label_path, out_wav_path, out_label_path)

    def convert(self):
        audio = self.get_audio()
        audio = audio.set_frame_rate(self.sample_rate)
        audio.export(self.out_wav_path, format='wav')
        shutil.copyfile(self.label_path, self.out_label_path)


class DummyConverter(Converter):
    def __init__(self, wav_path, label_path, out_wav_path, out_label_path):
        super().__init__(wav_path, label_path, out_wav_path, out_label_path)

    def convert(self):
        shutil.copyfile(self.wav_path, self.out_wav_path)
        shutil.copyfile(self.label_path, self.out_label_path)


        
