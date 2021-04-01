import numpy as np
import os
from pydub import AudioSegment
from utils import semitones_to_hz
from mir_eval.melody import hz2cents

class Converter:
    def __init__(self, wav_path, label_path):
        self.wav_path = wav_path
        self.label_path = label_path
        self.sample_rate = 16000
        self.t0 = 0
        self.ts = 0.032

    def get_model_time(self, audio_duration):
        return np.arange(self.t0, audio_duration + 0.0001, self.ts)

    def get_audio(self):
        return AudioSegment.from_file(self.wav_path)

    def get_label(self):
        return np.loadtxt(self.label_path)


class MirConverter(Converter):
    def __init__(self, wav_path, label_path, voicing_path):
        self.voicing_path = voicing_path
        self.label_t0 = 0.02
        self.label_ts = 0.02
        super().__init__(wav_path, label_path)

    def get_voicing(self):
        return np.loadtxt(self.voicing_path)

    def convert_audio(self):
        audio = self.get_audio()
        return audio.split_to_mono()[1]

    def convert_label(self, audio_duration):
        labels = self.get_label()
        labels  = np.where(labels <= 0.0, labels, labels - 3.5)
        t_model = self.get_model_time(audio_duration)
        t_labels = np.arange(self.label_t0, audio_duration - (self.label_ts - 0.00001),  self.label_ts)
        labels_model = np.interp(t_model, t_labels, labels)
        voicing = self.get_voicing()
        voicing_model = np.interp(t_model, t_labels, voicing)
        voicing_model = np.round(voicing_model)
        return np.transpose(np.vstack((t_model, labels_model, voicing_model)))


class MdbConverter(Converter):
    def __init__(self, in_dir_wav, in_dir_label):
        super().__init__(in_dir_wav, in_dir_label)

    def convert_audio(self, file_name):
        audio = self.get_audio()
        return audio.set_frame_rate(self.sample_rate)

    def convert_label(self, file_name, audio_duration):
        labels = self.get_label()
        labels[:, 1] = hz2cents(labels[:, 1]) / 100.0
        t_model = self.get_model_time(audio_duration)
        labels_model = np.interp(t_model, labels[:, 0], labels[:, 1])
        voicing_model = np.where(labels > 0, 1, 0)
        return np.transpose(np.vstack((t_model, labels_model, voicing_model)))