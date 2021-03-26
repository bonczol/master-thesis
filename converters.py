import numpy as np
from pydub import AudioSegment
from utils import semitones_to_hz
from mir_eval.melody import hz2cents

class Converter:
    def __init__(self):
        self.sample_rate = 16000
        self.t0 = 0
        self.ts = 0.032

    def get_model_time(self, audio_duration):
        return np.arange(self.t0, audio_duration + 0.0001, self.ts)


class MirConverter(Converter):
    def __init__(self):
        self.label_t0 = 0.02
        self.label_ts = 0.02
        super().__init__()

    def convert_audio(self, audio):
        return audio.split_to_mono()[1]

    def convert_label(self, labels, audio_duration):
        labels  = np.where(labels <= 0.0, labels, labels - 3.5)
        t_model = self.get_model_time(audio_duration)
        t_labels = np.arange(self.label_t0, audio_duration - (self.label_ts - 0.00001),  self.label_ts)
        labels_model = np.interp(t_model, t_labels, labels)
        return np.transpose(np.vstack((t_model, labels_model)))


class MdbConverter(Converter):
    def convert_audio(self, audio):
        audio = audio.set_frame_rate(self.sample_rate)
        return audio

    def convert_label(self, labels, audio_duration):
        labels[:, 1] = hz2cents(labels[:, 1]) / 100.0
        t_model = self.get_model_time(audio_duration)
        labels_model = np.interp(t_model, labels[:, 0], labels[:, 1])
        return np.transpose(np.vstack((t_model, labels_model)))