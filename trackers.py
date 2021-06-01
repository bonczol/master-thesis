import tensorflow_hub as hub
import tensorflow as tf
import crepe2.crepe as crepe
import aubio
import utils
import numpy as np
import os
import gin
import ddsp
import ddsp.training
import time
from method import Method


class Tracker: 
    def __init__(self):
        self.step_size = 32
        self.hop = 512

    def predict(self, audio):
        pass

class Spice(Tracker):
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/spice/2")
        self.method = Method.SPICE
        super().__init__()

    def predict(self, audio):
        t0 = time.perf_counter()
        model_output = self.model.signatures["serving_default"](tf.constant(audio, tf.float32))
        pitch_pred = self._output2hz(model_output["pitch"])
        confidence_pred = 1 - model_output["uncertainty"]
        time_ = np.arange(pitch_pred.shape[0]) * self.step_size / 1000.0
        return time_, pitch_pred.numpy(), confidence_pred.numpy(), time.perf_counter() - t0

    def _output2hz(self, pitch_output):
        PT_OFFSET = 25.58
        PT_SLOPE = 63.07
        cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
        return utils.semitones2hz(cqt_bin)


class Crepe(Tracker):
    def __init__(self, version, sr):
        self.version = version
        self.sr = sr
        self.model = crepe.build_and_load_model(version)
        self.method = Method.CREPE_TINY
        super().__init__()

    def predict(self, audio):
        t0 = time.perf_counter()
        time_, pitch_pred, confidence_pred, _ = crepe.predict(audio, self.sr, self.model, step_size=32, verbose=0)
        return time_, pitch_pred, confidence_pred , time.perf_counter() - t0


class Yin(Tracker):
    def __init__(self, sr, threshold):
        self.method = Method.YIN
        self.yin = aubio.pitch("yinfft", 1024, 512, sr)
        self.yin.set_tolerance(threshold)
        super().__init__()


    def predict(self, audio):
        t0 = time.perf_counter()
        results = [self._nth_frame_pitch(audio, i) for i in range(0, len(audio) // self.hop)]
        pitch_pred, confidence_pred = [np.array(t) for t in zip(*results)]
        cmin, cmax = confidence_pred.min(), confidence_pred.max()
        confidence_pred = 1 - (confidence_pred - cmin) / (cmax - cmin + 0.0001)
        time_ = np.arange(pitch_pred.shape[0]) * self.step_size / 1000.0
        return time_, pitch_pred, confidence_pred, time.perf_counter() - t0


    def _nth_frame_pitch(self, audio, n):
        start = n * self.hop
        end = start + 512
        sample = audio[start:end]
        return self.yin(sample)[0], self.yin.get_confidence()


class InverseTracker(Tracker):
    def __init__(self, model_dir):
        self.method = Method.DDSP_INV
        self.model_dir = model_dir
        self.ckpt = self._get_checpoint_path()
        self.gin_file = os.path.join(self.model_dir, 'operative_config-1281000.gin')
        self.hop = 512

        with gin.unlock_config():
            gin.parse_config_file(self.gin_file, skip_unknown=True)

        super().__init__()

    def _get_checpoint_path(self):
        ckpt_files = [f for f in tf.io.gfile.listdir(self.model_dir) if 'ckpt' in f]
        ckpt_name = ckpt_files[0].split('.')[0]
        return os.path.join(self.model_dir, ckpt_name)

    def predict(self, audio):
        audio = audio.reshape(1, -1)
        time_steps = int(audio.shape[1] / self.hop)
        n_samples = time_steps * self.hop
        audio = audio[:, :n_samples]

        gin_params = [
            'TranscribingAutoencoder.n_samples = {}'.format(n_samples),
            'oscillator_bank.use_angular_cumsum = True',
        ]

        with gin.unlock_config():
            gin.parse_config(gin_params)

        model = ddsp.training.models.TranscribingAutoencoder()
        model.restore(self.ckpt)



        t0 = time.perf_counter()
        controls = model.get_controls({'audio': audio}, training=False)
        print(model.summary())

        evaluation_time =  time.perf_counter() - t0

        pitch_pred = controls['f0_hz']
        pitch_pred = pitch_pred.numpy().ravel()
        confidence_pred = np.ones(pitch_pred.shape)
        time_ = np.arange(pitch_pred.shape[0]) * self.step_size / 1000.0
        return time_, pitch_pred, confidence_pred, evaluation_time




            