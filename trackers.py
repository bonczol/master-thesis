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


class Tracker: 
    def predict(self, audio):
        pass
        

class Spice(Tracker):
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/spice/2")
        super().__init__()

    def predict(self, audio):
        t0 = time.perf_counter()
        model_output = self.model.signatures["serving_default"](tf.constant(audio, tf.float32))
        pitch_pred = self._output2hz(model_output["pitch"])
        confidence_pred = 1 - model_output["uncertainty"]
        return pitch_pred.numpy(), confidence_pred.numpy(), time.perf_counter() - t0

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
        super().__init__()

    def predict(self, audio):
        t0 = time.perf_counter()
        _, pitch_pred, confidence_pred, _ = crepe.predict(audio, self.sr, self.model, step_size=32, verbose=0)
        return pitch_pred, confidence_pred , time.perf_counter() - t0


class Yin(Tracker):
    def __init__(self, sr, threshold):
        self.yin = aubio.pitch("yinfft", 1024, 512, sr)
        self.yin.set_tolerance(threshold)
        self.hop = 512
        super().__init__()


    def predict(self, audio):
        t0 = time.perf_counter()
        n = len(audio) // self.hop + 1
        pitch_pred = np.zeros(n)
        confidence_pred = np.zeros(n)

        for i in range(0, len(audio) // self.hop):
            start = i * self.hop
            end = start + 512
            sample = audio[start:end]
            pitch_pred[i] = self.yin(sample)[0]
            confidence_pred[i] = self.yin.get_confidence()

        cmin = confidence_pred.min()
        cmax = confidence_pred.max()

        confidence_pred = 1 - (confidence_pred - cmin) / (cmax - cmin + 0.0001)
        return pitch_pred, confidence_pred, time.perf_counter() - t0



class InverseTracker(Tracker):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.ckpt = self._get_checpoint_path
        self.gin_file = 'urmp_ckpt/operative_config-1281000.gin'
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

        controls = model.get_controls({'audio': audio}, training=False)
        pitch_pred = controls['f0_hz']
        confidence_pred = np.ones(pitch_pred)
        return pitch_pred, confidence_pred




            