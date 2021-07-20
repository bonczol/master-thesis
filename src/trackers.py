import os
from re import A
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow_hub as hub
import tensorflow as tf
import crepe_mod.crepe as crepe
import aubio
import utils
import numpy as np
import gin
import ddsp
import ddsp.training
import time
import consts
import pysptk
import matlab.engine
import librosa
from method import Tracker
from pathlib import Path


class AbstractTracker: 
    def __init__(self):
        self.step_size = 32
        self.hop = 512

    def predict(self, audio):
        pass

class Spice(AbstractTracker):
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/spice/2")
        self.method = Tracker.SPICE
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


class Crepe(AbstractTracker):
    def __init__(self, version='tiny'):
        self.version = version
        self.model = crepe.build_and_load_model(version)
        self.method = Tracker.CREPE_TINY
        super().__init__()

    def predict(self, audio):
        t0 = time.perf_counter()
        time_, pitch_pred, confidence_pred, _ = crepe.predict(audio, consts.SR, self.model, step_size=32, verbose=0)
        return time_, pitch_pred, confidence_pred , time.perf_counter() - t0


class Yin(AbstractTracker):
    def __init__(self, threshold=0.8):
        self.method = Tracker.YIN
        self.threshold = threshold
        super().__init__()


    def predict(self, audio):
        yin = aubio.pitch("yinfft", 1024, 512, consts.SR)
        yin.set_tolerance(self.threshold)

        t0 = time.perf_counter()
        results = [self._nth_frame_pitch(yin, audio, i) for i in range(0, len(audio) // self.hop)]
        pitch_pred, confidence_pred = [np.array(t) for t in zip(*results)]
        cmin, cmax = confidence_pred.min(), confidence_pred.max()
        confidence_pred = 1 - (confidence_pred - cmin) / (cmax - cmin + 0.0001)
        time_ = np.arange(pitch_pred.shape[0]) * self.step_size / 1000.0
        return time_, pitch_pred, confidence_pred, time.perf_counter() - t0


    def _nth_frame_pitch(self, yin, audio, n):
        start = n * self.hop
        end = start + 512
        sample = audio[start:end]
        return yin(sample)[0], yin.get_confidence()


class InverseTracker(AbstractTracker):
    def __init__(self, ckpt_version='urmp_ckpt'):
        self.method = Tracker.DDSP_INV
        self.model_dir = Path(consts.CHECKPOINTS_PATH) / ckpt_version
        self.ckpt = self._get_checpoint_path()
        self.gin_file = self.model_dir / 'operative_config-1281000.gin'
        self.hop = 512

        with gin.unlock_config():
            gin.parse_config_file(self.gin_file, skip_unknown=True)

        super().__init__()

    def _get_checpoint_path(self):
        ckpt_files = [f for f in tf.io.gfile.listdir(self.model_dir) if 'ckpt' in f]
        ckpt_name = ckpt_files[0].split('.')[0]
        return self.model_dir / ckpt_name

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
        # print(model.summary())

        evaluation_time =  time.perf_counter() - t0

        pitch_pred = controls['f0_hz']
        pitch_pred = pitch_pred.numpy().ravel()
        confidence_pred = np.ones(pitch_pred.shape)
        time_ = np.arange(pitch_pred.shape[0]) * self.step_size / 1000.0
        return time_, pitch_pred, confidence_pred, evaluation_time


class Swipe(AbstractTracker):
    def __init__(self):
        self.method = Tracker.SWIPE
        super().__init__()

    def predict(self, audio):
        t0 = time.perf_counter()
        audio = audio.astype(dtype=np.float64)
        pitch_pred = pysptk.sptk.swipe(audio, consts.SR, self.hop, min=32, max=2000, threshold=0)
        confidence_pred = np.ones(pitch_pred.shape)
        time_ = np.arange(pitch_pred.shape[0]) * self.step_size / 1000.0
        return time_, pitch_pred, confidence_pred, time.perf_counter() - t0


class Hf0(AbstractTracker):
    def __init__(self):
        self.method = Tracker.HF0
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(str(consts.SRC_DIR / 'hf0_mod'))
        self.eng.mirverbose(0)
        self.tracker = self.eng.getfield(self.eng.load('convModel.mat'), 'convnet')
        super().__init__()

    def predict(self, audio):
        t0 = time.perf_counter()
        audio = matlab.double(audio.tolist())
        pitch_pred, time_, _ = self.eng.PitchExtraction(audio, self.tracker, nargout=3)
        pitch_pred, time_ = np.array(pitch_pred).flatten(), np.array(time_).flatten()
        confidence_pred = (pitch_pred > 0).astype(np.int)
        return time_, pitch_pred, confidence_pred, time.perf_counter() - t0


class PYin(AbstractTracker):
    def __init__(self):
        self.method = Tracker.PYIN
        super().__init__()

    def predict(self, audio):
        t0 = time.perf_counter()
        pitch_pred, voiced_flags, _ = librosa.pyin(audio, fmin=32, fmax=2000, sr=consts.SR)
        time_ = np.arange(pitch_pred.shape[0]) * self.step_size / 1000.0
        return time_, pitch_pred, voiced_flags, time.perf_counter() - t0