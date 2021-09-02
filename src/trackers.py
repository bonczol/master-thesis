from abc import abstractmethod
import tensorflow_hub as hub
import crepe_mod.crepe as crepe
import aubio
from method import Method
import utils
import numpy as np
import gin
import ddsp
import ddsp.training
import time
import consts
import pysptk
# import matlab.engine
from pathlib import Path
import tensorflow as tf
import subprocess

class AbstractMethod: 
    def __init__(self, method, is_multicore):
        self.method = method
        self.is_multicore = is_multicore

    @abstractmethod
    def predict(self, audio):
        pass


class Spice(AbstractMethod):
    def __init__(self):
        self.step_size = 32
        self.model = hub.load("https://tfhub.dev/google/spice/2")
        super().__init__(Method.SPICE, is_multicore=False)

    def _predict_chunk(self, chunk):
        model_output = self.model.signatures["serving_default"](tf.constant(chunk, tf.float32))
        pitch = self._output2hz(model_output["pitch"])
        confidence = 1 - model_output["uncertainty"]
        return pitch.numpy().tolist(), confidence.numpy().tolist()

    def predict(self, audio):
        t0 = time.perf_counter()

        frame_length =  512000

        frames = [audio[i*frame_length:(i+1)*frame_length] 
                  for i in range(int(np.ceil(len(audio)/frame_length)))]

        merged_pitch, merged_confidence = self._predict_chunk(frames[0])

        for frame in frames[1:]:
            pitch, confidence = self._predict_chunk(frame)
            merged_pitch.extend(pitch[1:])
            merged_confidence.extend(confidence[1:])

        merged_pitch, merged_confidence  = np.array(merged_pitch), np.array(merged_confidence)
        time_ = np.arange(merged_pitch.shape[0]) * self.step_size / 1000.0

        return time_, merged_pitch, merged_confidence, time.perf_counter() - t0

    def _output2hz(self, pitch_output):
        PT_OFFSET = 25.58
        PT_SLOPE = 63.07
        cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
        return utils.semitones2hz(cqt_bin)


class Crepe(AbstractMethod):
    def __init__(self, step_size=10, version='full', smooth_pitch=False):
        self.step_size = step_size
        self.smooth_pitch = smooth_pitch
        self.model = crepe.build_and_load_model(version)
        super().__init__(Method.CREPE, is_multicore=False)

    def predict(self, audio):
        t0 = time.perf_counter()
        time_, pitch_pred, confidence_pred, _ = crepe.predict(audio, consts.SR, self.model, step_size=self.step_size,
                                                              viterbi=self.smooth_pitch, verbose=0)
        return time_, pitch_pred, confidence_pred , time.perf_counter() - t0


class Yin(AbstractMethod):
    def __init__(self, threshold=0.8):
        self.step_size = 32
        self.hop = 512
        self.threshold = threshold
        super().__init__(Method.CREPE, is_multicore=False)


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


class InverseTracker(AbstractMethod):
    def __init__(self, ckpt_version='urmp_ckpt'):
        self.model_dir = Path(consts.CHECKPOINTS_PATH) / ckpt_version
        self.ckpt = self._get_checpoint_path()
        self.gin_file = self.model_dir / 'operative_config-1281000.gin'
        self.hop = 512
        self.step_size = 32

        with gin.unlock_config():
            gin.parse_config_file(self.gin_file, skip_unknown=True)

        super().__init__(Method.DDSP_INV, is_multicore=False)

    def _get_checpoint_path(self):
        ckpt_files = [f for f in tf.io.gfile.listdir(self.model_dir) if 'ckpt' in f]
        ckpt_name = ckpt_files[0].split('.')[0]
        return self.model_dir / ckpt_name


    def _setup_model(self, n_samples):
        gin_params = [
            'TranscribingAutoencoder.n_samples = {}'.format(n_samples),
            'oscillator_bank.use_angular_cumsum = True',
        ]

        with gin.unlock_config():
            gin.parse_config(gin_params)

        model = ddsp.training.models.TranscribingAutoencoder()
        model.restore(self.ckpt)
        return model

    def _predict_chunk(self, model, audio):
        controls = model.get_controls({'audio': audio}, training=False)
        return controls['f0_hz'].numpy().ravel()


    def predict(self, audio):
        total_time = 0

        # Round to 512
        time_steps = int(audio.shape[0] / self.hop)
        n_samples = time_steps * self.hop
        audio = audio[:n_samples]
        
        frame_length =  512000
        frames = [audio[i*frame_length:(i+1)*frame_length] 
                  for i in range(int(np.ceil(len(audio)/frame_length)))]

        if len(frames[0]) == frame_length:
            model = self._setup_model(frame_length)

        merged_pitch = []
        for frame in frames:
            if len(frame) < frame_length:
                model = self._setup_model(len(frame))

            frame = frame.reshape(1, -1)

            start = time.perf_counter()
            merged_pitch.append(self._predict_chunk(model, frame))
            total_time += time.perf_counter() - start

        merged_pitch = np.concatenate(merged_pitch)
        confidence_pred = np.ones(merged_pitch.shape)
        time_ = np.arange(merged_pitch.shape[0]) * self.step_size / 1000.0

        return time_, merged_pitch, confidence_pred, total_time


class Swipe(AbstractMethod):
    def __init__(self):
        self.step_size = 32
        self.hop = 512
        super().__init__(Method.SWIPE, is_multicore=True)

    def predict(self, audio):
        t0 = time.perf_counter()
        audio = audio.astype(dtype=np.float64)
        pitch_pred = pysptk.sptk.swipe(audio, consts.SR, self.hop, min=32, max=2000, threshold=0)
        confidence_pred = np.ones(pitch_pred.shape)
        time_ = np.arange(pitch_pred.shape[0]) * self.step_size / 1000.0
        return time_, pitch_pred, confidence_pred, time.perf_counter() - t0


# class Hf0(AbstractMethod):
#     def __init__(self):
#         self.eng = matlab.engine.start_matlab()
#         self.eng.cd(str(consts.SRC_DIR / 'hf0_mod'))
#         self.eng.mirverbose(0)
#         self.tracker = self.eng.getfield(self.eng.load('convModel.mat'), 'convnet')
#         super().__init__(Method.HF0, is_multicore=False)

#     def predict(self, audio):
#         t0 = time.perf_counter()
#         audio = matlab.double(audio.tolist())
#         pitch_pred, time_, _ = self.eng.PitchExtraction(audio, self.tracker, nargout=3)
#         pitch_pred, time_ = np.array(pitch_pred).flatten(), np.array(time_).flatten()
#         confidence_pred = (pitch_pred > 0).astype(np.int)
#         return time_, pitch_pred, confidence_pred, time.perf_counter() - t0



class OrignalPYin(AbstractMethod):
    def __init__(self):
        super().__init__(Method.PYIN, is_multicore=True)

    def predict(self, wav_path):
        t0 = time.perf_counter()
        cmd = ['sonic-annotator',
                '-t', consts.TRANS_PATH / 'pyin.n3',
                '-w', 'csv',
                '--csv-basedir', consts.PYIN_TMP,
                '--csv-force',
                wav_path,
                '--normalise']
        # subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd)

        total_time = time.perf_counter() - t0

        output_path = consts.PYIN_TMP / f'{wav_path.stem}_vamp_pyin_pyin_smoothedpitchtrack.csv'
        output_file = np.loadtxt(output_path, delimiter=',')

        time_ = output_file[:,0]
        pitch_pred = output_file[:,1]
        pitch_pred = np.where(pitch_pred > 0, pitch_pred, 0)
        confidence_pred = (pitch_pred > 0).astype(np.int)

        return time_, pitch_pred, confidence_pred, total_time

