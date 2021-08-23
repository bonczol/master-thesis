import numpy as np
from method import Method
from trackers import Crepe
import crepe_mod.crepe as crepe
import consts
import pretty_midi
import utils
import librosa
import subprocess
import matplotlib.pyplot as plt
from trackers import AbstractMethod
from method import Method

 

def rms(y):
    return np.sqrt(np.mean(y**2))


def intervals_to_midi(intervals):
    melody = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for interval in intervals:
        note = pretty_midi.Note(velocity=100, pitch=interval[2], start=interval[0], end=interval[1])
        instrument.notes.append(note)

    melody.instruments.append(instrument)
    return melody

    
class CrepeTrans(AbstractMethod):
    def __init__(self, step_size, confidence_t, amplitude_t):
        super().__init__(Method.CREPE_MIDI, is_multicore=False)
        self.step_size = step_size
        self.confidence_t = confidence_t
        self.amplitude_t = amplitude_t
        self.tracker = Crepe(step_size=step_size, smooth_pitch=True)


class CrepeHmmTrans(CrepeTrans):
    def __init__(self, step_size=10, confidence_t=0.5, amplitude_t=0.1):
        super().__init__(step_size, confidence_t, amplitude_t)

    def predict(self, audio):
        _, pitch, confidence, _ = self.tracker.predict(audio)

        audio = utils.normalize_peak(audio)
        frames = crepe.get_frames(audio, self.step_size)
        frames_rms = np.apply_along_axis(rms, 1, frames)

        # voicing = crepe.predict_voicing(confidence)
        
        pitch = pitch * ((confidence > self.confidence_t) & (frames_rms > self.amplitude_t))

        return [], []



class PyinTrans(AbstractMethod):
    def __init__(self):
        super().__init__(Method.PYIN_MIDI, is_multicore=True)
        
    def predict(self, audio):
        wav_path = audio
        cmd = ['sonic-annotator',
                '-t', consts.TRANS_PATH / 'pyin_note.n3',
                '-w', 'csv',
                '--csv-basedir', consts.PYIN_TMP,
                '--csv-force',
                wav_path,
                '--normalise']

        subprocess.run(cmd)

        output_path = consts.PYIN_TMP / f'{wav_path.stem}_vamp_pyin_pyin_notes.csv'
        output_file = np.loadtxt(output_path, delimiter=',')
        intervals = output_file[:, 0:2]
        intervals[:,1] = intervals[:,0] + intervals[:,1]
        pitch = output_file[:, 2]
        return intervals, pitch
