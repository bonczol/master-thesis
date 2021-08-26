import os
import numpy as np
import mir_eval
import librosa
import scipy
import pandas as pd
import pretty_midi
from visual_midi import Plotter



def intervals_to_midi(intervals, pitches):
    melody = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for interval, pitch in zip(intervals, pitches):
        note = pretty_midi.Note(velocity=100, pitch=hz2midi(pitch), start=interval[0], end=interval[1])
        instrument.notes.append(note)

    melody.instruments.append(instrument)
    return melody


def read_label(path):
    ts = np.loadtxt(path, delimiter=",")
    f_name = os.path.splitext(os.path.basename(path))[0]
    return [f_name, ts[:,0], ts[:,1]]


def read_notes(path):
    notes = np.loadtxt(path, delimiter=",")
    return [notes[:, 0:2], notes[:, 2]]


def normalize_peak(audio):
    return audio / np.max(np.abs(audio))


def hz2midi(hz):
    return int(np.around(librosa.hz_to_midi(hz)))


def semitones2hz(semitones):
    FMIN = 10.0
    BINS_PER_OCTAVE = 12.0
    return FMIN * 2.0 ** (1.0 * semitones / BINS_PER_OCTAVE)


def rpa_multi_tolerance(ref_voicing, ref_cent, est_voicing, est_cent,
                       cent_tolerances):

    mir_eval.melody.validate_voicing(ref_voicing, est_voicing)
    mir_eval.melody.validate(ref_voicing, ref_cent, est_voicing, est_cent)
 
    if ref_voicing.size == 0 or ref_voicing.sum() == 0 \
       or ref_cent.size == 0 or est_cent.size == 0:
        return 0.

    nonzero_freqs = np.logical_and(est_cent != 0, ref_cent != 0)

    if sum(nonzero_freqs) == 0:
        return 0.

    freq_diff_cents = np.abs(ref_cent - est_cent)[nonzero_freqs]

    voiced = np.sum(ref_voicing)
    
    return [np.sum(ref_voicing[nonzero_freqs] * (freq_diff_cents < t)) / voiced
             for t in cent_tolerances]



def resample_zeros(times, frequencies, times_new):
    frequencies_held = np.array(frequencies)
    for n, frequency in enumerate(frequencies[1:]):
        if frequency == 0:
            frequencies_held[n + 1] = frequencies_held[n]

    frequencies_resampled = scipy.interpolate.interp1d(times, frequencies_held, 'linear', fill_value="extrapolate")(times_new)
    frequency_mask = scipy.interpolate.interp1d(times, frequencies, 'zero', fill_value="extrapolate")(times_new)
    frequencies_resampled *= (frequency_mask != 0)
    return frequencies_resampled



def rca_multi_tolerance(ref_voicing, ref_cent, est_voicing, est_cent,
                        cent_tolerances):

    mir_eval.melody.validate_voicing(ref_voicing, est_voicing)
    mir_eval.melody.validate(ref_voicing, ref_cent, est_voicing, est_cent)
    # When input arrays are empty, return 0 by special case
    # If there are no voiced frames in reference, metric is 0
    if ref_voicing.size == 0 or ref_voicing.sum() == 0 \
       or ref_cent.size == 0 or est_cent.size == 0:
        return 0.

    # # Raw chroma = same as raw pitch except that octave errors are ignored.
    nonzero_freqs = np.logical_and(est_cent != 0, ref_cent != 0)

    if sum(nonzero_freqs) == 0:
        return 0.

    freq_diff_cents = np.abs(ref_cent - est_cent)[nonzero_freqs]
    octave = 1200.0 * np.floor(freq_diff_cents / 1200 + 0.5)
    voiced = np.sum(ref_voicing)
    return (
        [np.sum(ref_voicing[nonzero_freqs] * (np.abs(freq_diff_cents - octave) < t)) / voiced
         for t in cent_tolerances]
    )


def explode_custom(data, cols_to_explode):
    cols_to_preserve = list(set(data.columns) - set(cols_to_explode))

    flat_df = pd.concat([data[cols_to_preserve + [col]].explode(col) 
                            for col in cols_to_explode], axis=1)
    return flat_df.loc[:,~flat_df.columns.duplicated()]

