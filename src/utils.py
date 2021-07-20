import os
import re
import argparse
import numpy as np
import configparser
import mir_eval
import scipy


def resample_zeros(times, frequencies, times_new):
    frequencies_held = np.array(frequencies)
    for n, frequency in enumerate(frequencies[1:]):
        if frequency == 0:
            frequencies_held[n + 1] = frequencies_held[n]

    frequencies_resampled = scipy.interpolate.interp1d(times, frequencies_held, 'linear', fill_value="extrapolate")(times_new)
    frequency_mask = scipy.interpolate.interp1d(times, frequencies, 'zero', fill_value="extrapolate")(times_new)
    frequencies_resampled *= (frequency_mask != 0)
    return frequencies_resampled


def get_parser_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_name', type=str)
    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read('consts.conf')
    return parser, conf


def read_label(path):
    ts = np.loadtxt(path, delimiter=",")
    f_name = os.path.splitext(os.path.basename(path))[0]
    return [f_name, ts[:,0], ts[:,1]]


def get_wav_paths(dir_):
    return sorted([os.path.join(dir_, f) for f in  os.listdir(dir_) 
                        if not f.startswith(".") and f.endswith(".wav")])


def get_time_series_paths(dir_):
    return sorted([os.path.join(dir_, f) for f in  os.listdir(dir_) 
                        if not f.startswith(".") and re.match(r".*\.(pv|csv)$", f)])


def get_vocal_paths(dir_):
    return sorted([os.path.join(dir_, f) for f in  os.listdir(dir_) 
                        if not f.startswith(".") and re.match(r".*\.(vocal)$", f)])


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
