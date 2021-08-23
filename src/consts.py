import seaborn as sns
import pandas as pd
from method import Method
from pathlib import Path
from pandas.api.types import CategoricalDtype

'''
Project structure
'''

# WD = Path.cwd().parent
WD = Path('C:/Users/Filip/Projects/master-thesis')
DATA_DIR = WD / "data"
OUT_DIR = WD / "out"
SRC_DIR = WD / "src"

WAV_DIR = "Wavfile"
LABEL_DIR = "PitchLabel"
NOTES_DIR = "Notes"
LABELS_BIN_FILE = "labels.pkl"

RAW_PATH = DATA_DIR / "raw"
PROCESSED_PATH = DATA_DIR / "processed"
SPECTROGRAMS_WAV_PATH = DATA_DIR / "spectrograms"

RESULTS_PATH = OUT_DIR / "results"
RESULTS_NOTES_PATH = OUT_DIR / "results_notes"
PLOTS_PATH = OUT_DIR / "plots"
SPECTROGRAMS_PATH =  OUT_DIR / "spectrograms"
PYIN_TMP = OUT_DIR / 'pyin_tmp'
POST_RESULTS_PATH = OUT_DIR / "post_results"

CHECKPOINTS_PATH =  DATA_DIR / "checkpoints"
DEGRADE_TMP_PATH = DATA_DIR / "degrade_tmp"
TRANS_PATH = DATA_DIR / 'trans'


PROC_WAV_EXT = 'wav'
PROC_LABEL_EXT = 'csv'
PROC_NOTE_EXT = 'csv'

'''
Plot params
'''
palette = sns.color_palette()

COLORS = {method.value: color for method, color in zip(Method, palette)}
METHODS_VAL = [m.value for m in list(Method)]
DS_ORDER = ['MIR-1k', 'MDB-stem-synth', 'URMP']

DS_CAT = CategoricalDtype(categories=DS_ORDER, ordered=True)
METHOD_CAT = CategoricalDtype(categories=METHODS_VAL, ordered=True)

LABELS = {
    Method.SPICE: 'SPICE',
    Method.CREPE: 'CREPE tiny',
    Method.DDSP_INV: 'DDSP-INV',
    Method.SWIPE: 'SWIPE',
    Method.HF0: 'HF0',
    Method.PYIN: 'pYin'
}


THRESHOLDS = {
    Method.SPICE: 0.88,
    Method.CREPE: 0.67,
    Method.DDSP_INV: 0.5,
    Method.SWIPE: 0.5,
    Method.HF0: 0.5,
    Method.PYIN: 0.5
}

SR = 16000
MAX_ABS_INT16 = 32768.0

