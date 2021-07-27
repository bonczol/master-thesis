import seaborn as sns
from method import Tracker
from pathlib import Path
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
LABELS_BIN_FILE = "labels.pkl"

RAW_PATH = DATA_DIR / "raw"
PROCESSED_PATH = DATA_DIR / "processed"

RESULTS_PATH = OUT_DIR / "results"
PLOTS_PATH = OUT_DIR / "plots"
SPECTROGRAMS_PATH =  OUT_DIR / "spectrograms"

CHECKPOINTS_PATH =  DATA_DIR / "checkpoints"
DEGRADE_TMP_PATH = DATA_DIR / "degrade_tmp"

PROC_WAV_EXT = 'wav'
PROC_LABEL_EXT = 'csv'

'''
Plot params
'''
palette = sns.color_palette()

COLORS = {method.value: color for method, color in zip(Tracker, palette)}

LABELS = {
    Tracker.SPICE: 'SPICE',
    Tracker.CREPE: 'CREPE tiny',
    Tracker.DDSP_INV: 'DDSP-INV',
    Tracker.YIN: 'YIN',
    Tracker.SWIPE: 'SWIPE',
    Tracker.HF0: 'HF0',
    Tracker.PYIN: 'pYin'
}

THRESHOLDS = {
    Tracker.SPICE: 0.88,
    Tracker.CREPE: 0.67,
    Tracker.DDSP_INV: 0.5,
    Tracker.YIN: 0.5,
    Tracker.SWIPE: 0.5,
    Tracker.HF0: 0.5,
    Tracker.PYIN: 0.5
}

SR = 16000
MAX_ABS_INT16 = 32768.0

