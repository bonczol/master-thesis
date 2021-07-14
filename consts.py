import seaborn as sns
from method import Tracker

'''
Project structure
'''
DATA_DIR = "data"
LIB_DIR = "lib"
OUT_DIR = "out"
RESOURCES_DIR = "resources"

WAV_DIR = "Wavfile"
LABEL_DIR = "PitchLabel"
LABELS_BIN_FILE = "labels.pkl"

RAW_PATH = f"{DATA_DIR}/raw"
PROCESSED_PATH = f"{DATA_DIR}/processed"

RESULTS_PATH = f"{OUT_DIR}/results"
PLOTS_PATH = f"{OUT_DIR}/plots"
SPECTROGRAMS_PATH = f"{OUT_DIR}/spectrograms"

CHECKPOINTS_PATH = f"{RESOURCES_DIR}/checkpoints"

'''
Plot params
'''
palette = sns.color_palette()

COLORS = {method.value: color for method, color in zip(Tracker, palette)}

LABELS = {
    Tracker.SPICE: 'SPICE',
    Tracker.CREPE_TINY: 'CREPE tiny',
    Tracker.DDSP_INV: 'DDSP-INV',
    Tracker.YIN: 'YIN',
}

THRESHOLDS = {
    Tracker.SPICE: 0.88,
    Tracker.CREPE_TINY: 0.67,
    Tracker.DDSP_INV: 0.5,
    Tracker.YIN: 0.5,
}

SR = 16000
MAX_ABS_INT16 = 32768.0

