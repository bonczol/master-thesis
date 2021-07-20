import consts
from pathlib import Path


class DatasetInput:
    def __init__(self, name, label_ext, wav_dir='', label_dir='', wav_prefix='', label_prefix=''):
        self.name = name
        self._label_ext = label_ext
        self._dataset_dir = Path(f'{consts.RAW_PATH}/{name}')
        # self._wav_path = Path(f'{consts.RAW_PATH}/{name}/{wav_dir}')
        # self._label_path = Path(f'{consts.RAW_PATH}/{name}/{label_dir}')
        self._wav_dir = wav_dir
        self._label_dir = label_dir
        self._wav_prefix = wav_prefix
        self._label_prefix = label_prefix
        
    def get_files(self):
        return [f.stem for f in self.get_wavs()]

    def get_wavs(self):
        return sorted([p for p in self._dataset_dir.glob(f'**/{self._wav_dir}/**/*.wav') 
                       if not str(p.name).startswith('.') 
                       and str(p.name).startswith(self._wav_prefix)])

    def get_labels(self):
        return sorted([p for p in self._dataset_dir.glob(f'**/{self._label_dir}/**/*.{self._label_ext}')
                       if not str(p.name).startswith('.') 
                       and str(p.name).startswith(self._label_prefix)])

    


class DatasetOutput:
    def __init__(self, name, files=None):
        self.name = name

        # data
        self._wav_path = Path(f'{consts.PROCESSED_PATH}/{name}/{consts.WAV_DIR}')
        self._wav_bg_path = Path(f'{self._wav_path}_background')
        self._label_path = Path(f'{consts.PROCESSED_PATH}/{name}/{consts.LABEL_DIR}')

        # out
        self._plots_path = Path(f'{consts.PLOTS_PATH}/{name}')
        self._results_path = Path(f'{consts.RESULTS_PATH}/{name}')
        self._spectrograms_path = Path(f'{consts.SPECTROGRAMS_PATH}/{name}')

        # direct paths to files
        self.label_bin_path = Path(f'{self._label_path}/{consts.LABELS_BIN_FILE}')
        self.latency_path = Path(f'{consts.RESULTS_PATH}/{name}/latency.csv')
        self.summary_path = Path(f'{consts.RESULTS_PATH}/{name}/summary.csv')

        if files: # If proc dir is empty
            self.files = files
        else: 
            self.files = self._get_files()

    def _get_files(self):
        return sorted([f.stem for f in self._wav_path.glob('*.wav')])

    def get_wavs(self, noise=None, snr=None):
        p = Path("_".join([s for s in [str(self._wav_path), noise, snr] if s]))
        return [p / f'{f}.{consts.PROC_WAV_EXT}' for f in self.files]

    def get_labels(self):
        return [self._label_path / f'{f}.{consts.PROC_LABEL_EXT}' for f in self.files]

    def get_plot(self, plot_name, tracker_name=None):
        name = "_".join([s for s in [plot_name, self.name, tracker_name] if s]) + '.pdf'
        return self._plots_path / name

    def get_result(self, tracker_name, noise_type=None, snr=None):
        name = "_".join([s for s in [tracker_name, noise_type, snr] if s]) + '.pkl'
        return self._results_path / name

    def get_spectrogram(self, track_name):
        return self._spectrograms_path / (track_name + '.pdf')

    def get_background(self):
        return [self._wav_bg_path / (f + '.wav') for f in self.files]

    # TODO Metadata
