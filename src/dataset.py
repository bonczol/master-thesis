import consts
from pathlib import Path


class Input:
    def __init__(self, name, ext, directory='', prefix=''):
        self.name = name
        self._ext = ext
        self._dir = directory
        self._prefix = prefix

    def get_values(self, ds_name):
        return sorted([p for p in Path(f'{consts.RAW_PATH}/{ds_name}').glob(f'**/{self._dir}/**/*.{self._ext}')
                       if not str(p.name).startswith('.') 
                       and str(p.name).startswith(self._prefix)])


class DatasetInput:
    def __init__(self, name, inputs):
        self.name = name
        self.inputs = {i.name: i for i in inputs}

    def get(self, input_name):
        return self.inputs[input_name].get_values(self.name)

    def get_filenames(self, input_name):
        return [path.stem for path in self.inputs[input_name].get_values(self.name)]
        

class MirInput(DatasetInput):
    def __init__(self):
        super().__init__(
            'MIR-1k', 
            [Input('audio', 'wav', 'Wavfile'), 
             Input('pitch', 'pv', 'PitchLabel'),
             Input('vocal', 'vocal', 'vocal-nonvocalLabel')]
        )

class MdbInput(DatasetInput):
    def __init__(self):
        super().__init__(
            'MDB-stem-synth', 
            [Input('audio', 'wav', 'audio_stems'), 
             Input('pitch', 'csv', 'annotation_stems'),
             Input('metadata', 'yaml', 'metadata')]
        )

class UrmpInput(DatasetInput):
    def __init__(self):
        super().__init__(
            'URMP',
            [Input('audio', 'wav', prefix='AuSep'),
             Input('pitch', 'txt', prefix='F0s'),
             Input('notes', 'txt', prefix='Notes')]
        )


class DatasetOutput:
    def __init__(self, name, files=None):
        self.name = name

        # data
        self._wav_path = consts.PROCESSED_PATH / name / consts.WAV_DIR
        self._wav_bg_path = Path(f'{self._wav_path}_background')
        self._label_path = consts.PROCESSED_PATH / name / consts.LABEL_DIR
        self._notes_path = consts.PROCESSED_PATH / name / consts.NOTES_DIR

        # out
        self._plots_path = consts.PLOTS_PATH / name
        self._results_path = consts.RESULTS_PATH / name
        self._results_notes_path = consts.RESULTS_NOTES_PATH / name
        self._spectrograms_path = consts.SPECTROGRAMS_PATH / name

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
        p = Path("_".join([str(s) for s in [self._wav_path, noise, snr] if s is not None]))
        return [p / f'{f}.{consts.PROC_WAV_EXT}' for f in self.files]

    def get_labels(self):
        return [self._label_path / f'{f}.{consts.PROC_LABEL_EXT}' for f in self.files]

    def get_plot(self, plot_name, tracker_name=None):
        name = "_".join([str(s) for s in [plot_name, self.name, tracker_name] if s is not None]) + '.pdf'
        return self._plots_path / name

    def get_result(self, tracker_name, noise_type=None, snr=None):
        name = "_".join([str(s) for s in [tracker_name, noise_type, snr] if s is not None]) + '.pkl'
        return self._results_path / name

    def get_result_notes(self, transcriber_name, noise_type=None, snr=None):
        name = "_".join([str(s) for s in [transcriber_name, noise_type, snr] if s is not None]) + '.pkl'
        return self._results_notes_path / name

    def get_spectrogram(self, track_name):
        return self._spectrograms_path / (track_name + '.pdf')

    def get_background(self):
        return [self._wav_bg_path / (f + '.wav') for f in self.files]

    def get_notes(self):
        return [self._notes_path / f'{f}.{consts.PROC_NOTE_EXT}' for f in self.files]
