import consts
import utils
import pandas as pd
import pickle
from pathlib import Path
from multiprocessing import Pool
from converters import MirConverter, MDBConverter


class Dataset:
    def __init__(self, name, raw_wav_dir, raw_label_dir, label_ext, download_url, file_name):
        # data
        self.name = name
        self.raw_wav_path = Path(f'{consts.RAW_PATH}/{name}/{raw_wav_dir}')
        self.raw_label_path = Path(f'{consts.RAW_PATH}/{name}/{raw_label_dir}')
        self.proc_wav_path = Path(f'{consts.PROCESSED_PATH}/{name}/{consts.WAV_DIR}')
        self.proc_wav_bg_path = Path(f'{self.proc_wav_path}_background')
        self.proc_label_path = Path(f'{consts.PROCESSED_PATH}/{name}/{consts.LABEL_DIR}')
        self.proc_label_bin_path = Path(f'{self.proc_label_path}/{consts.LABELS_BIN_FILE}')
        self.metadata_path = Path(f'{consts.RAW_PATH}/{name}/metadata')
        
        # out
        self.plots_path = Path(f'{consts.PLOTS_PATH}/{name}')
        self.results_path = Path(f'{consts.RESULTS_PATH}/{name}')
        self.spectrograms_path = Path(f'{consts.SPECTROGRAMS_PATH}/{name}')
        self.latency_path = Path(f'{consts.RESULTS_PATH}/{name}/latency.csv')
        self.summary_path = Path(f'{consts.RESULTS_PATH}/{name}/summary.csv')

        # others
        self.label_ext = label_ext
        self.download_url = download_url
        self.file_name = file_name

        # Files
        self.raw_wav_paths = sorted([p for p in self.raw_wav_path.glob('*.wav') if not str(p.name).startswith('.')])
        self.raw_label_paths = sorted(self.raw_label_path.glob(f'*.{self.label_ext}'))
        self.files = [f.stem for f in self.raw_wav_paths]
        self.proc_wav_bg_paths = [self.proc_wav_bg_path / (f + '.wav') for f in self.files]
        self.proc_label_paths = [self.proc_label_path / (f + '.csv') for f in self.files]
        self.metadata_paths = sorted(self.metadata_path.glob('*'))

    def get_proc_wav_paths(self, noise=None, snr=None):
        p = Path("_".join([s for s in [str(self.proc_wav_path), noise, snr] if s]))
        return [p / (f + '.wav') for f in self.files]

    def get_plot_path(self, plot_name, tracker_name=None):
        name = "_".join([s for s in [plot_name, self.name, tracker_name] if s]) + '.pdf'
        return self.plots_path / name

    def get_result_path(self, tracker_name, noise_type=None, snr=None):
        name = "_".join([s for s in [tracker_name, noise_type, snr] if s]) + '.pkl'
        return self.results_path / name

    def get_spectrogram_path(self, track_name):
        return self.spectrograms_path / (track_name + '.pdf')

    def init_converters(self):
        pass
    
    def _convert_example(self, converter):
        converter.convert()

    def prepare(self):
        converters = self.init_converters()
        with Pool() as pool:
            pool.map(self._convert_example, converters)
        self.save_labels_binary()
        
    def save_labels_binary(self):
        with Pool() as pool:
            labels = pool.map(utils.read_label, self.proc_label_paths)
            
        labels_df = pd.DataFrame(labels, columns=['file', 'label_time', 'label_pitch'])
        labels_df['duration'] = labels_df['label_time'].map(lambda x: x[-1])
        with open(self.proc_label_bin_path, 'wb') as f:
            pickle.dump(labels_df, f)


class MirDataset(Dataset):
    def __init__(self):
        super().__init__(
            'MIR-1k',
            raw_wav_dir="Wavfile",
            raw_label_dir="PitchLabel",
            label_ext="pv", 
            download_url="https://ndownloader.figshare.com/files/10256751",
            file_name = "MIR-1k.rar"
            )

    def init_converters(self):
        return [MirConverter(w, l, ow, ol, owb) 
                for w, l, ow, ol, owb in zip(
                    self.raw_wav_paths, self.raw_label_paths, self.get_proc_wav_paths(), 
                    self.proc_label_paths, self.proc_wav_bg_paths
                )]
        

class MdbDataset(Dataset):
    def __init__(self):
        super().__init__(
            'MDB-stem-synth', 
            label_ext="csv",
            raw_wav_dir="audio_stems",
            raw_label_dir="annotation_stems",
            download_url="https://zenodo.org/record/1481172/files/MDB-stem-synth.tar.gz",
            file_name = "MDB-stem-synth.tar.gz"
            )

    def init_converters(self):
        return [MDBConverter(w, l, ow, ol) 
                for w, l, ow, ol in zip(
                    self.raw_wav_paths, self.raw_label_paths, 
                    self.get_proc_wav_paths(), self.proc_label_paths
                )]