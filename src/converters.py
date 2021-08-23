import numpy as np
import shutil
import pickle
import utils
import pandas as pd
import yaml
from pathlib import Path
from multiprocessing import Pool
from utils import semitones2hz
import consts
import sox
import re


class Converter:
    def __init__(self, raw, proc):
        self.raw = raw        
        self.proc = proc

    def _convert_args(self):
        return zip(self.raw.get('audio'), self.raw.get('pitch'), 
                   self.proc.get_wavs(), self.proc.get_labels())

    def _convert_func(self):
        pass

    def _convert(self):
        with Pool() as pool:
            pool.starmap(self._convert_func(), self._convert_args())


    def _save_labels_binary(self, labels):
        with open(self.proc.label_bin_path, 'wb') as f:
            pickle.dump(labels, f)


    def _get_labels_data(self):
        with Pool() as pool:
            labels = pool.map(utils.read_label, self.proc.get_labels())

        labels_df = pd.DataFrame(labels, columns=['file', 'label_time', 'label_pitch'])
        
        labels_df['duration'] = labels_df['label_time'].map(lambda x: x[-1])
        return labels_df

    def prepare(self):
        self._convert()
        labels = self._get_labels_data()
        self._save_labels_binary(labels)


class MirConverter(Converter):
    def __init__(self, raw, proc):
        self.label_t0 = 0.02
        self.label_ts = 0.02
        super().__init__(raw, proc)

    def _convert_args(self):
        return zip(self.raw.get('audio'), self.raw.get('pitch'), self.proc.get_wavs(), 
                   self.proc.get_labels(), self.proc.get_background())

    def _convert_func(self):
        return self._convert_example

    def _convert_example(self, wav_path, label_path, out_wav_path, out_label_path, out_wav_background_path):
        sox.Transformer().remix({1: [2]}) \
                         .build(str(wav_path), str(out_wav_path))

        sox.Transformer().remix({1: [1]}) \
                         .build(str(wav_path), str(out_wav_background_path))

        freq_est = np.loadtxt(label_path, delimiter=',')
        freq_est  = np.where(freq_est <= 0.0, freq_est, semitones2hz(freq_est - 3.5))
        duration = sox.file_info.duration(str(wav_path))
        t = np.arange(self.label_t0, duration - (self.label_ts - 0.00001),  self.label_ts)
        time_series = np.transpose(np.vstack((t, freq_est)))
        np.savetxt(out_label_path, time_series, delimiter=',', fmt='%1.6f')

    def _get_labels_data(self):
        labels_df = super()._get_labels_data()

        with Pool(processes=1) as pool: 
            vocals = pool.map(np.loadtxt, self.raw.get('vocal'))
        
        vocals = pd.Series([v.astype(np.bool) for v in vocals], name='vocal')
        return pd.concat([labels_df, vocals], axis=1)


class MdbConverter(Converter):
    def __init__(self, dataset_raw, datset_proc):
        super().__init__(dataset_raw, datset_proc)

    def _convert_func(self):
        return self._convert_example

    def _convert_example(self, wav_path, label_path, out_wav_path, out_label_path):
        sox.Transformer().convert(consts.SR, 1, 16) \
                         .build(str(wav_path), str(out_wav_path))
        shutil.copyfile(label_path, out_label_path)

    def _get_instrument(self, path):
            instruments = dict()
            with open(path) as f:
                meta = yaml.full_load(f)
                for info in meta['stems'].values():
                    filename = Path(info['filename']).stem + ".RESYN"
                    instruments[filename] = info['instrument']
            return instruments

    def _get_labels_data(self):
        labels_df = super()._get_labels_data()

        with Pool() as pool: 
            instruments_dicts = pool.map(self._get_instrument, self.raw.get('metadata'))

        instruments_dict_merged = {filename: instrument for d in instruments_dicts 
                                   for filename, instrument in d.items()}

        labels_df['instrument'] = labels_df['file'].map(instruments_dict_merged) 
        labels_df['avg_pitch']  = labels_df['label_pitch'].apply(lambda p: np.sum(p) / np.count_nonzero(p))
        return labels_df


class UrmpConverter(Converter):
    def __init__(self, raw, proc):
        super().__init__(raw, proc)

    def _convert_args(self):
        return zip(self.raw.get('audio'), self.raw.get('pitch'), self.raw.get('notes'),
         self.proc.get_wavs(), self.proc.get_labels(), self.proc.get_notes())

    def _convert_func(self):
        return self._convert_example

    def _convert_example(self, wav_path, label_path, note_path, out_wav_path, out_label_path, out_note_path):
        sox.Transformer().convert(consts.SR, 1, 16) \
                         .build(str(wav_path), str(out_wav_path))
    
        label = np.loadtxt(label_path, delimiter='\t')
        np.savetxt(out_label_path, label, delimiter=',', fmt='%1.6f')

        start, pitch, duration =  pd.read_csv(note_path, sep='\t+', engine='python').T.to_numpy()
        intervals = np.transpose(np.vstack((start, start + duration, pitch)))
        np.savetxt(out_note_path, intervals, delimiter=',', fmt='%1.6f')

    def _get_labels_data(self):
        labels_df = super()._get_labels_data()

        with Pool() as pool: 
            notes = pool.map(utils.read_notes, self.proc.get_notes())
        a = 1
        notes_df = pd.DataFrame(notes, columns=['ref_note_interval', 'ref_note_pitch'])
        return pd.concat([labels_df, notes_df], axis=1)


