import subprocess
import json
import consts
from itertools import repeat
from multiprocessing import Pool


def degrade(input_path, output_path, degradation_path):
    subprocess.call([
        'audio-degradation-toolbox',
        '-d', degradation_path,
        input_path, 
        output_path
    ])


def check_directory(path):
    parent_dir = path.parent
    if not parent_dir.exists():
        parent_dir.mkdir()


class Modifier:
    def __init__(self, proc):
        self.proc = proc

    def add_noise(self, color, snr):
        output_paths = self.proc.get_wavs(color, snr)
        check_directory(output_paths[0])

        degradation_path = consts.DEGRADE_TMP_PATH / f'noise_{color}_{snr}.json'
        with open(degradation_path, 'w') as f:
            json.dump([{"name": "noise", "color": color, "snr": snr}], f)

        with Pool() as pool:
            pool.starmap(degrade, zip(self.proc.get_wavs(), output_paths, repeat(degradation_path)))


    def add_accompaniment(self, snr):
        output_paths = self.proc.get_wavs('acco', snr)
        check_directory(output_paths[0])

        degradation_paths = []
        for bg_path in self.proc.get_background():
            degradation_path = consts.DEGRADE_TMP_PATH / f'acco_{bg_path.stem}_{snr}.json'
            degradation_paths.append(degradation_path)
            with open(degradation_path, 'w') as f:
                json.dump([{"name": "mix", "path": str(bg_path), "snr": snr}], f)

        with Pool() as pool:
            pool.starmap(degrade, zip(self.proc.get_wavs(), output_paths, degradation_paths))

    @staticmethod
    def clear_tmp_files():
        for path in consts.DEGRADE_TMP_PATH.glob('*.json'):
            if path.is_file():
                path.unlink()
