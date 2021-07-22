import subprocess
import json
import consts


def degrade(input_path, output_path, degradation_conf):
    degradation_path = consts.OUT_DIR / 'tmp_degrade_conf.json'
    
    with open(degradation_path, 'w') as f:
        json.dump(degradation_conf, f)

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

        for in_path, out_path in zip(self.proc.get_wavs(), output_paths):
            degradation = {
                "name": "noise",
                "color": color,
                "snr": snr
            }
            degrade(in_path, out_path, [degradation])

    def add_accompaniment(self, snr):
        output_paths = self.proc.get_wavs('acco', snr)
        check_directory(output_paths[0])

        for in_path, out_path, bg_path in zip(self.proc.get_wavs(), output_paths, self.proc.get_background()):
            degradation = {
                "name": "mix",
                "path": str(bg_path),
                "snr": snr
            }
            degrade(in_path, out_path, [degradation])
