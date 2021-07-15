import utils
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from librosa import display as librosadisplay
from scipy.io import wavfile
from scipy import signal

MAX_ABS_INT16 = 32768.0


def plot_stft(x, sample_rate, output_path, show_black_and_white=False):
  x_stft = np.abs(librosa.stft(x, hop_length=256, n_fft=2048))
  fig, ax = plt.subplots()
  fig.set_size_inches(12, 5)
  x_stft_db = librosa.amplitude_to_db(x_stft, ref=np.max)
  if(show_black_and_white):
    librosadisplay.specshow(data=x_stft_db, y_axis='log', 
                             sr=sample_rate, cmap='gray_r')
  else:
    librosadisplay.specshow(data=x_stft_db, y_axis='log', sr=sample_rate)

  plt.colorbar(format='%+2.0f dB')
  plt.savefig(output_path)
  plt.close(fig)


def main():
    parser, conf = utils.get_parser_and_config()
    args = parser.parse_args()
    conf = conf[args.ds_name]

    wav_paths = utils.get_wav_paths(conf['processed_wav_dir'])

    for wav_path in wav_paths:
        sr, waveform = wavfile.read(wav_path, 'rb')
        fname = os.path.splitext(os.path.basename(wav_path))[0]
        output_path = f'{conf["specgram_dir"]}/{fname}.jpg'
        fig = plot_stft(waveform / MAX_ABS_INT16 , sr, output_path, show_black_and_white=True)

if __name__ == "__main__":
    main()