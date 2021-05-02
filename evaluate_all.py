import evaluate
import utils
from detectors import Detector


def main():
    _, conf = utils.get_parser_and_config()

    dataset = 'MIR-1k'
    noise_types = ['acco']
    snrs = [20, 10, 0]

    for detector in Detector:
        for noise_type in noise_types:
            for snr in snrs:
                in_dir = f'{conf["output_dir_wav"]}_{noise_type}_{snr}'
                evaluate.evaluate(detector, in_dir)
    



if __name__ == "__main__":
    main()