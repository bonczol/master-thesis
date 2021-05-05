import evaluate
import utils


def main():
    NOISE = False

    # dataset = 'MDB-stem-synth'
    dataset = 'MIR-1k'
    noise_types = ['acco']
    snrs = [20, 10, 0]
    detectors = ["SPICE", "CREPETINY", 'YIN']

    _, conf = utils.get_parser_and_config()
    conf = conf[dataset]

    # Clean
    for detector in detectors:
        print(f'{dataset} - CLEAN - {detector}')
        evaluate.evaluate(detector, conf["processed_wav_dir"], conf['results_dir'])

    # With noise
    if NOISE:
        for detector in detectors:
            for noise_type in noise_types:
                for snr in snrs:
                    print(f'{dataset} - {noise_type} - {snr} - {detector}')
                    in_dir = f'{conf["processed_wav_dir"]}_{noise_type}_{snr}'
                    evaluate.evaluate(detector, in_dir, conf['results_dir'])
    



if __name__ == "__main__":
    main()