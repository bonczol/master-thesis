import utils
import os
import audio_degrader as ad
import sox

class MyAudioDegrader(ad.AudioFile):
    def to_wav(self, output_path):
        tfm = sox.Transformer()
        tfm.convert(n_channels=1, bitdepth=16)
        tfm.build(self.tmp_path, output_path)


def add_acco(input_wav_dir, input_wav_background_dir):
    vocal_paths = utils.get_wav_paths(input_wav_dir)
    background_paths = utils.get_wav_paths(input_wav_background_dir)
    background_paths = [os.path.abspath(p) for p in background_paths]    
    background_paths = [p.replace('\\', '/') for p in background_paths]    
    snrs = [20, 10 , 0]

    for snr in snrs:
        output_dir = f'{input_wav_dir}_acco_{snr}'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for vocal_path, background_path in zip(vocal_paths, background_paths):
            audio_file = MyAudioDegrader(vocal_path, './tmp_dir')
            degradations = ad.ParametersParser.parse_degradations_args([f'mix,{background_path},{snr}'])
            
            for d in degradations:
                audio_file.apply_degradation(d)

            output_path = os.path.join(output_dir, os.path.basename(vocal_path))
            audio_file.to_wav(output_path)
            audio_file.delete_tmp_files()


def main():
    parser, conf = utils.get_parser_and_config()
    args = parser.parse_args()
    conf = conf[args.ds_name]
    print(args.ds_name)

    if args.ds_name == "MIR-1k":
        add_acco(conf['output_dir_wav'], conf['output_dir_wav_background'])



if __name__ == "__main__":
    main()
