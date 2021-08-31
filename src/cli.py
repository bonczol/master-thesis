import argparse
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import evaluate
import spectrograms
import post
from itertools import product
import ploting
from method import Method
from dataset import DatasetOutput, MirInput, MdbInput, UrmpInput, IapasInput
from transcribers import CrepeHmmTrans, PyinTrans
from converters import MirConverter, MdbConverter, UrmpConverter, IapasConverter
from degrader import Modifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='commands', dest='which')

    # Common args
    dataset_parser = argparse.ArgumentParser(add_help=False)
    dataset_parser.add_argument('-d', '--datasets', nargs='*')

    tracker_parser = argparse.ArgumentParser(add_help=False)
    tracker_parser.add_argument('-t', '--trackers', nargs='*')

    # Prepare
    prepare_parser = subparsers.add_parser('prepare', parents=[dataset_parser])

    # Evaluate
    evaluate_parser = subparsers.add_parser('evaluate', parents=[dataset_parser, tracker_parser])
    evaluate_parser.add_argument('--noise', action='store_true')
    evaluate_parser.add_argument('--notes', action='store_true')

    # Post
    post_parser = subparsers.add_parser('post')
    post_parser.add_argument('--notes', action='store_true')

     # Subplot
    subplot_parser = subparsers.add_parser('subplot')

    # Spectrograms
    spec_parser = subparsers.add_parser('spec', parents=[tracker_parser])

    # Degrade
    degrade_parser = subparsers.add_parser('degrade', parents=[dataset_parser])
    degrade_parser.add_argument('-t', '--type', nargs='?')

    args = parser.parse_args()


    snrs = [20, 10, 0]
    colors = ['white', 'pink', 'brown', 'blue', 'violet', 'acco']
    all_datasets = ['MIR-1k', 'MDB-stem-synth', 'URMP']
    all_trackers = list(Method)


    if args.which in ['evaluate', 'degrade']:
        datasets_outputs = [DatasetOutput(d) for d in args.datasets]


    if args.which in ['evaluate', 'spec']:
        from trackers import *
        TRACKER = {
            Method.SPICE: Spice,
            Method.CREPE: Crepe,
            Method.DDSP_INV: InverseTracker,
            Method.SWIPE: Swipe,
            # Method.HF0: Hf0,
            Method.PYIN: OrignalPYin,
            Method.CREPE_MIDI: CrepeHmmTrans,
            Method.PYIN_MIDI: PyinTrans
         }

        trackers = [Method(t) for t in args.trackers]


    if args.which == 'prepare':
        DATASET_INPUT = {
            'MIR-1k': MirInput, 'MDB-stem-synth': MdbInput, 
            'URMP': UrmpInput, 'IAPAS': IapasInput
        }
        CONVERTERS = {
            'MIR-1k': MirConverter,'MDB-stem-synth': MdbConverter, 
            'URMP': UrmpConverter, 'IAPAS': IapasConverter
        }
        
        datasets_inputs = [DATASET_INPUT[d]() for d in args.datasets]
        datasets_outputs = [DatasetOutput(d.name, d.get_filenames('audio')) for d in datasets_inputs]
        converters = [CONVERTERS[in_.name](in_, out_) for in_, out_ in zip(datasets_inputs, datasets_outputs)]

        for converter in converters:
            converter.prepare()


    if args.which == 'evaluate':
        concrete_trackers = [TRACKER[t]() for t in trackers]

        if args.noise:
            for dataset, tracker, color, snr in product(datasets_outputs, concrete_trackers, colors, snrs):
                evaluate.run_evaluation(tracker, dataset, color, snr, args.notes)
        else:
            for dataset, tracker in product(datasets_outputs, concrete_trackers):
                    evaluate.run_evaluation(tracker, dataset, None, None, args.notes)


    if args.which == 'post':
        post.transform(
            [DatasetOutput(n) for n in ['MIR-1k', 'MDB-stem-synth', 'URMP']],
            list(Method),
            colors,
            snrs,
            args.notes
        )
 

    if args.which == 'subplot':
        ploting.subplot()


    if args.which == 'spec':
        concrete_trackers = [TRACKER[t]() for t in trackers]
        spectrograms.generate(concrete_trackers)
        

    if args.which == 'degrade':
        modifiers = [Modifier(d) for d in datasets_outputs]

        if args.type == 'noise':
            for snr, color, modifier in product(snrs, colors, modifiers):
                modifier.add_noise(color, snr)
        elif args.type == 'acco':
            for snr, modifier in product(snrs, modifiers):
                modifier.add_accompaniment(snr)
        else:
            raise NotImplemented 

        Modifier.clear_tmp_files()





        
