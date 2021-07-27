import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import consts
import evaluate
import plots
import itertools
import ploting
from method import Tracker
from dataset import DatasetOutput, MirInput, MdbInput, UrmpInput, PtdbInput
from converters import MirConverter, MdbConverter, UrmpConverter, PtdbConverter
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
    evaluate_parser.add_argument('--noise', type=str)
    evaluate_parser.add_argument('--snr', type=int)

    # Plot
    plot_parser = subparsers.add_parser('plot', parents=[dataset_parser, tracker_parser])

     # Subplot
    subplot_parser = subparsers.add_parser('subplot', parents=[dataset_parser, tracker_parser])

    # Degrade
    degrade_parser = subparsers.add_parser('degrade', parents=[dataset_parser])
    degrade_parser.add_argument('-t', '--type', nargs='?')

    args = parser.parse_args()


    if args.which in ['evaluate', 'plot', 'subplot', 'degrade']:
        datasets_outputs = [DatasetOutput(d) for d in args.datasets]


    if args.which in ['evaluate', 'plot', 'subplot']:
        from trackers import *
        TRACKER = {
            Tracker.SPICE: Spice,
            Tracker.CREPE: Crepe,
            Tracker.DDSP_INV: InverseTracker,
            Tracker.YIN: Yin,
            Tracker.SWIPE: Swipe,
            Tracker.HF0: Hf0,
            Tracker.PYIN: PYin
         }

        trackers = [Tracker(t) for t in args.trackers]


    if args.which == 'prepare':
        DATASET_INPUT = {
            'MIR-1k': MirInput, 'MDB-stem-synth': MdbInput, 
            'URMP': UrmpInput,'PTDB-TUG': PtdbInput
        }
        CONVERTERS = {
            'MIR-1k': MirConverter,'MDB-stem-synth': MdbConverter, 
            'URMP': UrmpConverter,'PTDB-TUG': PtdbConverter
        }
        
        datasets_inputs = [DATASET_INPUT[d]() for d in args.datasets]
        datasets_outputs = [DatasetOutput(d.name, d.get_files()) for d in datasets_inputs]
        converters = [CONVERTERS[in_.name](in_, out_) for in_, out_ in zip(datasets_inputs, datasets_outputs)]
        for converter in converters:
            converter.prepare()


    if args.which == 'evaluate':
        concrete_trackers = [TRACKER[t]() for t in trackers]
        for dataset in datasets_outputs:
            for tracker in concrete_trackers:
                evaluate.run_evaluation(tracker, dataset, args.noise, args.snr)


    if args.which == 'plot':
        for dataset in datasets_outputs:
            plots.plot(dataset,  trackers)


    if args.which == 'subplot':
        ploting.subplot(datasets_outputs, trackers)


    if args.which == 'degrade':
        snrs = [20, 10, 0]
        colors = ['white', 'pink', 'brown']
        modifiers = [Modifier(d) for d in datasets_outputs]

        if args.type == 'noise':
            for snr, color, modifier in itertools.product(snrs, colors, modifiers):
                modifier.add_noise(color, snr)
        elif args.type == 'acco':
            for snr, modifier in itertools.product(snrs, modifiers):
                modifier.add_accompaniment(snr)
        else:
            raise NotImplemented

        Modifier.clear_tmp_files()





        
