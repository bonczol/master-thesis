import argparse
from method import Tracker
import consts
import evaluate
import plots
from dataset import MirDataset, MdbDataset
from trackers import PYin, Spice, Crepe, Yin, InverseTracker, Swipe, Hf0, PYin


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
    evaluate_parser.add_argument('-A', '--all', action="store_true")

    # Plot
    plot_parser = subparsers.add_parser('plot', parents=[dataset_parser, tracker_parser])
    plot_parser.add_argument('-A', '--all', action="store_true")

    args = parser.parse_args()

    if args.which in ['prepare', 'evaluate', 'plot']:
        DATASET = { "MIR-1k": MirDataset, "MDB-stem-synth": MdbDataset}
        datasets = [DATASET[d]() for d in args.datasets]


    if args.which in ['evaluate', 'plot']:
        TRACKER = {
            Tracker.SPICE: Spice,
            Tracker.CREPE_TINY: Crepe,
            Tracker.DDSP_INV: InverseTracker,
            Tracker.YIN: Yin,
            Tracker.SWIPE: Swipe,
            Tracker.HF0: Hf0,
            Tracker.PYIN: PYin
         }
        trackers = [Tracker(t) for t in args.trackers]


    if args.which == 'prepare':
        for dataset in datasets:
            dataset.prepare()

    if args.which == 'evaluate':
        concrete_trackers = [TRACKER[t]() for t in trackers]
        for dataset in datasets:
            for tracker in concrete_trackers:
                evaluate.run_evaluation(tracker, dataset, args.noise, args.snr)

    if args.which == 'plot':
        for dataset in datasets:
            plots.plot(dataset, trackers)

            




        
