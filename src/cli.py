import argparse
from method import Tracker
import evaluate
import plots
from dataset import DatasetOutput, MirInput, MdbInput, UrmpInput, PtdbInput
from converters import MirConverter, MdbConverter, UrmpConverter, PtdbConverter


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


    if args.which in ['evaluate', 'plot']:
        datasets_outputs = [DatasetOutput(d) for d in args.datasets]


    if args.which in ['evaluate', 'plot']:
        from trackers import PYin, Spice, Crepe, Yin, InverseTracker, Swipe, Hf0, PYin
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
            plots.plot(dataset, trackers)

            




        
