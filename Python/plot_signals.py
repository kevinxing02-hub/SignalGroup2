#!/usr/bin/env python3
"""
Script to visualize EDF signals and hypnogram from XML annotations.

Usage:
    python plot_signals.py --edf <path_to_edf> --xml <path_to_xml> [--epoch <epoch_number>]

Example:
    python plot_signals.py --edf data/training/file.edf --xml data/training/file.xml --epoch 100
"""

""" ADDING CODE FOR PLOTTING SIGNALS AFTER PREPROCESSING!! """

import argparse
import matplotlib.pyplot as plt
from src.visualization import plot_sample_epoch, plot_hypnogram, plot_preprocessed_epoch
from src.data_loader import load_single_recording
from src.preprocessing import preprocess
import config


def main():
    parser = argparse.ArgumentParser(description='Visualize sleep signals and hypnogram')
    parser.add_argument('--edf', type=str, help='Path to EDF file')
    parser.add_argument('--xml', type=str, help='Path to XML annotation file')
    parser.add_argument('--epoch', type=int, default=0, help='Epoch number to plot (default: 0)')
    parser.add_argument('--epoch-duration', type=int, default=30, help='Epoch duration in seconds (default: 30)')
    parser.add_argument('--preprocessed', action='store_true',
                        help='Plot preprocessed signals instead of raw')

    args = parser.parse_args()

    # Plot preprocessed signals if flag is set
    if args.preprocessed:
        if args.edf and args.xml:
            print(f"Loading and preprocessing data from: {args.edf}")

            # Load data
            epochs, labels = load_single_recording(args.edf, args.xml, args.epoch_duration)
            multi_channel_data = {'eeg': epochs}

            # Get channel info
            samples_per_epoch = epochs.shape[2]
            fs = samples_per_epoch / args.epoch_duration
            channel_info = {
                'eeg_fs': fs,
                'epoch_length': args.epoch_duration,
                'eeg_names': ['EEG_1', 'EEG_2']
            }

            # Preprocess
            preprocessed_data = preprocess(multi_channel_data, config, channel_info=channel_info)

            # Plot preprocessed epoch
            print(f"\nPlotting preprocessed epoch {args.epoch}...")
            plot_preprocessed_epoch(preprocessed_data, epoch_idx=args.epoch,
                                    epoch_duration=args.epoch_duration, channel_info=channel_info)
        else:
            print("Error: --edf and --xml are required for preprocessed plotting")
            parser.print_help()
        return

    # Plot raw signals from EDF file
    if args.edf:
        print(f"Plotting signals from EDF file: {args.edf}")
        plot_sample_epoch(args.edf, epoch_idx=args.epoch, epoch_duration=args.epoch_duration)
    else:
        print("No EDF file specified. Use --edf to specify a file.")

    # Plot hypnogram from XML file
    if args.xml:
        print(f"\nPlotting hypnogram from XML file: {args.xml}")
        plot_hypnogram(args.xml, edf_path=args.edf)
    else:
        print("No XML file specified. Use --xml to specify a file.")

    # Display all plots
    if args.edf or args.xml:
        plt.show()

    # If neither specified, show usage
    if not args.edf and not args.xml:
        parser.print_help()
        print("\n" + "=" * 70)
        print("Example usage:")
        print("  # Plot raw signals:")
        print("  python plot_signals.py --edf data/training/R1.edf --xml data/training/R1.xml")
        print("\n  # Plot preprocessed signals:")
        print(
            "  python plot_signals.py --edf data/training/R1.edf --xml data/training/R1.xml --preprocessed --epoch 100")
        print("=" * 70)


if __name__ == "__main__":
    main()