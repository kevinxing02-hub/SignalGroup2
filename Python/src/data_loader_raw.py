"""
Data Loader Module

This module provides complete implementations for loading EDF and XML files
for sleep stage classification.
"""

import numpy as np
import mne
import os
import sys
import csv
from pathlib import Path

# Handle both package import and standalone execution
try:
    from .xml_parser import parse_xml_annotations, create_epoch_labels
except ImportError:
    from xml_parser import parse_xml_annotations, create_epoch_labels

def load_single_recording(edf_file_path, xml_file_path, epoch_length=30):
    """
    Load one EDF file with its corresponding XML annotations.

    This function loads a single recording, extracts EEG channels, segments into
    30-second epochs, and returns epochs with their corresponding sleep stage labels.

    According to the project guide, this function should:
    1. Load EDF file using MNE-Python
    2. Parse XML annotations
    3. Extract EEG channels only (for iteration 1)
    4. Segment into 30-second epochs
    5. Return epochs and labels

    Args:
        edf_file_path (str): Path to the EDF file
        xml_file_path (str): Path to the XML annotation file
        epoch_length (float): Epoch duration in seconds (default 30)

    Returns:
        tuple: (epochs, labels) where:
            - epochs (np.ndarray): Shape (n_epochs, n_channels, samples_per_epoch)
            - labels (np.ndarray): Shape (n_epochs,), integer labels 0-4
                (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)

    Raises:
        FileNotFoundError: If EDF or XML file doesn't exist
        ValueError: If no EEG channels are found

    Example:
        >>> epochs, labels = load_single_recording('data/training/R1.edf', 'data/training/R1.xml')
        >>> print(f"Loaded {epochs.shape[0]} epochs with shape {epochs.shape}")
    """
    # Validate files exist
    if not os.path.exists(edf_file_path):
        raise FileNotFoundError(f"EDF file not found: {edf_file_path}")
    if not os.path.exists(xml_file_path):
        raise FileNotFoundError(f"XML file not found: {xml_file_path}")

    # Load the EDF file using MNE-Python
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
    recording_duration = raw.times[-1]  # Duration in seconds

    # Parse the XML annotations
    parsed_xml = parse_xml_annotations(xml_file_path)
    stages = parsed_xml['stages']

    # Create epoch labels from XML annotations
    # Use np.ceil to match create_epoch_labels() calculation
    # This ensures we handle partial epochs correctly
    n_epochs = int(np.ceil(recording_duration / epoch_length))
    labels = create_epoch_labels(stages, recording_duration, epoch_length)

    # Identify EEG channels (for iteration 1, we only use EEG)
    channel_names = raw.ch_names
    eeg_channels = []

    for ch in channel_names:
        ch_upper = ch.upper()
        # Match EEG channels, but exclude SaO2, SpO2, etc.
        if 'EEG' in ch_upper and 'SAO' not in ch_upper and 'SPO' not in ch_upper:
            eeg_channels.append(ch)
        # Also match common EEG electrode patterns
        elif any(pattern in ch_upper for pattern in ['C3', 'C4', 'F3', 'F4', 'O1', 'O2']):
            # Make sure it's not already matched and not an EOG/EMG channel
            if ch not in eeg_channels:
                eeg_channels.append(ch)

    if not eeg_channels:
        raise ValueError(f"No EEG channels found in {edf_file_path}. Available channels: {channel_names}")

    # For iteration 1, use first 2 EEG channels (typically C3-A2 and C4-A1)
    # Select up to 2 channels to match project requirements
    selected_channels = eeg_channels[:2]
    eeg_raw = raw.copy().pick_channels(selected_channels)
    fs = eeg_raw.info['sfreq']

    print(f"  Selected EEG channels: {selected_channels} (sampling rate: {fs} Hz)")

    # Extract data: shape (n_channels, n_samples)
    data = eeg_raw.get_data()

    # Calculate samples per epoch
    samples_per_epoch = int(epoch_length * fs)
    total_samples_needed = n_epochs * samples_per_epoch

    # Trim or pad data to match exact number of epochs
    if data.shape[1] > total_samples_needed:
        # Trim excess samples
        data = data[:, :total_samples_needed]
    elif data.shape[1] < total_samples_needed:
        # Pad with zeros if recording is shorter than expected
        padding = total_samples_needed - data.shape[1]
        data = np.pad(data, ((0, 0), (0, padding)), mode='constant', constant_values=0)

    # Reshape into epochs: (n_channels, n_epochs, samples_per_epoch)
    # Then transpose to (n_epochs, n_channels, samples_per_epoch)
    n_channels = data.shape[0]
    epochs = data.reshape(n_channels, n_epochs, samples_per_epoch)
    epochs = np.transpose(epochs, (1, 0, 2))  # Final shape: (n_epochs, n_channels, samples_per_epoch)

    # Trim labels to match number of epochs (handle any rounding differences)
    if len(labels) > n_epochs:
        labels = labels[:n_epochs]
    elif len(labels) < n_epochs:
        # If labels are shorter, pad with the last label (rare case)
        padding = n_epochs - len(labels)
        labels = np.concatenate([labels, np.full(padding, labels[-1])])

    # Validate that epochs and labels match
    assert epochs.shape[0] == len(labels), \
        f"Mismatch: {epochs.shape[0]} epochs but {len(labels)} labels"

    return epochs, labels



def load_training_data(edf_file_path, xml_file_path, epoch_length=30):
    """
    Load EDF and XML files for training.

    This function loads a complete EDF recording with XML annotations,
    extracts relevant channels (EEG, EOG, EMG), segments into epochs,
    and returns data with corresponding labels.

    Args:
        edf_file_path (str): Path to the EDF file
        xml_file_path (str): Path to the XML annotation file
        epoch_length (float): Epoch duration in seconds (default 30)

    Returns:
        tuple: (multi_channel_data, labels, channel_info) where:
            - multi_channel_data (dict): Containing 'eeg', 'eog', 'emg' arrays
                - 'eeg': np.ndarray, shape (n_epochs, n_eeg_channels, samples_per_epoch)
                - 'eog': np.ndarray, shape (n_epochs, n_eog_channels, samples_per_epoch)
                - 'emg': np.ndarray, shape (n_epochs, n_emg_channels, samples_per_epoch)
            - labels (np.ndarray): Shape (n_epochs,), integer labels 0-4
            - channel_info (dict): Metadata about channels and sampling rates

    Example:
        >>> data, labels, info = load_training_data('R1.edf', 'R1.xml')
        >>> print(f"Loaded {labels.shape[0]} epochs")
        >>> print(f"EEG shape: {data['eeg'].shape}")
    """
    print(f"Loading training data from {edf_file_path} and {xml_file_path}...")

    # Validate files exist
    if not os.path.exists(edf_file_path):
        raise FileNotFoundError(f"EDF file not found: {edf_file_path}")
    if not os.path.exists(xml_file_path):
        raise FileNotFoundError(f"XML file not found: {xml_file_path}")

    # Load EDF file
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)

    # Get recording duration
    recording_duration = raw.times[-1]  # Duration in seconds

    # Parse XML annotations
    parsed_xml = parse_xml_annotations(xml_file_path)
    stages = parsed_xml['stages']

    # Create epoch labels
    # Use np.ceil to match create_epoch_labels() calculation and handle partial epochs
    n_epochs = int(np.ceil(recording_duration / epoch_length))
    labels = create_epoch_labels(stages, recording_duration, epoch_length)

    # Identify channels by name patterns
    channel_names = raw.ch_names

    # EOG channels (check first to avoid conflicts)
    eog_channels = [ch for ch in channel_names if 'EOG' in ch.upper()]

    # EMG channels (check before EEG to avoid conflicts)
    emg_channels = [ch for ch in channel_names if 'EMG' in ch.upper() or 'CHIN' in ch.upper()]

    # EEG channels - match specific patterns, exclude already identified channels
    eeg_candidates = []
    for ch in channel_names:
        ch_upper = ch.upper()
        # Match EEG with specific patterns, but exclude SaO2, SpO2, etc.
        if 'EEG' in ch_upper and 'SAO' not in ch_upper and 'SPO' not in ch_upper:
            eeg_candidates.append(ch)
        # Match C3, C4 (central), F3, F4 (frontal), O1, O2 (occipital), etc.
        elif any(pattern in ch_upper for pattern in ['C3', 'C4', 'F3', 'F4', 'O1-', 'O2-', 'CZ', 'FZ', 'PZ']):
            eeg_candidates.append(ch)

    # Remove duplicates and exclude EOG/EMG channels
    eeg_channels = [ch for ch in eeg_candidates
                    if ch not in eog_channels and ch not in emg_channels]

    print(f"Identified channels:")
    print(f"  EEG: {eeg_channels}")
    print(f"  EOG: {eog_channels}")
    print(f"  EMG: {emg_channels}")

    # Extract data for each signal type
    multi_channel_data = {}
    channel_info = {'epoch_length': epoch_length}

    # Extract EEG data
    if eeg_channels:
        eeg_raw = raw.copy().pick_channels(eeg_channels)
        eeg_data, eeg_fs = _extract_epochs(eeg_raw, epoch_length, n_epochs)
        multi_channel_data['eeg'] = eeg_data
        channel_info['eeg_names'] = eeg_channels
        channel_info['eeg_fs'] = eeg_fs
        print(f"  EEG: {eeg_data.shape[1]} channels, {eeg_data.shape[2]} samples/epoch, {eeg_fs} Hz")

    # Extract EOG data
    if eog_channels:
        eog_raw = raw.copy().pick_channels(eog_channels)
        eog_data, eog_fs = _extract_epochs(eog_raw, epoch_length, n_epochs)
        multi_channel_data['eog'] = eog_data
        channel_info['eog_names'] = eog_channels
        channel_info['eog_fs'] = eog_fs
        print(f"  EOG: {eog_data.shape[1]} channels, {eog_data.shape[2]} samples/epoch, {eog_fs} Hz")

    # Extract EMG data
    if emg_channels:
        emg_raw = raw.copy().pick_channels(emg_channels)
        emg_data, emg_fs = _extract_epochs(emg_raw, epoch_length, n_epochs)
        multi_channel_data['emg'] = emg_data
        channel_info['emg_names'] = emg_channels
        channel_info['emg_fs'] = emg_fs
        print(f"  EMG: {emg_data.shape[1]} channels, {emg_data.shape[2]} samples/epoch, {emg_fs} Hz")

    # Print label distribution
    print(f"\nLoaded {n_epochs} epochs ({n_epochs*epoch_length/3600:.2f} hours)")
    _print_label_distribution(labels)

    # Trim labels to match data (in case of rounding issues)
    labels = labels[:n_epochs]

    return multi_channel_data, labels, channel_info


def load_holdout_data(edf_file_path, epoch_length=30):
    """
    Load holdout EDF file (no labels) for inference.

    Args:
        edf_file_path (str): Path to the EDF file
        epoch_length (float): Epoch duration in seconds (default 30)

    Returns:
        tuple: (multi_channel_data, record_info) where:
            - multi_channel_data (dict): Same structure as load_training_data
            - record_info (dict): Metadata including record_id, n_epochs, channels

    Example:
        >>> data, info = load_holdout_data('H1.edf')
        >>> print(f"Record ID: {info['record_id']}")
        >>> print(f"Epochs: {info['n_epochs']}")
    """
    print(f"Loading holdout data from {edf_file_path}...")

    # Validate file exists
    if not os.path.exists(edf_file_path):
        raise FileNotFoundError(f"EDF file not found: {edf_file_path}")

    # Extract record ID from filename
    record_id = Path(edf_file_path).stem

    # Load EDF file
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)

    # Get recording duration
    recording_duration = raw.times[-1]
    n_epochs = int(recording_duration / epoch_length)

    # Identify channels (same as training)
    channel_names = raw.ch_names

    # EOG channels (check first to avoid conflicts)
    eog_channels = [ch for ch in channel_names if 'EOG' in ch.upper()]

    # EMG channels (check before EEG to avoid conflicts)
    emg_channels = [ch for ch in channel_names if 'EMG' in ch.upper() or 'CHIN' in ch.upper()]

    # EEG channels - match specific patterns, exclude already identified channels
    eeg_candidates = []
    for ch in channel_names:
        ch_upper = ch.upper()
        # Match EEG with specific patterns, but exclude SaO2, SpO2, etc.
        if 'EEG' in ch_upper and 'SAO' not in ch_upper and 'SPO' not in ch_upper:
            eeg_candidates.append(ch)
        # Match C3, C4 (central), F3, F4 (frontal), O1, O2 (occipital), etc.
        elif any(pattern in ch_upper for pattern in ['C3', 'C4', 'F3', 'F4', 'O1-', 'O2-', 'CZ', 'FZ', 'PZ']):
            eeg_candidates.append(ch)

    # Remove duplicates and exclude EOG/EMG channels
    eeg_channels = [ch for ch in eeg_candidates
                    if ch not in eog_channels and ch not in emg_channels]

    print(f"Identified channels:")
    print(f"  EEG: {eeg_channels}")
    print(f"  EOG: {eog_channels}")
    print(f"  EMG: {emg_channels}")

    # Extract data for each signal type
    multi_channel_data = {}
    sampling_rates = {}

    if eeg_channels:
        eeg_raw = raw.copy().pick_channels(eeg_channels)
        eeg_data, eeg_fs = _extract_epochs(eeg_raw, epoch_length, n_epochs)
        multi_channel_data['eeg'] = eeg_data
        sampling_rates['eeg'] = eeg_fs
        print(f"  EEG: {eeg_data.shape[1]} channels, {eeg_data.shape[2]} samples/epoch, {eeg_fs} Hz")

    if eog_channels:
        eog_raw = raw.copy().pick_channels(eog_channels)
        eog_data, eog_fs = _extract_epochs(eog_raw, epoch_length, n_epochs)
        multi_channel_data['eog'] = eog_data
        sampling_rates['eog'] = eog_fs
        print(f"  EOG: {eog_data.shape[1]} channels, {eog_data.shape[2]} samples/epoch, {eog_fs} Hz")

    if emg_channels:
        emg_raw = raw.copy().pick_channels(emg_channels)
        emg_data, emg_fs = _extract_epochs(emg_raw, epoch_length, n_epochs)
        multi_channel_data['emg'] = emg_data
        sampling_rates['emg'] = emg_fs
        print(f"  EMG: {emg_data.shape[1]} channels, {emg_data.shape[2]} samples/epoch, {emg_fs} Hz")

    # Create record info
    record_info = {
        'record_id': record_id,
        'n_epochs': n_epochs,
        'channels': eeg_channels + eog_channels + emg_channels,
        'sampling_rates': sampling_rates,
        'epoch_length': epoch_length
    }

    print(f"Loaded {n_epochs} epochs ({n_epochs*epoch_length/3600:.2f} hours)")

    return multi_channel_data, record_info


def _extract_epochs(raw, epoch_length, n_epochs):
    """
    Extract fixed-length epochs from continuous MNE Raw data.

    Args:
        raw (mne.io.Raw): MNE Raw object
        epoch_length (float): Epoch duration in seconds
        n_epochs (int): Number of epochs to extract

    Returns:
        tuple: (epochs_array, sampling_rate) where:
            - epochs_array: np.ndarray, shape (n_epochs, n_channels, samples_per_epoch)
            - sampling_rate: float, sampling frequency in Hz
    """
    # Get data and sampling rate
    data = raw.get_data()  # Shape: (n_channels, n_samples)
    fs = raw.info['sfreq']
    n_channels = data.shape[0]

    # Calculate samples per epoch
    samples_per_epoch = int(epoch_length * fs)

    # Calculate total samples needed
    total_samples_needed = n_epochs * samples_per_epoch

    # Trim or pad data if necessary
    if data.shape[1] > total_samples_needed:
        data = data[:, :total_samples_needed]
    elif data.shape[1] < total_samples_needed:
        # Pad with zeros if needed
        padding = total_samples_needed - data.shape[1]
        data = np.pad(data, ((0, 0), (0, padding)), mode='constant')

    # Reshape into epochs: (n_epochs, n_channels, samples_per_epoch)
    epochs = data.reshape(n_channels, n_epochs, samples_per_epoch)
    epochs = np.transpose(epochs, (1, 0, 2))  # (n_epochs, n_channels, samples)

    return epochs, fs


def _print_label_distribution(labels):
    """Print sleep stage distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    print("Sleep stage distribution:")
    for stage, count in zip(unique, counts):
        if stage < len(stage_names):
            pct = (count / len(labels)) * 100
            print(f"  {stage_names[stage]}: {count} epochs ({pct:.1f}%)")


def load_all_training_data(training_dir, epoch_length=30, use_single_recording=True):
    """
    Load all training recordings from a directory.

    According to the project guide, this function should:
    1. Loop through all EDF files in the training directory
    2. Use load_single_recording() for each recording (for iteration 1)
    3. Concatenate epochs and labels from all recordings
    4. Track record IDs for LOSO cross-validation
    5. Return in format expected by the pipeline

    Args:
        training_dir (str): Path to directory containing EDF and XML files
        epoch_length (float): Epoch duration in seconds (default 30)
        use_single_recording (bool): If True, use load_single_recording (iteration 1 style).
                                     If False, use load_training_data (multi-channel).

    Returns:
        tuple: (all_data, all_labels, all_record_ids, channel_info) where:
            - all_data (dict): Combined multi-channel data from all recordings
                - 'eeg': np.ndarray, shape (n_epochs, n_channels, samples_per_epoch)
                - 'eog': np.ndarray (if available, for later iterations)
                - 'emg': np.ndarray (if available, for later iterations)
            - all_labels (np.ndarray): Concatenated labels, shape (n_epochs,)
            - all_record_ids (np.ndarray): Record ID for each epoch, shape (n_epochs,)
            - channel_info (dict): Channel information (sampling rates, channel names)

    Example:
        >>> data, labels, record_ids, info = load_all_training_data('data/training/')
        >>> print(f"Total epochs: {len(labels)}")
        >>> print(f"Unique recordings: {len(np.unique(record_ids))}")
        >>> print(f"EEG shape: {data['eeg'].shape}")
    """
    from glob import glob

    print(f"Loading all training data from {training_dir}...")

    # Find all EDF files
    edf_files = sorted(glob(os.path.join(training_dir, '*.edf')))

    if not edf_files:
        raise FileNotFoundError(f"No EDF files found in {training_dir}")

    print(f"Found {len(edf_files)} recordings")

    # Initialize lists to store data from all recordings
    all_eeg = []
    all_eog = []
    all_emg = []
    all_labels = []
    all_record_ids = []
    channel_info = None

    # Load each recording
    for edf_file in edf_files:
        # Get corresponding XML file
        xml_file = edf_file.replace('.edf', '.xml')

        if not os.path.exists(xml_file):
            print(f"  WARNING: Skipping {edf_file} - no corresponding XML file")
            continue

        # Extract record ID from filename
        record_id = Path(edf_file).stem

        print(f"\nLoading {record_id}...")

        try:
            if use_single_recording:
                # Use load_single_recording for iteration 1 (EEG only)
                epochs, labels = load_single_recording(edf_file, xml_file, epoch_length)

                # Wrap in dict format for compatibility
                multi_channel_data = {'eeg': epochs}

                # Create channel info from first recording
                if channel_info is None:
                    # Extract sampling rate and channel info from epochs shape
                    # epochs shape: (n_epochs, n_channels, samples_per_epoch)
                    samples_per_epoch = epochs.shape[2]
                    fs = samples_per_epoch / epoch_length
                    n_channels = epochs.shape[1]

                    # Get channel names from load_single_recording output
                    # load_single_recording already printed the selected channels
                    # We can extract this info without reloading the file
                    # Since we know n_channels, we can infer from the epochs structure
                    # For now, use a placeholder - the actual channel names are less critical
                    # than the sampling rate and channel count
                    channel_info = {
                        'eeg_names': [f'EEG_{i+1}' for i in range(n_channels)],  # Placeholder names
                        'eeg_fs': float(fs),
                        'epoch_length': epoch_length
                    }

                    print(f"  Channel info: {n_channels} channels, fs={fs:.1f} Hz")

            else:
                # Use load_training_data for multi-channel (iterations 2-4)
                multi_channel_data, labels, info = load_training_data(
                    edf_file, xml_file, epoch_length
                )

                # Store channel info from first recording
                if channel_info is None:
                    channel_info = info

            # Append data
            if 'eeg' in multi_channel_data:
                all_eeg.append(multi_channel_data['eeg'])
            if 'eog' in multi_channel_data:
                all_eog.append(multi_channel_data['eog'])
            if 'emg' in multi_channel_data:
                all_emg.append(multi_channel_data['emg'])

            all_labels.append(labels)

            # Track record ID for each epoch (critical for LOSO cross-validation)
            all_record_ids.extend([record_id] * len(labels))

        except Exception as e:
            print(f"  ERROR loading {record_id}: {type(e).__name__}: {e}")
            continue

    # Concatenate all recordings
    combined_data = {}

    if all_eeg:
        combined_data['eeg'] = np.concatenate(all_eeg, axis=0)
        print(f"\nCombined EEG shape: {combined_data['eeg'].shape}")

    if all_eog:
        combined_data['eog'] = np.concatenate(all_eog, axis=0)
        print(f"Combined EOG shape: {combined_data['eog'].shape}")

    if all_emg:
        combined_data['emg'] = np.concatenate(all_emg, axis=0)
        print(f"Combined EMG shape: {combined_data['emg'].shape}")

    combined_labels = np.concatenate(all_labels, axis=0)
    combined_record_ids = np.array(all_record_ids)

    print(f"\nTotal loaded: {len(combined_labels)} epochs from {len(np.unique(all_record_ids))} recordings")
    print(f"Record IDs: {sorted(np.unique(all_record_ids))}")
    _print_label_distribution(combined_labels)

    return combined_data, combined_labels, combined_record_ids, channel_info


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) > 2:
        edf_file = sys.argv[1]
        xml_file = sys.argv[2]

        print(f"Loading {edf_file} and {xml_file}")
        data, labels, info = load_training_data(edf_file, xml_file)

        print(f"\nSummary:")
        print(f"  Total epochs: {labels.shape[0]}")
        for signal_type in data.keys():
            print(f"  {signal_type.upper()} shape: {data[signal_type].shape}")

    else:
        print("Usage: python data_loader.py <edf_file> <xml_file>")
        print("Example: python data_loader.py data/training/R1.edf data/training/R1.xml")