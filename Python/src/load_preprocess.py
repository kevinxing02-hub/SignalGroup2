# Patches: strengthened channel_info in loader + strict shape/fs validation in preprocess
# Save/replace relevant functions in your project with these implementations.
# These functions assume the rest of your data_loader and eeg_preprocessing_iter3 modules
# and helper functions (compute_recording_info, _extract_epochs_from_raw, parse_xml_annotations, create_epoch_labels, pad_or_trim_signal, resample_raw_if_needed, identify_channels, align_epochs_to_canonical, _safe_pad_labels, etc.) remain available.

import os
from pathlib import Path
import numpy as np
import mne

# -------------------------
# Updated load_training_data
# -------------------------
def load_training_data(edf_file_path, xml_file_path, epoch_length=30, target_fs=None, canonical_channel_lists=None):
    """
    Strengthened loader that explicitly records per-group sampling rates and samples_per_epoch
    into channel_info. This makes downstream preprocessing/feature extraction deterministic.

    Returns:
      multi_channel_data (dict of arrays), labels (n_epochs,), channel_info (dict)
    """
    if not os.path.exists(edf_file_path):
        raise FileNotFoundError(f"EDF not found: {edf_file_path}")
    if not os.path.exists(xml_file_path):
        raise FileNotFoundError(f"XML not found: {xml_file_path}")

    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)

    # Resample if requested (operate on full recording so group sample counts are consistent)
    if target_fs is not None:
        raw = resample_raw_if_needed(raw, target_fs)

    rec_info = compute_recording_info(raw, epoch_length)
    n_epochs = rec_info['n_epochs']

    # Parse XML annotations and create labels
    parsed_xml = parse_xml_annotations(xml_file_path)
    stages = parsed_xml.get('stages', [])
    labels = create_epoch_labels(stages, rec_info['duration'], epoch_length)
    labels = _safe_pad_labels(labels, n_epochs)

    channel_names = raw.ch_names
    eeg_channels, eog_channels, emg_channels, other_channels = identify_channels(channel_names)

    multi_channel_data = {}
    # channel_info will include per-group fs and samples_per_epoch where possible
    channel_info = {
        'epoch_length': epoch_length,
        'record_id': Path(edf_file_path).stem,
        'record_duration': rec_info['duration'],
        'global_fs': rec_info['fs'],
        'global_samps_per_epoch': rec_info['samples_per_epoch']
    }

    # Helper to extract and record group info
    def _extract_and_record(group_name, ch_list):
        if not ch_list:
            return None
        picked = raw.copy().pick_channels(ch_list)
        epochs, fs = _extract_epochs_from_raw(picked, epoch_length, n_epochs)
        # store
        multi_channel_data[group_name] = epochs
        channel_info[f"{group_name}_fs"] = fs
        channel_info[f"{group_name}_samps_per_epoch"] = epochs.shape[2]
        channel_info[f"{group_name}_names"] = ch_list
        return epochs

    _extract_and_record('eeg', eeg_channels)
    _extract_and_record('eog', eog_channels)
    _extract_and_record('emg', emg_channels)
    _extract_and_record('other', other_channels)

    # Align to canonical names if provided (makes concatenation safe across recordings)
    if canonical_channel_lists:
        for key in ['eeg', 'eog', 'emg', 'other']:
            if key in canonical_channel_lists:
                desired = canonical_channel_lists[key]
                if key in multi_channel_data:
                    from_names = channel_info.get(f'{key}_names', [])
                    multi_channel_data[key] = align_epochs_to_canonical(multi_channel_data[key], from_names, desired)
                    # overwrite recorded names/samps if needed
                    channel_info[f'{key}_names'] = desired
                    channel_info[f'{key}_samps_per_epoch'] = multi_channel_data[key].shape[2]
                    # fs remains unchanged — alignment is reordering, not resampling
                else:
                    # recording missing this group: create explicit zero-array using conservative samples_per_epoch
                    samps = channel_info.get('global_samps_per_epoch')
                    if key in channel_info:
                        samps = channel_info.get(f'{key}_samps_per_epoch', samps)
                    if samps is None:
                        # fallback: compute expected samples using target_fs or 125
                        assumed_fs = target_fs if target_fs is not None else channel_info.get('global_fs', 125)
                        samps = int(epoch_length * assumed_fs)
                    multi_channel_data[key] = np.zeros((n_epochs, len(desired), samps))
                    channel_info[f'{key}_names'] = desired
                    channel_info[f'{key}_samps_per_epoch'] = samps
                    channel_info[f'{key}_fs'] = (samps / epoch_length) if epoch_length > 0 else None

    # Summary prints
    print(f"Loaded {channel_info['record_id']}: duration={rec_info['duration']:.1f}s, epochs={n_epochs}, global_fs={rec_info['fs']} Hz")
    print(f"  EEG: {len(eeg_channels)}  EOG: {len(eog_channels)}  EMG: {len(emg_channels)}  Other: {len(other_channels)}")

    return multi_channel_data, labels, channel_info


# -------------------------
# Minimal helper: updated zero-array creation in bulk loader
# (to be used inside load_all_training_data where previously a hardcoded 125 was used)
# -------------------------

def _create_zero_group(n_epochs, canonical_names, samples_per_epoch=None, epoch_length=30, target_fs=None, global_fs=None):
    """
    Create zero array for missing group with careful determination of samples_per_epoch.
    """
    if samples_per_epoch is None:
        if target_fs is not None:
            samples_per_epoch = int(epoch_length * target_fs)
        elif global_fs is not None:
            samples_per_epoch = int(epoch_length * global_fs)
        else:
            samples_per_epoch = int(epoch_length * 125)  # conservative fallback
    return np.zeros((n_epochs, len(canonical_names), samples_per_epoch)), samples_per_epoch


# -------------------------
# Updated preprocess_multi_channel (from eeg_preprocessing_iter3.py)
# -------------------------
from scipy.signal import filtfilt

def preprocess_multi_channel(multi_channel_data: dict, config, channel_info: dict = None) -> dict:
    """
    Preprocess each group and strictly validate samples_per_epoch against channel_info.
    Raises informative errors when mismatches occur so caller can correct loader issues.
    """
    preprocessed_data = {}

    # ---------- EEG ----------
    if 'eeg' in multi_channel_data:
        eeg_data = multi_channel_data['eeg']
        n_epochs, n_channels, samples_per_epoch = eeg_data.shape

        # If channel_info provided, validate
        if channel_info is not None:
            expected = channel_info.get('eeg_samps_per_epoch')
            if expected is not None and int(expected) != int(samples_per_epoch):
                raise ValueError(f"EEG samples_per_epoch mismatch: data has {samples_per_epoch}, channel_info lists {expected}.\n"
                                 "This typically indicates the loader produced inconsistent epoch lengths or a resampling discrepancy.")
            # ensure fs is available
            eeg_fs = channel_info.get('eeg_fs', samples_per_epoch / 30.0)
        else:
            eeg_fs = samples_per_epoch / 30.0

        # design filters using validated eeg_fs
        hp_cut = 0.3
        lp_cut = 40.0
        notch_f = 50.0
        b_hp, a_hp = butter_highpass(hp_cut, eeg_fs, order=2)
        b_lp, a_lp = butter_lowpass(lp_cut, eeg_fs, order=4)

        # process continuous
        continuous_eeg = eeg_data.transpose(1, 0, 2).reshape(n_channels, -1)
        preprocessed_continuous_eeg = np.zeros_like(continuous_eeg)

        for ch in range(n_channels):
            x = continuous_eeg[ch, :].copy()
            x = filtfilt(b_hp, a_hp, x)
            x = apply_notch(x, notch_f, int(eeg_fs), Q=30)
            # optional 100Hz notch if present in data
            if 100.0 < 0.5 * eeg_fs:
                x = apply_notch(x, 100.0, int(eeg_fs), Q=30)
            x = filtfilt(b_lp, a_lp, x)
            preprocessed_continuous_eeg[ch, :] = x

        preprocessed_eeg = preprocessed_continuous_eeg.reshape(n_channels, n_epochs, samples_per_epoch).transpose(1, 0, 2)
        preprocessed_data['eeg'] = preprocessed_eeg

    # ---------- EOG ----------
    if 'eog' in multi_channel_data and config.CURRENT_ITERATION >= 2:
        eog_data = multi_channel_data['eog']
        n_eog_epochs, n_eog_channels, eog_samps = eog_data.shape

        # validation
        if channel_info is not None:
            expected = channel_info.get('eog_samps_per_epoch')
            if expected is not None and int(expected) != int(eog_samps):
                raise ValueError(f"EOG samples_per_epoch mismatch: data {eog_samps} vs channel_info {expected}")
            eog_fs = channel_info.get('eog_fs', eog_samps / 30.0)
        else:
            eog_fs = eog_samps / 30.0

        b_eog_hp, a_eog_hp = butter_highpass(0.5, eog_fs, order=2)
        b_eog_lp, a_eog_lp = butter_lowpass(30.0, eog_fs, order=4)

        continuous_eog = eog_data.transpose(1, 0, 2).reshape(n_eog_channels, -1)
        preprocessed_continuous_eog = np.zeros_like(continuous_eog)

        for ch in range(n_eog_channels):
            x = continuous_eog[ch, :].copy()
            x = filtfilt(b_eog_hp, a_eog_hp, x)
            x = apply_notch(x, 50.0, int(eog_fs), Q=30)
            if 100.0 < 0.5 * eog_fs:
                x = apply_notch(x, 100.0, int(eog_fs), Q=30)
            x = filtfilt(b_eog_lp, a_eog_lp, x)
            preprocessed_continuous_eog[ch, :] = x

        preprocessed_eog = preprocessed_continuous_eog.reshape(n_eog_channels, n_eog_epochs, eog_samps).transpose(1, 0, 2)
        preprocessed_data['eog'] = preprocessed_eog

    # ---------- EMG ----------
    if 'emg' in multi_channel_data and config.CURRENT_ITERATION >= 3:
        emg_data = multi_channel_data['emg']
        n_emg_epochs, n_emg_ch, emg_samps = emg_data.shape

        if channel_info is not None:
            expected = channel_info.get('emg_samps_per_epoch')
            if expected is not None and int(expected) != int(emg_samps):
                raise ValueError(f"EMG samples_per_epoch mismatch: data {emg_samps} vs channel_info {expected}")
            emg_fs = channel_info.get('emg_fs', emg_samps / 30.0)
        else:
            emg_fs = emg_samps / 30.0

        preprocessed_emg = np.zeros_like(emg_data)
        for ep in range(n_emg_epochs):
            for ch in range(n_emg_ch):
                signal = emg_data[ep, ch, :]
                filtered_signal = lowpass_filter(signal, 70, int(emg_fs))
                preprocessed_emg[ep, ch, :] = filtered_signal
        preprocessed_data['emg'] = preprocessed_emg

    # ---------- Iteration 3 artifact handling (EOG regression, EMG-adaptive) ----------
    if config.CURRENT_ITERATION >= 3:
        if 'eog' in preprocessed_data and 'eeg' in preprocessed_data:
            print("Iteration 3: Removing EOG artifacts from EEG using LinearRegression...")
            preprocessed_data['eeg'] = remove_eog_artifacts_from_eeg(preprocessed_data['eeg'], preprocessed_data['eog'])
        if 'emg' in preprocessed_data and 'eeg' in preprocessed_data:
            print("Iteration 3: Applying EMG-based adaptive low-pass filtering on EEG...")
            eeg_fs_val = channel_info.get('eeg_fs', None) if channel_info else None
            emg_fs_val = channel_info.get('emg_fs', None) if channel_info else None
            preprocessed_data['eeg'] = apply_emg_adaptive_filtering(preprocessed_data['eeg'], preprocessed_data['emg'], eeg_fs=int(eeg_fs_val) if eeg_fs_val else int(samples_per_epoch/30.0), emg_fs=int(emg_fs_val) if emg_fs_val else int(emg_samps/30.0))

    return preprocessed_data
