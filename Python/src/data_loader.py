"""
Advanced Data Loader Module (Iteration 4 ready)

- Loads EDF + XML annotations
- Supports EEG, EOG, EMG, and other channels
- Robust channel detection, resampling, alignment and epoching
- Returns multi-channel epoch arrays (n_epochs, n_channels, samples_per_epoch)
"""

import numpy as np
import mne
import os
from pathlib import Path
from glob import glob

# Import xml parser from project
try:
    from .xml_parser import parse_xml_annotations, create_epoch_labels
except Exception:
    from xml_parser import parse_xml_annotations, create_epoch_labels

# -------------------------
# Helpers
# -------------------------
def unique_ordered(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def identify_channels(channel_names):
    """
    Return (eeg_channels, eog_channels, emg_channels, other_channels)
    Uses broad patterns to capture many naming conventions.
    """
    up = [c.upper() for c in channel_names]

    eog_patterns = ['EOG', 'HEOG', 'VEOG', 'LOC', 'ROC', 'EOGL', 'EOGR', 'E1', 'E2', 'LEFT EYE', 'RIGHT EYE']
    emg_patterns = ['EMG', 'CHIN', 'LEG', 'TIB']
    eeg_patterns = ['EEG', 'C3', 'C4', 'F3', 'F4', 'O1', 'O2', 'CZ', 'FZ', 'PZ', 'T3', 'T4', 'T5', 'T6', 'FP1', 'FP2']

    eog = [ch for ch in channel_names if any(p in ch.upper() for p in eog_patterns)]
    emg = [ch for ch in channel_names if any(p in ch.upper() for p in emg_patterns)]
    eeg_cand = [ch for ch in channel_names if any(p in ch.upper() for p in eeg_patterns)]
    eeg = [ch for ch in eeg_cand if ch not in eog and ch not in emg]

    # everything else as "other" (respiratory, oxygen, ECG etc.)
    other = [ch for ch in channel_names if ch not in eeg and ch not in eog and ch not in emg]

    return unique_ordered(eeg), unique_ordered(eog), unique_ordered(emg), unique_ordered(other)

def compute_recording_info(raw, epoch_length):
    """Return fs, n_samples, duration (sec), n_epochs (ceil), samples_per_epoch."""
    fs = float(raw.info['sfreq'])
    n_samples = int(raw.n_times)
    duration = n_samples / fs
    samples_per_epoch = int(epoch_length * fs)
    n_epochs = int(np.ceil(duration / epoch_length))
    total_needed = n_epochs * samples_per_epoch
    return {'fs': fs, 'n_samples': n_samples, 'duration': duration,
            'n_epochs': n_epochs, 'samples_per_epoch': samples_per_epoch,
            'total_needed': total_needed}

def pad_or_trim_signal(data, total_needed):
    """Data shape: (n_channels, n_samples). Return (n_channels, total_needed)."""
    if data.shape[1] > total_needed:
        return data[:, :total_needed]
    elif data.shape[1] < total_needed:
        pad = total_needed - data.shape[1]
        return np.pad(data, ((0,0),(0,pad)), mode='constant')
    return data

def resample_raw_if_needed(raw, target_fs):
    """Return raw copy resampled to target_fs if needed (inplace copy avoided)."""
    fs = float(raw.info['sfreq'])
    if fs == target_fs:
        return raw
    raw_rs = raw.copy()
    raw_rs.resample(target_fs, npad="auto")
    return raw_rs

def _safe_pad_labels(labels, n_epochs):
    """Pad or trim labels to length n_epochs. If labels empty, fill with zeros."""
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return np.zeros(n_epochs, dtype=int)
    if labels.size > n_epochs:
        return labels[:n_epochs]
    if labels.size < n_epochs:
        last = labels[-1] if labels.size > 0 else 0
        return np.concatenate([labels, np.full(n_epochs - labels.size, last, dtype=int)])
    return labels

def align_epochs_to_canonical(epochs, from_names, canonical_names):
    """
    epochs: (n_epochs, n_channels_from, samples)
    from_names: list with length n_channels_from
    canonical_names: list with length n_canonical
    Returns: aligned (n_epochs, n_canonical, samples) where missing channels are zeros
    """
    n_epochs, _, n_samples = epochs.shape
    aligned = np.zeros((n_epochs, len(canonical_names), n_samples), dtype=epochs.dtype)

    for i, ch in enumerate(canonical_names):
        if ch in from_names:
            idx = from_names.index(ch)
            aligned[:, i, :] = epochs[:, idx, :]
        else:
            # remain zeros if channel missing
            pass
    return aligned

# -------------------------
# Core epoch extraction
# -------------------------
def _extract_epochs_from_raw(raw, epoch_length, n_epochs=None):
    """
    Extract epochs from an MNE Raw object.
    If n_epochs is None, compute from raw's samples (ceil).
    Returns: epochs (n_epochs, n_channels, samples_per_epoch), fs
    """
    info = compute_recording_info(raw, epoch_length)
    fs = info['fs']
    if n_epochs is None:
        n_epochs = info['n_epochs']
    samples_per_epoch = info['samples_per_epoch']
    total_needed = n_epochs * samples_per_epoch

    data = raw.get_data()  # (n_channels, n_samples)
    data = pad_or_trim_signal(data, total_needed)
    n_channels = data.shape[0]

    epochs = data.reshape(n_channels, n_epochs, samples_per_epoch)
    epochs = np.transpose(epochs, (1, 0, 2))  # (n_epochs, n_channels, samples_per_epoch)

    return epochs, fs

# -------------------------
# Loaders
# -------------------------
def load_training_data(edf_file_path, xml_file_path, epoch_length=30, target_fs=None, canonical_channel_lists=None):
    """
    Load EDF + XML for one recording (multi-channel).
    - target_fs: if set, resample raw to this fs before epoching
    - canonical_channel_lists: dict like {'eeg': [...], 'eog': [...], 'emg': [...], 'other': [...]}
       if provided, epochs will be aligned to those lists (missing channels zero-padded).
    Returns:
      multi_channel_data (dict of arrays), labels (n_epochs,), channel_info (dict)
    """
    if not os.path.exists(edf_file_path):
        raise FileNotFoundError(f"EDF not found: {edf_file_path}")
    if not os.path.exists(xml_file_path):
        raise FileNotFoundError(f"XML not found: {xml_file_path}")

    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)

    # Resample if requested
    if target_fs is not None:
        raw = resample_raw_if_needed(raw, target_fs)

    rec_info = compute_recording_info(raw, epoch_length)
    n_epochs = rec_info['n_epochs']

    # Parse XML annotations and create labels (create_epoch_labels expected to handle durations)
    parsed_xml = parse_xml_annotations(xml_file_path)
    stages = parsed_xml.get('stages', [])
    # use recording duration (computed from samples) so create_epoch_labels matches epoching
    labels = create_epoch_labels(stages, rec_info['duration'], epoch_length)
    labels = _safe_pad_labels(labels, n_epochs)

    channel_names = raw.ch_names
    eeg_channels, eog_channels, emg_channels, other_channels = identify_channels(channel_names)

    multi_channel_data = {}
    # CHANGED: channel_info now records per-group fs, samples_per_epoch and channel names
    channel_info = {
        'epoch_length': epoch_length,
        'global_fs': rec_info['fs'],
        'global_samps_per_epoch': rec_info['samples_per_epoch'],
        'duration': rec_info['duration'],
        'record_id': Path(edf_file_path).stem
    }

    # Extract per-type epochs if present and record per-group metadata
    if eeg_channels:
        eeg_raw = raw.copy().pick_channels(eeg_channels)
        eeg_epochs, eeg_fs = _extract_epochs_from_raw(eeg_raw, epoch_length, n_epochs)
        multi_channel_data['eeg'] = eeg_epochs
        channel_info['eeg_fs'] = eeg_fs
        channel_info['eeg_samps_per_epoch'] = eeg_epochs.shape[2]
        channel_info['eeg_names'] = eeg_channels

    if eog_channels:
        eog_raw = raw.copy().pick_channels(eog_channels)
        eog_epochs, eog_fs = _extract_epochs_from_raw(eog_raw, epoch_length, n_epochs)
        multi_channel_data['eog'] = eog_epochs
        channel_info['eog_fs'] = eog_fs
        channel_info['eog_samps_per_epoch'] = eog_epochs.shape[2]
        channel_info['eog_names'] = eog_channels

    if emg_channels:
        emg_raw = raw.copy().pick_channels(emg_channels)
        emg_epochs, emg_fs = _extract_epochs_from_raw(emg_raw, epoch_length, n_epochs)
        multi_channel_data['emg'] = emg_epochs
        channel_info['emg_fs'] = emg_fs
        channel_info['emg_samps_per_epoch'] = emg_epochs.shape[2]
        channel_info['emg_names'] = emg_channels

    if other_channels:
        other_raw = raw.copy().pick_channels(other_channels)
        other_epochs, other_fs = _extract_epochs_from_raw(other_raw, epoch_length, n_epochs)
        multi_channel_data['other'] = other_epochs
        channel_info['other_fs'] = other_fs
        channel_info['other_samps_per_epoch'] = other_epochs.shape[2]
        channel_info['other_names'] = other_channels

    # Align to canonical names if provided (makes concatenation safe across recordings)
    if canonical_channel_lists:
        for key in ['eeg', 'eog', 'emg', 'other']:
            if key in canonical_channel_lists and key in multi_channel_data:
                # CHANGED: use recorded from_names (if present) when aligning
                from_names = channel_info.get(f'{key}_names', [])
                canonical_names = canonical_channel_lists[key]
                multi_channel_data[key] = align_epochs_to_canonical(multi_channel_data[key], from_names, canonical_names)
                channel_info[f'{key}_names'] = canonical_names
                # after alignment, samples_per_epoch unchanged; names updated
            # If canonical asks for channel but recording missing, create zeros
            elif key in canonical_channel_lists and key not in multi_channel_data:
                # CHANGED: determine n_samples for zero array using best available info
                # priority: canonical_channel_lists -> channel_info specific group -> global samples per epoch -> target_fs -> fallback 125 Hz
                if key + '_samps_per_epoch' in channel_info:
                    n_samples = int(channel_info[f'{key}_samps_per_epoch'])
                else:
                    # prefer global_samps_per_epoch (from rec_info)
                    n_samples = int(channel_info.get('global_samps_per_epoch', 0))
                    if n_samples == 0:
                        # try to infer from target_fs if caller passed it via channel_info 'target_fs' (not standard)
                        assumed_fs = channel_info.get('target_fs', None)
                        if assumed_fs is None:
                            # fallback to global fs or 125
                            assumed_fs = channel_info.get('global_fs', 125)
                        n_samples = int(epoch_length * assumed_fs)
                multi_channel_data[key] = np.zeros((n_epochs, len(canonical_channel_lists[key]), n_samples))
                channel_info[f'{key}_names'] = canonical_channel_lists[key]
                channel_info[f'{key}_samps_per_epoch'] = n_samples
                channel_info[f'{key}_fs'] = (n_samples / epoch_length) if epoch_length > 0 else None

    # Print summary
    print(f"Loaded {Path(edf_file_path).stem}: duration={rec_info['duration']:.1f}s, epochs={n_epochs}, fs={rec_info['fs']} Hz")
    print(f"  EEG: {len(eeg_channels)}  EOG: {len(eog_channels)}  EMG: {len(emg_channels)}  Other: {len(other_channels)}")

    return multi_channel_data, labels, channel_info

def load_holdout_data(edf_file_path, epoch_length=30, target_fs=None, canonical_channel_lists=None):
    """
    Load a holdout EDF (no XML). Returns multi_channel_data, record_info.
    record_info includes record_id, n_epochs, channels, sampling_rates, epoch_length.
    """
    if not os.path.exists(edf_file_path):
        raise FileNotFoundError(f"EDF not found: {edf_file_path}")

    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)

    if target_fs is not None:
        raw = resample_raw_if_needed(raw, target_fs)

    rec_info = compute_recording_info(raw, epoch_length)
    n_epochs = rec_info['n_epochs']

    channel_names = raw.ch_names
    eeg_channels, eog_channels, emg_channels, other_channels = identify_channels(channel_names)

    multi_channel_data = {}
    sampling_rates = {}
    if eeg_channels:
        eeg_raw = raw.copy().pick_channels(eeg_channels)
        eeg_epochs, eeg_fs = _extract_epochs_from_raw(eeg_raw, epoch_length, n_epochs)
        multi_channel_data['eeg'] = eeg_epochs
        sampling_rates['eeg'] = eeg_fs
    if eog_channels:
        eog_raw = raw.copy().pick_channels(eog_channels)
        eog_epochs, eog_fs = _extract_epochs_from_raw(eog_raw, epoch_length, n_epochs)
        multi_channel_data['eog'] = eog_epochs
        sampling_rates['eog'] = eog_fs
    if emg_channels:
        emg_raw = raw.copy().pick_channels(emg_channels)
        emg_epochs, emg_fs = _extract_epochs_from_raw(emg_raw, epoch_length, n_epochs)
        multi_channel_data['emg'] = emg_epochs
        sampling_rates['emg'] = emg_fs
    if other_channels:
        other_raw = raw.copy().pick_channels(other_channels)
        other_epochs, other_fs = _extract_epochs_from_raw(other_raw, epoch_length, n_epochs)
        multi_channel_data['other'] = other_epochs
        sampling_rates['other'] = other_fs

    # Align to canonical if requested (ensures output channels order consistent)
    if canonical_channel_lists:
        for key in ['eeg', 'eog', 'emg', 'other']:
            if key in canonical_channel_lists and key in multi_channel_data:
                from_names = eeg_channels if key == 'eeg' else (eog_channels if key == 'eog' else (emg_channels if key == 'emg' else other_channels))
                canonical_names = canonical_channel_lists[key]
                multi_channel_data[key] = align_epochs_to_canonical(multi_channel_data[key], from_names, canonical_names)
            elif key in canonical_channel_lists and key not in multi_channel_data:
                n_samples = rec_info['samples_per_epoch']
                multi_channel_data[key] = np.zeros((n_epochs, len(canonical_channel_lists[key]), n_samples))

    record_info = {
        'record_id': Path(edf_file_path).stem,
        'n_epochs': n_epochs,
        'channels': eeg_channels + eog_channels + emg_channels + other_channels,
        'sampling_rates': sampling_rates,
        'epoch_length': epoch_length
    }
    print(f"Holdout {record_info['record_id']}: epochs={n_epochs}, fs={rec_info['fs']} Hz, channels={len(record_info['channels'])}")
    return multi_channel_data, record_info

# -------------------------
# Aggregate loader for directory (LOSO-ready)
# -------------------------
def load_all_training_data(training_dir, epoch_length=30, use_single_recording=False,
                           target_fs=None, canonical_channel_lists=None):
    """
    Load all training recordings from a directory and concatenate them safely.

    - training_dir: folder with .edf + .xml pairs
    - use_single_recording: left for backward compatibility (if True will call load_single_recording)
      For iteration >=4 you should set this False.
    - target_fs: optional target sampling rate to resample all recordings to (recommended)
    - canonical_channel_lists: optional canonical channel ordering to align all recordings.
        Format: {'eeg': [...], 'eog': [...], 'emg': [...], 'other': [...]}
        If None, the first non-empty recording's channel lists are used as canonical.
    Returns:
       combined_data (dict), combined_labels (np.ndarray), combined_record_ids (np.ndarray), channel_info (dict)
    """
    edf_files = sorted(glob(os.path.join(training_dir, '*.edf')))
    if not edf_files:
        raise FileNotFoundError(f"No EDF files in {training_dir}")

    all_eeg = []; all_eog = []; all_emg = []; all_other = []
    all_labels = []; all_record_ids = []
    channel_info = None

    # If canonical channels not provided, we'll infer from first recording that has data
    inferred_canonical = canonical_channel_lists is not None

    for edf in edf_files:
        xml = edf.replace('.edf', '.xml')
        if not os.path.exists(xml):
            print(f"WARNING: skipping {edf} (no XML annotation found)")
            continue

        record_id = Path(edf).stem
        print(f"\nProcessing {record_id}...")

        try:
            # For iteration 4 prefer multi-channel loader
            if use_single_recording:
                # keep earlier behavior: single EEG-only loader (deprecated for iter>=4)
                epochs, labels = load_single_recording(edf, xml, epoch_length)
                multi_channel_data = {'eeg': epochs}
                info = {'eeg_names': [f'EEG_{i+1}' for i in range(epochs.shape[1])], 'eeg_fs': epochs.shape[2] / epoch_length, 'epoch_length': epoch_length}
            else:
                multi_channel_data, labels, info = load_training_data(edf, xml, epoch_length,
                                                                      target_fs=target_fs,
                                                                      canonical_channel_lists=(canonical_channel_lists if inferred_canonical else None))

            # If canonical not provided, set it from the first info we get
            if not inferred_canonical and channel_info is None:
                # build canonical lists from this recording's channel names (if present)
                canonical_channel_lists = {}
                for key in ['eeg', 'eog', 'emg', 'other']:
                    key_names = info.get(f'{key}_names', None)
                    if key_names:
                        canonical_channel_lists[key] = key_names
                inferred_canonical = True  # from now on use this canonical
                print("Inferred canonical channel lists from first recording.")

            # If canonical defined but this reading did not align, re-run alignment for this recording
            if canonical_channel_lists:
                # ensure current recording conforms to canonical lists
                for key in ['eeg', 'eog', 'emg', 'other']:
                    if key in canonical_channel_lists:
                        if key in multi_channel_data:
                            from_names = info.get(f'{key}_names', [])
                            if from_names != canonical_channel_lists[key]:
                                # align
                                multi_channel_data[key] = align_epochs_to_canonical(multi_channel_data[key], from_names, canonical_channel_lists[key])
                                info[f'{key}_names'] = canonical_channel_lists[key]
                        else:
                            # create zero array for missing channel group
                            n_epochs = int(np.ceil(info.get('duration', (labels.size*epoch_length)) / epoch_length)) if 'duration' in info else labels.size

                            # CHANGED: determine n_samples using info's group samp-per-epoch if available,
                            # otherwise fallback to target_fs/global fs or overall rec_info
                            if f'{key}_samps_per_epoch' in info:
                                n_samples = int(info[f'{key}_samps_per_epoch'])
                            else:
                                # try per-group fs if present
                                grp_fs = info.get(f'{key}_fs', None)
                                if grp_fs:
                                    n_samples = int(epoch_length * grp_fs)
                                else:
                                    # try canonical from previously inferred channel_info (outer variable)
                                    if channel_info is not None and f'{key}_samps_per_epoch' in channel_info:
                                        n_samples = int(channel_info[f'{key}_samps_per_epoch'])
                                    else:
                                        # fallback to provided target_fs or info global fs or 125
                                        assumed_fs = target_fs if target_fs is not None else info.get('global_fs', 125)
                                        n_samples = int(epoch_length * assumed_fs)

                            multi_channel_data[key] = np.zeros((n_epochs, len(canonical_channel_lists[key]), n_samples))
                            info[f'{key}_names'] = canonical_channel_lists[key]
                            info[f'{key}_samps_per_epoch'] = n_samples
                            info[f'{key}_fs'] = (n_samples / epoch_length)

            # Append to lists if present
            if 'eeg' in multi_channel_data:
                all_eeg.append(multi_channel_data['eeg'])
            if 'eog' in multi_channel_data:
                all_eog.append(multi_channel_data['eog'])
            if 'emg' in multi_channel_data:
                all_emg.append(multi_channel_data['emg'])
            if 'other' in multi_channel_data:
                all_other.append(multi_channel_data['other'])

            # safe label pad/trim
            n_epochs = next(iter(multi_channel_data.values())).shape[0]
            labels = _safe_pad_labels(labels, n_epochs)
            all_labels.append(labels)

            all_record_ids.extend([record_id] * len(labels))

            # set channel_info if first
            if channel_info is None:
                channel_info = info

        except Exception as e:
            print(f"ERROR loading {record_id}: {type(e).__name__}: {e}")
            continue

    combined = {}
    if all_eeg:
        combined['eeg'] = np.concatenate(all_eeg, axis=0)
        print(f"Combined EEG shape: {combined['eeg'].shape}")
    if all_eog:
        combined['eog'] = np.concatenate(all_eog, axis=0)
        print(f"Combined EOG shape: {combined['eog'].shape}")
    if all_emg:
        combined['emg'] = np.concatenate(all_emg, axis=0)
        print(f"Combined EMG shape: {combined['emg'].shape}")
    if all_other:
        combined['other'] = np.concatenate(all_other, axis=0)
        print(f"Combined OTHER shape: {combined['other'].shape}")

    combined_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=int)
    combined_record_ids = np.array(all_record_ids)

    print(f"\nTotal loaded: {len(combined_labels)} epochs from {len(np.unique(combined_record_ids))} recordings")
    if channel_info:
        print(f"Channel info keys: {sorted(channel_info.keys())}")
    _print_label_distribution(combined_labels)

    return combined, combined_labels, combined_record_ids, channel_info

# -------------------------
# Backwards-compatible single-recording loader (kept for legacy)
# -------------------------
def load_single_recording(edf_file_path, xml_file_path, epoch_length=30):
    """
    Legacy helper kept for compatibility (iteration 1 style).
    Prefer load_training_data for iteration >=4.
    """
    # Simple wrapper that extracts EEG only but uses robust epoch computation
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
    rec_info = compute_recording_info(raw, epoch_length)
    eeg_channels, eog_channels, emg_channels, other_channels = identify_channels(raw.ch_names)

    if not eeg_channels:
        raise ValueError("No EEG channels found in file.")

    selected = eeg_channels[:2]
    eeg_raw = raw.copy().pick_channels(selected)
    epochs, fs = _extract_epochs_from_raw(eeg_raw, epoch_length, rec_info['n_epochs'])

    # Parse labels
    parsed_xml = parse_xml_annotations(xml_file_path)
    labels = create_epoch_labels(parsed_xml.get('stages', []), rec_info['duration'], epoch_length)
    labels = _safe_pad_labels(labels, rec_info['n_epochs'])

    print(f"Legacy load_single_recording: selected={selected}, fs={fs}, epochs={epochs.shape[0]}")
    return epochs, labels

# -------------------------
# Utilities
# -------------------------
def _print_label_distribution(labels):
    if labels.size == 0:
        print("No labels to show.")
        return
    unique, counts = np.unique(labels, return_counts=True)
    stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    print("Label distribution:")
    for u, c in zip(unique, counts):
        name = stages[u] if u < len(stages) else str(u)
        pct = (c / labels.size) * 100
        print(f"  {name}: {c} ({pct:.1f}%)")

# -------------------------
# Example: CLI usage
# -------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2:
        edf_file = sys.argv[1]; xml_file = sys.argv[2]
        data, labels, info = load_training_data(edf_file, xml_file, epoch_length=30, target_fs=125)
        print("\nSummary:")
        print(f"  Total epochs: {labels.shape[0]}")
        for k, v in data.items():
            print(f"  {k.upper()} shape: {v.shape}")
    else:
        print("Usage: python data_loader_advanced.py <edf_file> <xml_file>")
