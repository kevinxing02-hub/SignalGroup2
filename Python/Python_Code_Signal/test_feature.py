#!/usr/bin/env python3
"""
Multi-Channel Feature Extraction Example (robust, iteration-ready)

Usage:
    python multichannel_feature_example.py

Key improvements over the original:
- Uses PROJECT_ROOT and config.TRAINING_DIR / SAMPLE_DIR to find files.
- Safe handling when some channel groups are missing.
- Corrects FFT bandpower calculation to use positive-frequency rfft.
- Fixes stage label mapping (default: 0=Wake,1=N1,2=N2,3=N3,4=REM) — adjust if your labels differ.
- Produces plots with correct time axes and saves them under outputs/.
"""

from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- project-relative import: assume script in project/Python or project/Python/tests ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust if you put script elsewhere
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config  # project's config.py
from data_loader import load_training_data

# -------------------------
# Utility helpers
# -------------------------
def safe_get(data: dict, key: str):
    """Return data[key] or None if missing."""
    return data.get(key, None)

def rfft_bandpower(sig: np.ndarray, fs: float, low: float, high: float) -> float:
    """Compute band power using rFFT (positive freqs only)."""
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0 or fs is None or fs <= 0:
        return 0.0
    # detrend mean to avoid DC leak
    sig = sig - np.mean(sig)
    fft_vals = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    power = np.abs(fft_vals) ** 2
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(power[mask], freqs[mask]))

def get_band_power_from_fft(power_spectrum, freqs, low_freq, high_freq):
    """Helper used by simpler demo: expects power_spectrum aligned with freqs (both full FFT or rfft)"""
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(power_spectrum[mask], freqs[mask]))

# -------------------------
# Feature extraction helpers (small, demo-level)
# -------------------------
def extract_eeg_features(eeg_data, fs):
    """
    eeg_data: (n_channels, samples)
    returns dict of features (flattened names)
    """
    features = {}
    if eeg_data is None:
        return features

    n_ch = eeg_data.shape[0]
    for ch in range(n_ch):
        signal = eeg_data[ch, :]
        ch_name = f"eeg_ch{ch+1}"

        features[f"{ch_name}_mean"] = float(np.mean(signal))
        features[f"{ch_name}_std"] = float(np.std(signal))
        features[f"{ch_name}_var"] = float(np.var(signal))
        features[f"{ch_name}_rms"] = float(np.sqrt(np.mean(signal**2)))
        features[f"{ch_name}_range"] = float(np.max(signal) - np.min(signal))

        # Use rfft for band powers (positive frequencies)
        delta_power = rfft_bandpower(signal, fs, 0.5, 4.0)
        theta_power = rfft_bandpower(signal, fs, 4.0, 8.0)
        alpha_power = rfft_bandpower(signal, fs, 8.0, 12.0)
        sigma_power = rfft_bandpower(signal, fs, 12.0, 15.0)
        beta_power = rfft_bandpower(signal, fs, 15.0, 30.0)

        features[f"{ch_name}_delta_power"] = delta_power
        features[f"{ch_name}_theta_power"] = theta_power
        features[f"{ch_name}_alpha_power"] = alpha_power
        features[f"{ch_name}_sigma_power"] = sigma_power
        features[f"{ch_name}_beta_power"] = beta_power

    return features

def extract_eog_features(eog_data, fs):
    features = {}
    if eog_data is None:
        return features
    n_ch = eog_data.shape[0]
    for ch in range(n_ch):
        sig = eog_data[ch, :]
        name = f"eog_ch{ch+1}"
        features[f"{name}_mean"] = float(np.mean(sig))
        features[f"{name}_std"] = float(np.std(sig))
        features[f"{name}_max_abs"] = float(np.max(np.abs(sig)))
        # movement energy
        diff = np.diff(sig)
        features[f"{name}_movement_energy"] = float(np.sum(diff**2))
    if n_ch >= 2:
        # correlation left/right
        try:
            corr = float(np.corrcoef(eog_data[0,:], eog_data[1,:])[0,1])
        except Exception:
            corr = 0.0
        features["eog_lr_corr"] = corr
    return features

def extract_emg_features(emg_data, fs):
    features = {}
    if emg_data is None:
        return features
    sig = emg_data[0,:]
    features["emg_mean"] = float(np.mean(sig))
    features["emg_std"] = float(np.std(sig))
    features["emg_rms"] = float(np.sqrt(np.mean(sig**2)))
    features["emg_energy"] = float(np.sum(sig**2))
    # HF power (20- min(250, fs/2))
    hf_high = min(250.0, fs/2.0 - 1e-6)
    if hf_high > 20.0:
        features["emg_hf_power"] = rfft_bandpower(sig, fs, 20.0, hf_high)
    else:
        features["emg_hf_power"] = 0.0
    return features

# -------------------------
# Visualization helpers
# -------------------------
def plot_epoch_multichannel(data, channel_info, epoch_num=0, save_dir=None, show=False):
    """Plot EEG/EOG/EMG for a single epoch and save or show."""
    eeg = safe_get(data, 'eeg')
    eog = safe_get(data, 'eog')
    emg = safe_get(data, 'emg')

    fs_eeg = float(channel_info.get('eeg_fs', channel_info.get('global_fs', 125.0)))
    fs_emg = float(channel_info.get('emg_fs', channel_info.get('global_fs', 125.0)))

    nplots = 0
    if eeg is not None:
        nplots += eeg.shape[1]
    if eog is not None:
        nplots += eog.shape[1]
    if emg is not None:
        nplots += emg.shape[1]

    if nplots == 0:
        print("No channels to plot.")
        return None

    fig, axes = plt.subplots(nplots, 1, figsize=(12, 2.2 * nplots), sharex=True)
    if nplots == 1:
        axes = [axes]

    axi = 0
    if eeg is not None:
        t = np.arange(eeg.shape[2]) / fs_eeg
        for ch in range(eeg.shape[1]):
            axes[axi].plot(t, eeg[epoch_num, ch, :], linewidth=0.6)
            name = channel_info.get('eeg_names', [f'EEG{ch}'])[ch]
            axes[axi].set_ylabel(f"{name}")
            axes[axi].grid(True, alpha=0.3)
            axi += 1

    if eog is not None:
        # use eeg timebase assuming same fs (resample otherwise)
        t = np.arange(eog.shape[2]) / fs_eeg if channel_info.get('eog_fs', fs_eeg)==fs_eeg else np.arange(eog.shape[2]) / float(channel_info.get('eog_fs', fs_eeg))
        for ch in range(eog.shape[1]):
            axes[axi].plot(t, eog[epoch_num, ch, :], linewidth=0.6)
            name = channel_info.get('eog_names', [f'EOG{ch}'])[ch]
            axes[axi].set_ylabel(f"{name}")
            axes[axi].grid(True, alpha=0.3)
            axi += 1

    if emg is not None:
        t = np.arange(emg.shape[2]) / fs_emg
        for ch in range(emg.shape[1]):
            axes[axi].plot(t, emg[epoch_num, ch, :], linewidth=0.6)
            name = channel_info.get('emg_names', [f'EMG{ch}'])[ch]
            axes[axi].set_ylabel(f"{name}")
            axes[axi].grid(True, alpha=0.3)
            axi += 1

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'Epoch {epoch_num} waveforms')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        p = os.path.join(save_dir, f'epoch_{epoch_num}_multichannel.png')
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print("Saved multichannel epoch plot:", p)
    if show:
        plt.show()
    plt.close()
    return True

def create_hypnogram(labels, save_dir=None, show=False):
    if labels is None or len(labels)==0:
        print("No labels for hypnogram.")
        return
    t_hours = np.arange(len(labels)) * 30.0 / 3600.0
    plt.figure(figsize=(12,3))
    plt.step(t_hours, labels, where='post')
    plt.ylim(-0.5, 4.5)
    plt.yticks([0,1,2,3,4], ['Wake','N1','N2','N3','REM'])
    plt.xlabel('Time (hours)')
    plt.ylabel('Stage')
    plt.title('Hypnogram')
    plt.grid(True, alpha=0.3)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        p = os.path.join(save_dir, 'hypnogram.png')
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print("Saved hypnogram:", p)
    if show:
        plt.show()
    plt.close()

# -------------------------
# Demo / main
# -------------------------
def main():
    print("=== MULTI-CHANNEL FEATURE EXTRACTION DEMO ===")

    # find EDF/XML from config; prefer SAMPLE_DIR then TRAINING_DIR
    sample_dir = Path(config.SAMPLE_DIR)
    training_dir = Path(config.TRAINING_DIR)
    candidate_dirs = [sample_dir, training_dir, Path(config.DATA_DIR)]

    edf_file = None
    xml_file = None
    for d in candidate_dirs:
        if not d:
            continue
        edf_cand = d / "R1.edf"
        xml_cand = d / "R1.xml"
        if edf_cand.exists() and xml_cand.exists():
            edf_file = str(edf_cand)
            xml_file = str(xml_cand)
            break

    if edf_file is None:
        print("No EDF/XML found in SAMPLE_DIR/TRAINING_DIR. Please update config paths.")
        return

    print(f"Using EDF: {edf_file}")
    print(f"Using XML: {xml_file}")

    multi_channel_data, labels, channel_info = load_training_data(edf_file, xml_file)

    print("\n--- Data summary ---")
    # safe prints
    print(f"Record ID: {channel_info.get('record_id', 'unknown')}")
    print(f"Duration (s): {channel_info.get('duration', 'N/A')}")
    print("Channels present:")
    for k in ['eeg','eog','emg','other']:
        arr = multi_channel_data.get(k, None)
        if arr is None:
            print(f"  {k.upper()}: MISSING")
        else:
            print(f"  {k.upper()}: shape={arr.shape}, fs={channel_info.get(f'{k}_fs', channel_info.get('global_fs'))}, names={channel_info.get(f'{k}_names', ['?'])}")

    # show example epoch
    epoch_example = min(100, multi_channel_data['eeg'].shape[0]-1)
    save_dir = os.path.join(config.OUTPUT_DIR, "multichannel_demo", channel_info.get('record_id','rec'))
    plot_epoch_multichannel(multi_channel_data, channel_info, epoch_num=epoch_example, save_dir=save_dir, show=False)
    create_hypnogram(labels, save_dir=save_dir, show=False)

    # demonstrate feature extraction for a single epoch
    print("\n--- Feature extraction for epoch", epoch_example, "---")
    eeg_epoch = multi_channel_data.get('eeg')
    eog_epoch = multi_channel_data.get('eog')
    emg_epoch = multi_channel_data.get('emg')

    # slice epoch (n_channels, samples)
    eeg_slice = eeg_epoch[epoch_example] if eeg_epoch is not None else None
    eog_slice = eog_epoch[epoch_example] if eog_epoch is not None else None
    emg_slice = emg_epoch[epoch_example] if emg_epoch is not None else None

    # read fs info (default fallback)
    eeg_fs = float(channel_info.get('eeg_fs', channel_info.get('global_fs', 125.0)))
    eog_fs = float(channel_info.get('eog_fs', channel_info.get('global_fs', eeg_fs)))
    emg_fs = float(channel_info.get('emg_fs', channel_info.get('global_fs', eeg_fs)))

    eeg_feats = extract_eeg_features(eeg_slice, eeg_fs)
    eog_feats = extract_eog_features(eog_slice, eog_fs)
    emg_feats = extract_emg_features(emg_slice, emg_fs)

    # combine
    combined = {**eeg_feats, **eog_feats, **emg_feats}
    print(f"Extracted {len(combined)} features. Sample keys (first 10):")
    for i, k in enumerate(sorted(combined.keys())[:10]):
        print(f"  {k}: {combined[k]:.4e}")

    print("\nDone. Visualizations & features saved under:", save_dir)

if __name__ == "__main__":
    main()
