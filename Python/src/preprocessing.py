"""
EEG/EOG/EMG preprocessing script for Iteration 3
- Implements filtering, notch, EOG regression, EMG-adaptive low-pass.
- Example usage in __main__ generates synthetic data and runs preprocess().

Dependencies:
- numpy
- scipy
- scikit-learn

Save as `eeg_preprocessing_iter3.py` and run with Python 3.8+.
"""

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from scipy.signal import butter, lfilter, filtfilt, iirnotch
from sklearn.linear_model import LinearRegression


# ------------------------- Utilities / Filters -------------------------

def lowpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 2) -> np.ndarray:
    """Simple low-pass Butterworth filter (forward-only using lfilter).

    This is a convenience function used for e.g. EMG preprocessing where
    zero-phase filtering is not strictly necessary.
    """
    nyquist = 0.5 * fs
    if cutoff >= nyquist:
        cutoff = 0.9 * nyquist
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, data)


def butter_highpass(cutoff: float, fs: int, order: int = 2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high")
    return b, a


def butter_lowpass(cutoff: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low")
    return b, a


def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def apply_notch(signal: np.ndarray, f0: float, fs: int, Q: float = 30.0) -> np.ndarray:
    """Apply an IIR notch filter (zero-phase via filtfilt).
    If the requested notch frequency is above Nyquist it will be skipped.
    """
    if f0 >= 0.5 * fs:
        return signal
    b, a = iirnotch(w0=f0 / (fs / 2.0), Q=Q)
    return filtfilt(b, a, signal)


# ------------------------- Artifact & metrics helpers -------------------------

def band_power(signal: np.ndarray, fs: int, low: float, high: float, order: int = 2) -> float:
    b, a = butter_bandpass(low, high, fs, order=order)
    filt_sig = filtfilt(b, a, signal)
    return np.mean(filt_sig ** 2)


def remove_eog_artifacts_from_eeg(eeg_epochs: np.ndarray, eog_epochs: np.ndarray) -> np.ndarray:
    """Remove EOG artifacts from EEG epochs using linear regression per epoch.

    Shapes:
        eeg_epochs: (n_epochs, n_eeg_channels, n_samples)
        eog_epochs: (n_epochs, n_eog_channels, n_samples)

    Returns cleaned EEG of same shape as eeg_epochs.
    """
    n_epochs, n_eeg_ch, n_samples = eeg_epochs.shape
    _, n_eog_ch, n_samples_eog = eog_epochs.shape
    if n_samples != n_samples_eog:
        raise ValueError(
            f"EOG/EEG mismatch in samples: EEG {n_samples}, EOG {n_samples_eog}."
            " Ensure same epoch length after resampling."
        )

    cleaned_eeg = eeg_epochs.copy()
    lr = LinearRegression()

    for ep in range(n_epochs):
        X = eog_epochs[ep].T  # (n_samples, n_eog_ch)
        if np.allclose(X, 0):
            continue
        for ch in range(n_eeg_ch):
            y = eeg_epochs[ep, ch, :]
            lr.fit(X, y)
            y_hat = lr.predict(X)
            residual = y - y_hat
            cleaned_eeg[ep, ch, :] = residual

    return cleaned_eeg


def apply_emg_adaptive_filtering(eeg_epochs: np.ndarray, emg_epochs: np.ndarray, eeg_fs: int, emg_fs: int) -> np.ndarray:
    """Adaptive low-pass EEG filtering driven by EMG 20–40Hz power.

    - Compute EMG power in 20–40 Hz per epoch.
    - Threshold at 75th percentile.
    - For epochs above threshold, apply a stronger low-pass (e.g. 20 Hz) to EEG.
    """
    n_epochs, n_eeg_ch, n_eeg_samples = eeg_epochs.shape
    n_epochs_emg, n_emg_ch, n_emg_samples = emg_epochs.shape
    if n_emg_ch != 1:
        raise ValueError("Expected exactly 1 EMG channel (shape: n_epochs, 1, samples).")

    n_common = min(n_epochs, n_epochs_emg)
    emg_powers = np.zeros(n_common)

    for ep in range(n_common):
        emg_sig = emg_epochs[ep, 0, :]
        emg_powers[ep] = band_power(emg_sig, emg_fs, 20.0, 40.0, order=2)

    threshold = np.percentile(emg_powers, 75)
    print(f"EMG 20–40 Hz power threshold (75th percentile): {threshold:.4e}")

    strong_lp_cut = 20.0
    b_strong_lp, a_strong_lp = butter_lowpass(strong_lp_cut, eeg_fs, order=4)
    eeg_filtered = eeg_epochs.copy()

    for ep in range(n_common):
        if emg_powers[ep] > threshold:
            for ch in range(n_eeg_ch):
                x = eeg_filtered[ep, ch, :]
                eeg_filtered[ep, ch, :] = filtfilt(b_strong_lp, a_strong_lp, x)

    return eeg_filtered


# ------------------------- Preprocessing pipeline -------------------------


@dataclass
class Config:
    CURRENT_ITERATION: int = 3
    LOW_PASS_FILTER_FREQ: float = 40.0


def preprocess(data: Any, config: Config, channel_info: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
    print(f"Preprocessing data for iteration {config.CURRENT_ITERATION}...")
    is_multi_channel = isinstance(data, dict) and "eeg" in data
    if is_multi_channel:
        return preprocess_multi_channel(data, config, channel_info=channel_info)
    else:
        return {"single": preprocess_single_channel(data, config)}


def preprocess_multi_channel(multi_channel_data: Dict[str, np.ndarray], config: Config, channel_info: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
    preprocessed_data: Dict[str, np.ndarray] = {}

    # ---------- EEG ----------
    eeg_data = multi_channel_data["eeg"]
    eeg_fs = (channel_info.get("eeg_fs") if channel_info and "eeg_fs" in channel_info else 125)
    hp_cut = 0.3
    lp_cut = 40.0
    notch_f = 50.0

    b_hp, a_hp = butter_highpass(hp_cut, eeg_fs, order=2)
    b_lp, a_lp = butter_lowpass(lp_cut, eeg_fs, order=4)

    n_epochs, n_channels, samples_per_epoch = eeg_data.shape
    continuous_eeg = eeg_data.transpose(1, 0, 2).reshape(n_channels, -1)
    preprocessed_continuous_eeg = np.zeros_like(continuous_eeg)

    for ch in range(n_channels):
        x = continuous_eeg[ch, :].copy()
        x = filtfilt(b_hp, a_hp, x)
        x = apply_notch(x, notch_f, eeg_fs, Q=30)
        if 100.0 < 0.5 * eeg_fs:
            x = apply_notch(x, 100.0, eeg_fs, Q=30)
        x = filtfilt(b_lp, a_lp, x)
        preprocessed_continuous_eeg[ch, :] = x

    preprocessed_eeg = preprocessed_continuous_eeg.reshape(n_channels, n_epochs, samples_per_epoch).transpose(1, 0, 2)
    preprocessed_data["eeg"] = preprocessed_eeg

    # ---------- EOG ----------
    if config.CURRENT_ITERATION >= 2 and "eog" in multi_channel_data:
        eog_data = multi_channel_data["eog"]
        eog_fs = (channel_info.get("eog_fs") if channel_info and "eog_fs" in channel_info else 50)
        eog_hp_cut = 0.5
        eog_lp_cut = 30.0
        eog_notch_f = 50.0

        b_eog_hp, a_eog_hp = butter_highpass(eog_hp_cut, eog_fs, order=2)
        b_eog_lp, a_eog_lp = butter_lowpass(eog_lp_cut, eog_fs, order=4)

        n_eog_epochs, n_eog_channels, eog_samples_per_epoch = eog_data.shape
        continuous_eog = eog_data.transpose(1, 0, 2).reshape(n_eog_channels, -1)
        preprocessed_continuous_eog = np.zeros_like(continuous_eog)

        for ch in range(n_eog_channels):
            x = continuous_eog[ch, :].copy()
            x = filtfilt(b_eog_hp, a_eog_hp, x)
            x = apply_notch(x, eog_notch_f, eog_fs, Q=30)
            if 100.0 < 0.5 * eog_fs:
                x = apply_notch(x, 100.0, eog_fs, Q=30)
            x = filtfilt(b_eog_lp, a_eog_lp, x)
            preprocessed_continuous_eog[ch, :] = x

        preprocessed_eog = preprocessed_continuous_eog.reshape(n_eog_channels, n_eog_epochs, eog_samples_per_epoch).transpose(1, 0, 2)
        preprocessed_data["eog"] = preprocessed_eog

    # ---------- EMG ----------
    if config.CURRENT_ITERATION >= 3 and "emg" in multi_channel_data:
        emg_data = multi_channel_data["emg"]
        emg_fs = (channel_info.get("emg_fs") if channel_info and "emg_fs" in channel_info else 125)
        preprocessed_emg = np.zeros_like(emg_data)
        for ep in range(emg_data.shape[0]):
            signal = emg_data[ep, 0, :]
            filtered_signal = lowpass_filter(signal, 70, emg_fs)
            preprocessed_emg[ep, 0, :] = filtered_signal
        preprocessed_data["emg"] = preprocessed_emg

    # Iteration 3 artifact handling
    if config.CURRENT_ITERATION >= 3:
        if "eog" in preprocessed_data:
            print("Iteration 3: Removing EOG artifacts from EEG using LinearRegression...")
            preprocessed_data["eeg"] = remove_eog_artifacts_from_eeg(preprocessed_data["eeg"], preprocessed_data["eog"])  # noqa: E501
        if "emg" in preprocessed_data:
            print("Iteration 3: Applying EMG-based adaptive low-pass filtering on EEG...")
            preprocessed_data["eeg"] = apply_emg_adaptive_filtering(preprocessed_data["eeg"], preprocessed_data["emg"], eeg_fs=eeg_fs, emg_fs=(channel_info.get("emg_fs") if channel_info and "emg_fs" in channel_info else 125))

    return preprocessed_data


def preprocess_single_channel(data: np.ndarray, config: Config) -> np.ndarray:
    if config.CURRENT_ITERATION == 1:
        fs = 125
        return lowpass_filter(data, config.LOW_PASS_FILTER_FREQ, fs)
    elif config.CURRENT_ITERATION == 2:
        print("TODO: Implement enhanced preprocessing for iteration 2")
        return data
    elif config.CURRENT_ITERATION >= 3:
        print("TODO: Use multi-channel data format for iteration 3+")
        return data
    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")


# ------------------------- Example usage -------------------------


if __name__ == "__main__":
    # Create a small synthetic dataset to demonstrate pipeline execution.
    np.random.seed(42)

    n_epochs = 10
    n_eeg_ch = 2
    n_samples = 1250  # e.g. 10 seconds at 125 Hz

    eeg = 1e-6 * np.random.randn(n_epochs, n_eeg_ch, n_samples)
    eog = 1e-5 * np.random.randn(n_epochs, 2, n_samples // 2)  # assume lower sample rate for EOG
    emg = 1e-4 * np.random.randn(n_epochs, 1, n_samples)

    data = {
        "eeg": eeg,
        "eog": eog,
        "emg": emg,
    }

    cfg = Config(CURRENT_ITERATION=3, LOW_PASS_FILTER_FREQ=40.0)
    channel_info = {"eeg_fs": 125, "eog_fs": 50, "emg_fs": 125}

    out = preprocess(data, cfg, channel_info=channel_info)
    print("Preprocessing finished. Output keys:", list(out.keys()))
