# ======================================================================
#                      FEATURE EXTRACTION (ITERATION 3)
# ======================================================================

import numpy as np
import scipy.stats  # Used for calculating skewness and kurtosis
from scipy.signal import welch, butter, filtfilt, find_peaks
from scipy.linalg import toeplitz

try:
    import nolds  # Sample Entropy package (optional)
except ImportError:
    nolds = None


# ======================================================================
#                       GLOBAL SETTINGS
# ======================================================================

ENABLE_SAMPLE_ENTROPY = False  # Toggle ON to enable SampEn


# ======================================================================
#                       SECTION 1: TIME-DOMAIN FEATURES (EEG)
# ======================================================================

def extract_time_domain_features(epoch: np.ndarray) -> dict:
    """
    Extract the 16 required time-domain features.
    """
    # --- Statistical (6)
    mean_val = np.mean(epoch)
    median_val = np.median(epoch)
    std_val = np.std(epoch)
    var_val = np.var(epoch)
    skew_val = scipy.stats.skew(epoch)
    kurt_val = scipy.stats.kurtosis(epoch)

    # --- Amplitude (4)
    rms_val = np.sqrt(np.mean(epoch ** 2))
    min_val = np.min(epoch)
    max_val = np.max(epoch)
    range_val = max_val - min_val

    # --- Hjorth (3)
    diff_sig = np.diff(epoch)
    hjorth_activity = var_val

    if var_val > 0:
        hjorth_mobility = np.sqrt(np.var(diff_sig) / var_val)
    else:
        hjorth_mobility = 0.0

    var_diff = np.var(diff_sig)
    if var_diff > 0 and hjorth_mobility > 0:
        mobility_diff = np.sqrt(np.var(np.diff(diff_sig)) / var_diff)
        hjorth_complexity = mobility_diff / hjorth_mobility
    else:
        hjorth_complexity = 0.0

    # --- Frequency-related (2)
    zero_crossings = np.sum(np.diff(np.sign(epoch)) != 0)
    total_energy = np.sum(epoch ** 2)

    # --- Complexity (1)
    sample_entropy_val = compute_sample_entropy(epoch)

    features = {
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'variance': var_val,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'rms': rms_val,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'hjorth_activity': hjorth_activity,
        'hjorth_mobility': hjorth_mobility,
        'hjorth_complexity': hjorth_complexity,
        'zero_crossings': zero_crossings,
        'total_energy': total_energy,
        'sample_entropy': sample_entropy_val,
    }

    return features


def compute_sample_entropy(epoch, m=2, r_ratio=0.2, max_entropy_length=1000):
    """
    Computes Sample Entropy using optional downsampling.
    """
    if not ENABLE_SAMPLE_ENTROPY:
        return 0.0

    if nolds is None:
        print("WARNING: nolds not installed. SampEn disabled.")
        return 0.0

    std_val = np.std(epoch)
    if std_val == 0 or len(epoch) < m + 2:
        return 0.0

    # Downsample long signals
    if len(epoch) > max_entropy_length:
        step = max(1, len(epoch) // max_entropy_length)
        epoch = epoch[::step]

    try:
        tolerance = r_ratio * np.std(epoch)
        return float(nolds.sampen(epoch, emb_dim=m, tolerance=tolerance))
    except Exception:
        return 0.0


# ======================================================================
#                SECTION 2: AR & WELCH SPECTRAL FEATURES (EEG)
# ======================================================================

# Sleep EEG bands according to AASM (sigma lagt lite överlappande)
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (11.0, 16.0),
    "beta":  (13.0, 30.0),
}


def _estimate_fs_from_epoch(epoch: np.ndarray, epoch_length_sec: float = 30.0) -> float:
    """Infer sampling frequency from epoch length and number of samples."""
    n_samples = len(epoch)
    if epoch_length_sec <= 0:
        raise ValueError("epoch_length_sec must be positive")
    return n_samples / float(epoch_length_sec)


def _bandpower(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    """Integrate PSD over a frequency band [fmin, fmax)."""
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _spectral_entropy(freqs: np.ndarray, psd: np.ndarray,
                      fmin: float = 0.5, fmax: float = 40.0) -> float:
    """Spectral entropy of PSD in a given band."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    band_psd = psd[mask]
    total = np.sum(band_psd)
    if total <= 0:
        return 0.0
    p = band_psd / total
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def _spectral_edge_frequency(freqs: np.ndarray, psd: np.ndarray,
                             edge: float = 0.95,
                             fmin: float = 0.5,
                             fmax: float = 40.0) -> float:
    """
    Spectral edge frequency: frequency below which `edge` (e.g. 0.95) of power lies.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    band_freqs = freqs[mask]
    band_psd = psd[mask]
    if band_freqs.size == 0:
        return 0.0

    total = np.trapz(band_psd, band_freqs)
    if total <= 0:
        return 0.0

    # numerisk integral genom trapezoid-summa
    cumsum = np.cumsum((band_psd[:-1] + band_psd[1:]) / 2.0 * np.diff(band_freqs))
    target = edge * total
    idx = np.searchsorted(cumsum, target)
    idx = min(idx, len(band_freqs) - 1)
    return float(band_freqs[idx])


def _peak_frequency(freqs: np.ndarray, psd: np.ndarray,
                    fmin: float = 0.5,
                    fmax: float = 40.0) -> float:
    """Dominant frequency (max PSD) in [fmin, fmax]."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    sub_freqs = freqs[mask]
    sub_psd = psd[mask]
    idx = int(np.argmax(sub_psd))
    return float(sub_freqs[idx])


def _compute_welch_psd(epoch: np.ndarray, fs: float):
    """
    Compute PSD using Welch's method with reasonable defaults for sleep EEG.
    """
    epoch = np.asarray(epoch, dtype=float)
    if fs is None or fs <= 0:
        fs = _estimate_fs_from_epoch(epoch)

    # 4-second windows with 50% overlap (common in sleep EEG)
    n_samples = len(epoch)
    nperseg = int(min(n_samples, max(fs * 4.0, fs)))  # at least 1 s, at most full epoch
    if nperseg < 8:  # very short fallback
        nperseg = n_samples
    noverlap = nperseg // 2

    freqs, psd = welch(
        epoch,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
    )
    return freqs, psd


def _compute_ar_psd(epoch: np.ndarray,
                    fs: float,
                    order: int = 16,
                    n_fft: int = 512):
    """
    Simple AR-based PSD using Yule-Walker equations.
    """
    from numpy.linalg import LinAlgError

    x = np.asarray(epoch, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if fs is None or fs <= 0:
        fs = _estimate_fs_from_epoch(x)

    # Ensure order is reasonable
    max_order = max(4, min(32, n // 4))
    order = int(min(order, max_order))

    # Autocorrelation (biased)
    r = np.correlate(x, x, mode="full")
    mid = len(r) // 2
    r = r[mid:mid + order + 1] / float(n)

    # Toeplitz system R a = r[1:]
    R = toeplitz(r[:-1])
    rhs = r[1:]
    try:
        a = np.linalg.solve(R, rhs)
    except LinAlgError:
        # Fallback: white-ish spectrum
        freqs = np.linspace(0, fs / 2.0, n_fft // 2 + 1)
        psd = np.ones_like(freqs) * np.var(x)
        return freqs, psd

    # Noise variance
    noise_var = r[0] - np.dot(a, rhs)
    noise_var = max(noise_var, 1e-12)

    # Frequency grid
    freqs = np.linspace(0, fs / 2.0, n_fft // 2 + 1)
    w = 2.0 * np.pi * freqs / fs

    # AR transfer function H(e^jw) = 1 / (1 - sum a_k e^{-jwk})
    denom = np.ones_like(freqs, dtype=complex)
    for k in range(1, order + 1):
        denom -= a[k - 1] * np.exp(-1j * w * k)

    psd = noise_var / np.abs(denom) ** 2
    return freqs, np.real(psd)


def extract_welch_features(epoch: np.ndarray, fs: float = None) -> dict:
    """
    Welch PSD features for a single epoch:
    - Absolute and relative band powers (delta, theta, alpha, sigma, beta)
    - Total power (0.5–40 Hz)
    - Spectral entropy
    - Peak frequency
    - Spectral edge frequency (95%)
    """
    if fs is None or fs <= 0:
        fs = _estimate_fs_from_epoch(epoch)

    freqs, psd = _compute_welch_psd(epoch, fs)

    # Total power in 0.5–40 Hz
    total_power = _bandpower(freqs, psd, 0.5, 40.0)
    features = {"welch_total_power": total_power}

    band_powers = {}
    for band_name, (fmin, fmax) in EEG_BANDS.items():
        p = _bandpower(freqs, psd, fmin, fmax)
        band_powers[band_name] = p
        features[f"welch_{band_name}_power"] = p

    # Relative band powers
    denom = total_power if total_power > 0 else (np.sum(list(band_powers.values())) + 1e-12)
    for band_name, p in band_powers.items():
        features[f"welch_{band_name}_rel_power"] = p / denom

    # Spectral entropy & shape measures
    features["welch_spectral_entropy"] = _spectral_entropy(freqs, psd)
    features["welch_peak_freq"] = _peak_frequency(freqs, psd)
    features["welch_spectral_edge_95"] = _spectral_edge_frequency(freqs, psd, edge=0.95)

    return features


def extract_ar_spectral_features(epoch: np.ndarray,
                                 fs: float = None,
                                 order: int = 16) -> dict:
    """
    AR-based spectral features for a single epoch:
    - Absolute and relative band powers (delta, theta, alpha, sigma, beta)
    - Total power (0.5–40 Hz)
    - Spectral entropy
    - Peak frequency
    - Spectral edge frequency (95%)
    """
    if fs is None or fs <= 0:
        fs = _estimate_fs_from_epoch(epoch)

    freqs, psd = _compute_ar_psd(epoch, fs, order=order)

    total_power = _bandpower(freqs, psd, 0.5, 40.0)
    features = {"ar_total_power": total_power}

    band_powers = {}
    for band_name, (fmin, fmax) in EEG_BANDS.items():
        p = _bandpower(freqs, psd, fmin, fmax)
        band_powers[band_name] = p
        features[f"ar_{band_name}_power"] = p

    denom = total_power if total_power > 0 else (np.sum(list(band_powers.values())) + 1e-12)
    for band_name, p in band_powers.items():
        features[f"ar_{band_name}_rel_power"] = p / denom

    features["ar_spectral_entropy"] = _spectral_entropy(freqs, psd)
    features["ar_peak_freq"] = _peak_frequency(freqs, psd)
    features["ar_spectral_edge_95"] = _spectral_edge_frequency(freqs, psd, edge=0.95)

    return features


# ======================================================================
#      SECTION 3: (Placeholder) WAVELET FEATURES
# ======================================================================

def extract_wavelet_features(epoch, fs):
    """
    Placeholder for Wavelet features.
    Implemented in next step.
    """
    return {}  # TEMPORARY


# ======================================================================
#      SECTION 4: HELPER FOR EOG/EMG (ITERATION 3)
# ======================================================================

def _butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high")
    return b, a


def _apply_highpass(x, cutoff, fs, order=2):
    if len(x) < 10:
        return x
    b, a = _butter_highpass(cutoff, fs, order)
    return filtfilt(b, a, x)


def _compute_band_power_fft(signal, fs, fmin, fmax):
    """
    Enkel bandkraft via FFT (används för EMG 20–40 Hz power ratio).
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0:
        return 0.0

    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / fs)
    power = np.abs(fft_vals) ** 2

    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0

    return float(np.sum(power[mask]))


# ======================================================================
#       SECTION 5: SINGLE-CHANNEL FEATURE EXTRACTION (legacy)
# ======================================================================

def extract_single_channel_features(data, config):
    """
    Backward compatibility for old single-channel data (mainly used in tests).

    Iteration 2: time + Welch + AR.
    Iteration 3+: we expect multi-channel format, so this is just a stub.
    """
    if config.CURRENT_ITERATION == 1:
        all_features = []
        for epoch in data:
            feats = extract_time_domain_features(epoch)
            all_features.append(list(feats.values()))
        features = np.array(all_features)
        print(f"{features.shape[1]} features extracted (Target: 16)")

    elif config.CURRENT_ITERATION == 2:
        all_features = []
        for epoch in data:
            # Infer fs from epoch length (assumed 30 s)
            fs = len(epoch) / 30.0

            time_feats = extract_time_domain_features(epoch)
            welch_feats = extract_welch_features(epoch, fs=fs)
            ar_feats = extract_ar_spectral_features(epoch, fs=fs)

            feature_vector = (
                list(time_feats.values())
                + list(welch_feats.values())
                + list(ar_feats.values())
            )
            all_features.append(feature_vector)

        features = np.array(all_features)
        print(
            f"Single-channel Iteration 2: {features.shape[1]} features/epoch "
            f"(time + Welch + AR)"
        )

    elif config.CURRENT_ITERATION >= 3:
        print("Iteration 3+: Use multi-channel data format (EEG + EOG + EMG).")
        n_epochs = data.shape[0] if len(data.shape) > 1 else 1
        features = np.zeros((n_epochs, 0))

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return features


# ======================================================================
#      SECTION 6: MULTI-CHANNEL FEATURE EXTRACTION (Iteration 2 & 3)
# ======================================================================

def extract_multi_channel_features(multi_channel_data, config):
    """
    Extract features for multi-channel input.

    EEG:
        - 16 time-domain features
        - Welch spectral features
        - AR spectral features

    Iteration 2:
        EEG + EOG (simpla EOG-features)
    Iteration 3:
        EEG (samma som iteration 2)
        EOG: ~6 features/kanal med REM-detektionsscore
        EMG: 3–4 features per kanal (power, var, std, 20–40 Hz ratio)

    Returns:
        features: (n_epochs, n_features)
    """
    # ------------- EEG -------------
    eeg = multi_channel_data["eeg"]  # (n_epochs, eeg_channels, samples)
    n_epochs, eeg_channels, eeg_samples = eeg.shape
    eeg_fs = eeg_samples / 30.0  # 30 s epoch

    # ------------- EOG -------------
    use_eog = ("eog" in multi_channel_data) and (config.CURRENT_ITERATION >= 2)
    if use_eog:
        eog = multi_channel_data["eog"]
        _, eog_channels, eog_samples = eog.shape
        eog_fs = eog_samples / 30.0
    else:
        eog_channels = 0
        eog_fs = None

    # ------------- EMG -------------
    use_emg = ("emg" in multi_channel_data) and (config.CURRENT_ITERATION >= 3)
    if use_emg:
        emg = multi_channel_data["emg"]  # (n_epochs, 1, samples)
        _, emg_channels, emg_samples = emg.shape
        emg_fs = emg_samples / 30.0
    else:
        emg_channels = 0
        emg_fs = None

    all_feature_vectors = []

    # ============================
    # LOOP OVER ALL EPOCHS
    # ============================
    for i in range(n_epochs):
        epoch_features = []

        # ============================================
        # 1. EEG FEATURES (Iteration 1 + spectral)
        # ============================================
        for ch in range(eeg_channels):
            signal = eeg[i, ch, :]

            # --- Time-domain ---
            t_feats = extract_time_domain_features(signal)

            # --- Welch PSD features ---
            w_feats = extract_welch_features(signal, fs=eeg_fs)

            # --- AR PSD features ---
            ar_feats = extract_ar_spectral_features(signal, fs=eeg_fs)

            epoch_features.extend(list(t_feats.values()))
            epoch_features.extend(list(w_feats.values()))
            epoch_features.extend(list(ar_feats.values()))

        # ============================================
        # 2. EOG FEATURES (Iteration 2 & 3)
        # ============================================
        if use_eog:
            for ch in range(eog_channels):
                eog_signal = eog[i, ch, :]
                eog_feats = extract_eog_features(eog_signal, fs=eog_fs)
                epoch_features.extend(list(eog_feats.values()))

            # EOG cross-channel correlation (vänster/höger)
            if eog_channels >= 2:
                eog_left = eog[i, 0, :]
                eog_right = eog[i, 1, :]
                if np.std(eog_left) > 0 and np.std(eog_right) > 0:
                    corr = float(np.corrcoef(eog_left, eog_right)[0, 1])
                else:
                    corr = 0.0
                epoch_features.append(corr)

        # ============================================
        # 3. EMG FEATURES (Iteration 3)
        # ============================================
        if use_emg and emg_channels >= 1:
            # antar 1 EMG-kanal
            emg_signal = emg[i, 0, :]
            emg_feats = extract_emg_features(emg_signal, fs=emg_fs)
            epoch_features.extend(list(emg_feats.values()))

        # Spara epoch-vektorn
        all_feature_vectors.append(epoch_features)

    features = np.array(all_feature_vectors)
    print(f"[DEBUG] Multi-channel Iteration {config.CURRENT_ITERATION}: {features.shape} features extracted.")

    return features


# ======================================================================
#                          SECTION 7: EOG FEATURES (Iteration 3)
# ======================================================================

def extract_eog_features(eog_signal, fs):
    """
    EOG Features (~6 features per channel), fokuserade på REM:

    - eog_peak_amp: max |x|
    - eog_variance
    - eog_zero_crossings
    - eog_rem_peak_count: antal snabba deflektioner i high-passat (>0.5 Hz) EOG
    - eog_rem_peak_rate: peaks per sekund
    - eog_movement_energy: sum(diff(x)^2)
    """
    eog_signal = np.asarray(eog_signal, dtype=float).ravel()
    n = len(eog_signal)

    if n == 0 or fs is None or fs <= 0:
        return {
            "eog_peak_amp": 0.0,
            "eog_variance": 0.0,
            "eog_zero_crossings": 0.0,
            "eog_rem_peak_count": 0.0,
            "eog_rem_peak_rate": 0.0,
            "eog_movement_energy": 0.0,
        }

    duration_sec = n / fs

    peak_amp = float(np.max(np.abs(eog_signal)))
    variance = float(np.var(eog_signal))
    zero_crossings = float(np.sum(np.diff(np.sign(eog_signal)) != 0))

    # High-pass > 0.5 Hz
    hp_sig = _apply_highpass(eog_signal, cutoff=0.5, fs=fs, order=2)
    hp_std = np.std(hp_sig)
    if hp_std == 0:
        rem_peak_count = 0
        rem_peak_rate = 0.0
    else:
        # tröskel 1.5 * std, min distance ~100 ms
        threshold = 1.5 * hp_std
        peaks, _ = find_peaks(
            np.abs(hp_sig),
            height=threshold,
            distance=int(0.1 * fs),
        )
        rem_peak_count = int(len(peaks))
        rem_peak_rate = float(rem_peak_count / duration_sec) if duration_sec > 0 else 0.0

    diff_sig = np.diff(eog_signal)
    movement_energy = float(np.sum(diff_sig ** 2))

    return {
        "eog_peak_amp": peak_amp,
        "eog_variance": variance,
        "eog_zero_crossings": zero_crossings,
        "eog_rem_peak_count": float(rem_peak_count),
        "eog_rem_peak_rate": rem_peak_rate,
        "eog_movement_energy": movement_energy,
    }


# ======================================================================
#                          SECTION 8: EMG FEATURES (Iteration 3)
# ======================================================================

def extract_emg_features(emg_signal, fs):
    """
    EMG Features (~3–4 features per channel):

    - emg_power: mean squared amplitude (muskeltonus)
    - emg_variance
    - emg_std
    - emg_hf_power_ratio_20_40: power(20–40 Hz) / power(0–40 Hz)
    """
    emg_signal = np.asarray(emg_signal, dtype=float).ravel()
    if len(emg_signal) == 0 or fs is None or fs <= 0:
        return {
            "emg_power": 0.0,
            "emg_variance": 0.0,
            "emg_std": 0.0,
            "emg_hf_power_ratio_20_40": 0.0,
        }

    power = float(np.mean(emg_signal ** 2))
    variance = float(np.var(emg_signal))
    std = float(np.sqrt(variance))

    band_20_40 = _compute_band_power_fft(emg_signal, fs, 20.0, 40.0)
    band_0_40 = _compute_band_power_fft(emg_signal, fs, 0.0, 40.0)
    if band_0_40 > 0:
        hf_ratio = float(band_20_40 / band_0_40)
    else:
        hf_ratio = 0.0

    return {
        "emg_power": power,
        "emg_variance": variance,
        "emg_std": std,
        "emg_hf_power_ratio_20_40": hf_ratio,
    }


# ======================================================================
#                          SECTION 9: DISPATCHER
# ======================================================================

def extract_features(data, config):
    """
    Unified feature extractor entry point.
    """
    print(f"Extracting features for iteration {config.CURRENT_ITERATION}...")

    if isinstance(data, dict) and "eeg" in data:
        return extract_multi_channel_features(data, config)
    else:
        return extract_single_channel_features(data, config)
