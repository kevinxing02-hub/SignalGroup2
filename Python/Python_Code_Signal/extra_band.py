import numpy as np
from scipy.signal import welch, butter, filtfilt, hilbert
from typing import Tuple, Dict, Any

# ---------------------------
# Helper: bandpower via Welch
# ---------------------------
def bandpower_welch(x: np.ndarray, fs: float, low: float, high: float, nperseg: int = None) -> float:
    """
    Compute band power using Welch (integrated PSD).
    x: 1D signal (samples,)
    returns scalar power
    """
    if x.size == 0:
        return 0.0
    if nperseg is None:
        nperseg = min(1024, len(x))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return 0.0
    return np.trapz(Pxx[mask], f[mask])

# ---------------------------
# Helper: bandpass filter (zero-phase)
# ---------------------------
def bandpass_filter(sig: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    lown = max(low / nyq, 1e-6)
    highn = min(high / nyq, 0.999999)
    b, a = butter(order, [lown, highn], btype='band')
    try:
        y = filtfilt(b, a, sig)
    except Exception:
        # fallback: naive lfilter if filtfilt fails
        from scipy.signal import lfilter
        y = lfilter(b, a, sig)
    return y

# ---------------------------
# Spindle detection (envelope threshold + duration)
# ---------------------------
def detect_spindles_envelope(epoch_sig: np.ndarray,
                             fs: float,
                             sigma_band: Tuple[float,float] = (12.0, 15.0),
                             env_smooth_ms: float = 50.0,
                             thresh_std: float = 1.5,
                             min_duration_s: float = 0.5,
                             max_duration_s: float = 3.0) -> int:
    """
    Detect approximate spindles in one epoch signal using Hilbert envelope thresholding.
    Returns integer count of spindle events in the epoch.
    Steps:
      - bandpass 12-15 Hz
      - compute analytic signal -> envelope
      - optionally smooth envelope (moving average via convolution)
      - threshold = mean_envelope + thresh_std * std_envelope (or use percentile)
      - find contiguous supra-threshold segments and enforce duration constraints
    """
    # 1) bandpass
    env_sig = bandpass_filter(epoch_sig, fs, sigma_band[0], sigma_band[1], order=4)
    # 2) envelope via Hilbert
    analytic = hilbert(env_sig)
    envelope = np.abs(analytic)

    # smooth envelope (simple moving average)
    window_samples = max(1, int((env_smooth_ms / 1000.0) * fs))
    if window_samples > 1:
        kernel = np.ones(window_samples) / window_samples
        envelope = np.convolve(envelope, kernel, mode='same')

    # 3) threshold
    mean_env = np.mean(envelope)
    std_env = np.std(envelope)
    threshold = mean_env + thresh_std * std_env

    # 4) find supra-threshold contiguous regions
    supra = envelope > threshold
    # find transitions
    edges = np.diff(supra.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1

    # handle boundaries
    if supra[0]:
        starts = np.insert(starts, 0, 0)
    if supra[-1]:
        ends = np.append(ends, len(supra))

    # 5) enforce durations
    min_samples = max(1, int(min_duration_s * fs))
    max_samples = max(1, int(max_duration_s * fs))

    count = 0
    for s, e in zip(starts, ends):
        dur = e - s
        if dur >= min_samples and dur <= max_samples:
            count += 1

    return count

# ---------------------------
# Main function: extract sigma features
# ---------------------------
def extract_sigma_features(eeg_epochs: np.ndarray,
                           fs: float,
                           sigma_band: Tuple[float,float] = (12.0, 15.0),
                           beta_band: Tuple[float,float] = (15.0, 30.0),
                           ref_band: Tuple[float,float] = (0.5, 40.0),
                           spindle_params: Dict[str,Any] = None,
                           per_channel: bool = False) -> Tuple[np.ndarray, Dict[str,str]]:
    """
    Compute sigma-related features for all epochs.

    Args:
      eeg_epochs: np.ndarray shape (n_epochs, n_channels, n_samples)
      fs: sampling frequency (Hz)
      sigma_band: (low, high) Hz, default (12,15)
      beta_band: (low, high) Hz for beta power
      ref_band: reference band to compute relative power (default 0.5-40 Hz)
      spindle_params: dict overriding detect_spindles_envelope params
      per_channel: if True returns per-channel features shape (n_epochs, n_channels * n_feats)
                   if False returns aggregated features per epoch (mean across channels)

    Returns:
      features: np.ndarray shape (n_epochs, n_features)
                features order: [sigma_abs, sigma_rel, sigma_beta_ratio, spindle_density]
      meta: dict describing columns (column -> description)
    """

    if spindle_params is None:
        spindle_params = dict(env_smooth_ms=50.0, thresh_std=1.5, min_duration_s=0.5, max_duration_s=3.0)

    n_epochs, n_ch, n_samps = eeg_epochs.shape
    # will store per-channel features if requested
    sigma_abs = np.zeros((n_epochs, n_ch))
    sigma_rel = np.zeros((n_epochs, n_ch))
    sigma_beta_ratio = np.zeros((n_epochs, n_ch))
    spindle_counts = np.zeros((n_epochs, n_ch), dtype=int)

    eps = 1e-12
    for ep in range(n_epochs):
        for ch in range(n_ch):
            sig = eeg_epochs[ep, ch, :]

            # absolute sigma power
            p_sigma = bandpower_welch(sig, fs, sigma_band[0], sigma_band[1])
            # beta power
            p_beta = bandpower_welch(sig, fs, beta_band[0], beta_band[1])
            # reference (total) power
            p_ref = bandpower_welch(sig, fs, ref_band[0], ref_band[1])

            sigma_abs[ep, ch] = p_sigma
            sigma_rel[ep, ch] = p_sigma / (p_ref + eps)
            sigma_beta_ratio[ep, ch] = p_sigma / (p_beta + eps)

            # spindle count
            cnt = detect_spindles_envelope(sig, fs,
                                           sigma_band=sigma_band,
                                           env_smooth_ms=spindle_params.get('env_smooth_ms', 50.0),
                                           thresh_std=spindle_params.get('thresh_std', 1.5),
                                           min_duration_s=spindle_params.get('min_duration_s', 0.5),
                                           max_duration_s=spindle_params.get('max_duration_s', 3.0))
            spindle_counts[ep, ch] = cnt

    # Convert counts to density: spindles per minute (epoch length known by samples/fs)
    epoch_seconds = n_samps / float(fs)
    spindles_per_min = (spindle_counts / epoch_seconds) * 60.0  # shape (n_epochs, n_ch)

    if per_channel:
        # Concatenate per-channel features in fixed order per channel
        # Order: [sigma_abs(c1), sigma_rel(c1), sigma_beta_ratio(c1), spindles_per_min(c1), sigma_abs(c2), ...]
        features = []
        for ch in range(n_ch):
            features.append(sigma_abs[:, ch])
            features.append(sigma_rel[:, ch])
            features.append(sigma_beta_ratio[:, ch])
            features.append(spindles_per_min[:, ch])
        features = np.stack(features, axis=1)  # shape (n_epochs, n_ch*4)
        col_meta = {}
        cols = []
        for ch in range(n_ch):
            cols += [f"sigma_abs_ch{ch}", f"sigma_rel_ch{ch}", f"sigma_beta_ratio_ch{ch}", f"spindle_density_ch{ch}"]
        for i, name in enumerate(cols):
            col_meta[i] = name
        return features, col_meta
    else:
        # aggregate across channels (use mean and also include max as optional improvements)
        sigma_abs_mean = sigma_abs.mean(axis=1)
        sigma_rel_mean = sigma_rel.mean(axis=1)
        sigma_beta_ratio_mean = sigma_beta_ratio.mean(axis=1)
        spindle_density_mean = spindles_per_min.mean(axis=1)

        features = np.stack([sigma_abs_mean, sigma_rel_mean, sigma_beta_ratio_mean, spindle_density_mean], axis=1)
        col_meta = {0: "sigma_abs", 1: "sigma_rel", 2: "sigma_beta_ratio", 3: "spindle_density_per_min"}
        return features, col_meta
