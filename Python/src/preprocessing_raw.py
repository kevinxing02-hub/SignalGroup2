from scipy.signal import butter, lfilter, filtfilt, iirnotch
import numpy as np
from sklearn.linear_model import LinearRegression  # <-- behövs för EOG-regression


def lowpass_filter(data, cutoff, fs, order=2):
    """
    EXAMPLE IMPLEMENTATION: Simple low-pass Butterworth filter.

    Students should understand this basic filter and consider:
    - Is 40Hz the right cutoff for EEG?
    - What about high-pass filtering?
    - Should you use bandpass instead?
    - What about notch filtering for powerline interference?

    Args:
        data (np.ndarray): The input signal.
        cutoff (float): The cutoff frequency of the filter.
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered signal.
    """
    nyquist = 0.5 * fs

    #Robustness: ensure that the cutoff is always below Nyquist
    if cutoff >= nyquist:
        # t.ex. 90% av Nyquist om cutoff är för hög
        cutoff = 0.9 * nyquist

    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=2):
    """
    Design Butterworth high-pass filter using scipy.signal.butter().
    Normalizes cutoff by Nyquist frequency (fs/2) as required.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a


def butter_lowpass(cutoff, fs, order=4):
    """
    Design Butterworth low-pass filter using scipy.signal.butter().
    Normalizes cutoff by Nyquist frequency (fs/2) as required.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a


def apply_notch(signal, f0, fs, Q=30):
    # Skip if beyond Nyquist
    if f0 >= 0.5 * fs:
        return signal
    b, a = iirnotch(w0=f0 / (fs / 2.0), Q=Q)
    return filtfilt(b, a, signal)


# ====================== HJÄLPFUNKTIONER FÖR ITERATION 3 ======================

def band_power(signal, fs, low, high, order=2):
    """
    Beräkna bandkraft (mean squared amplitude) i ett frekvensband via bandpass + RMS.
    Används för EMG 20–40 Hz.
    """
    b, a = butter_bandpass(low, high, fs, order=order)
    filt_sig = filtfilt(b, a, signal)
    return np.mean(filt_sig ** 2)


def remove_eog_artifacts_from_eeg(eeg_epochs, eog_epochs):
    """
    EOG artefact removal from EEG via linear regression per epoch.

    Modell: EEG = β * EOG + residual
    Vi tar residualen som 'rensad EEG'.

    Args:
        eeg_epochs: np.ndarray (n_epochs, n_eeg_channels, n_samples)
        eog_epochs: np.ndarray (n_epochs, n_eog_channels, n_samples)

    Returns:
        np.ndarray: EEG med EOG-artefakter borttagna (samma shape som eeg_epochs)
    """
    n_epochs, n_eeg_ch, n_samples = eeg_epochs.shape
    _, n_eog_ch, n_samples_eog = eog_epochs.shape

    if n_samples != n_samples_eog:
        raise ValueError(
            f"EOG/EEG mismatch i samples: EEG {n_samples}, EOG {n_samples_eog}. "
            f"Säkerställ samma epoch-längd efter resampling."
        )

    cleaned_eeg = eeg_epochs.copy()
    lr = LinearRegression()

    for ep in range(n_epochs):
        # X = [EOG_left, EOG_right, ...] (samples x n_eog_ch)
        X = eog_epochs[ep].T  # shape: (n_samples, n_eog_ch)
        # Om EOG är helt noll → hoppa
        if np.allclose(X, 0):
            continue

        for ch in range(n_eeg_ch):
            y = eeg_epochs[ep, ch, :]  # shape: (n_samples,)

            # Fit regression EEG ~ EOG
            lr.fit(X, y)
            y_hat = lr.predict(X)

            # residual = EEG - EOG-bidrag
            residual = y - y_hat
            cleaned_eeg[ep, ch, :] = residual

    return cleaned_eeg


def apply_emg_adaptive_filtering(eeg_epochs, emg_epochs, eeg_fs, emg_fs):
    """
    EMG-baserad adaptiv filtrering:

    1. Beräkna EMG-kraft i 20–40 Hz per epoch.
    2. Sätt ett tröskelvärde (75:e percentil).
    3. För epoker där EMG-kraften är > tröskel:
       - Applicera en starkare low-pass (t.ex. 20 Hz) på EEG.

    Args:
        eeg_epochs: np.ndarray (n_epochs, n_eeg_channels, n_samples_eeg)
        emg_epochs: np.ndarray (n_epochs, 1, n_samples_emg)
        eeg_fs: EEG samplingfrekvens
        emg_fs: EMG samplingfrekvens

    Returns:
        eeg_epochs_filtered: np.ndarray samma shape som eeg_epochs
    """
    n_epochs, n_eeg_ch, n_eeg_samples = eeg_epochs.shape
    n_epochs_emg, n_emg_ch, n_emg_samples = emg_epochs.shape

    if n_emg_ch != 1:
        raise ValueError("Förväntar exakt 1 EMG-kanal (shape: n_epochs, 1, samples).")

    # Säkerställ samma antal epoker
    n_common = min(n_epochs, n_epochs_emg)

    # 1. EMG 20–40 Hz bandkraft per epoch
    emg_powers = np.zeros(n_common)
    for ep in range(n_common):
        emg_sig = emg_epochs[ep, 0, :]
        emg_powers[ep] = band_power(emg_sig, emg_fs, 20.0, 40.0, order=2)

    # 2. Tröskel (75:e percentil, kan justeras vid behov)
    threshold = np.percentile(emg_powers, 75)
    print(f"EMG 20–40 Hz power threshold (75th percentile): {threshold:.4e}")

    # 3. Starkare low-pass på EEG-epoker med hög EMG-aktivitet
    strong_lp_cut = 20.0  # starkare low-pass än standard 40 Hz
    b_strong_lp, a_strong_lp = butter_lowpass(strong_lp_cut, eeg_fs, order=4)

    eeg_filtered = eeg_epochs.copy()

    for ep in range(n_common):
        if emg_powers[ep] > threshold:
            # Denna epoch har mycket muskelaktivitet → dämpa EEG >20 Hz extra
            for ch in range(n_eeg_ch):
                x = eeg_filtered[ep, ch, :]
                eeg_filtered[ep, ch, :] = filtfilt(b_strong_lp, a_strong_lp, x)

    return eeg_filtered


# ================================ HUVUDFUNKTIONER ================================

def preprocess(data, config, channel_info=None):
    """
    STUDENT IMPLEMENTATION AREA: Preprocess data based on current iteration.

    Handles both:
    - single-channel (legacy)
    - multi-channel dict: {'eeg', 'eog', 'emg'}

    Iteration 3:
    - EEG + EOG + EMG
    - EOG-regressionsborttagning på EEG
    - EMG-baserad adaptiv low-pass på EEG
    """
    print(f"Preprocessing data for iteration {config.CURRENT_ITERATION}...")

    # Detect data format
    is_multi_channel = isinstance(data, dict) and 'eeg' in data

    if is_multi_channel:
        print("Processing multi-channel data (EEG + EOG + EMG)")
        return preprocess_multi_channel(data, config, channel_info=channel_info)
    else:
        print("Processing single-channel data (backward compatibility)")
        return preprocess_single_channel(data, config)


def preprocess_multi_channel(multi_channel_data, config, channel_info=None):
    """
    Preprocess multi-channel data based on iteration:
    - Iteration 1: 2 EEG channels only
    - Iteration 2: 2 EEG + 2 EOG channels
    - Iteration 3+: 2 EEG + 2 EOG + 1 EMG channels

    Strategy A: processera kontinuerlig signal, sedan segmentera i epoker.
    """
    preprocessed_data = {}

    # ==================== EEG ====================
    eeg_data = multi_channel_data['eeg']
    eeg_fs = (channel_info.get('eeg_fs') if channel_info is not None and 'eeg_fs' in channel_info else 125)

    hp_cut = 0.3   # High-pass cutoff: 0.3 Hz
    lp_cut = 40.0  # Low-pass cutoff: 40 Hz
    notch_f = 50.0 # Powerline 50 Hz (+ ev. 100 Hz)

    b_hp, a_hp = butter_highpass(hp_cut, eeg_fs, order=2)
    b_lp, a_lp = butter_lowpass(lp_cut, eeg_fs, order=4)

    n_epochs, n_channels, samples_per_epoch = eeg_data.shape

    # Reshape EEG till kontinuerligt (n_channels, total_samples)
    continuous_eeg = eeg_data.transpose(1, 0, 2).reshape(n_channels, -1)
    preprocessed_continuous_eeg = np.zeros_like(continuous_eeg)

    for ch in range(n_channels):
        x = continuous_eeg[ch, :].copy()

        # High-pass
        x = filtfilt(b_hp, a_hp, x)

        # Notch 50 Hz (+ 100 Hz om under Nyquist)
        x = apply_notch(x, notch_f, eeg_fs, Q=30)
        if 100.0 < 0.5 * eeg_fs:
            x = apply_notch(x, 100.0, eeg_fs, Q=30)

        # Low-pass 40 Hz
        x = filtfilt(b_lp, a_lp, x)

        preprocessed_continuous_eeg[ch, :] = x

    # Resegmentera EEG till epoker
    preprocessed_eeg = preprocessed_continuous_eeg.reshape(n_channels, n_epochs, samples_per_epoch)
    preprocessed_eeg = preprocessed_eeg.transpose(1, 0, 2)  # (n_epochs, n_channels, samples)
    preprocessed_data['eeg'] = preprocessed_eeg

    # ==================== EOG (från iteration 2) ====================
    if config.CURRENT_ITERATION >= 2 and 'eog' in multi_channel_data:
        eog_data = multi_channel_data['eog']
        eog_fs = (channel_info.get('eog_fs') if channel_info is not None and 'eog_fs' in channel_info else 50)

        eog_hp_cut = 0.5   # high-pass 0.5 Hz
        eog_lp_cut = 30.0  # low-pass 30 Hz
        eog_notch_f = 50.0

        b_eog_hp, a_eog_hp = butter_highpass(eog_hp_cut, eog_fs, order=2)
        b_eog_lp, a_eog_lp = butter_lowpass(eog_lp_cut, eog_fs, order=4)

        n_eog_epochs, n_eog_channels, eog_samples_per_epoch = eog_data.shape

        continuous_eog = eog_data.transpose(1, 0, 2).reshape(n_eog_channels, -1)
        preprocessed_continuous_eog = np.zeros_like(continuous_eog)

        for ch in range(n_eog_channels):
            x = continuous_eog[ch, :].copy()

            # High-pass 0.5 Hz
            x = filtfilt(b_eog_hp, a_eog_hp, x)

            # Notch 50 Hz (+100 om möjligt)
            x = apply_notch(x, eog_notch_f, eog_fs, Q=30)
            if 100.0 < 0.5 * eog_fs:
                x = apply_notch(x, 100.0, eog_fs, Q=30)

            # Low-pass 30 Hz
            x = filtfilt(b_eog_lp, a_eog_lp, x)

            preprocessed_continuous_eog[ch, :] = x

        preprocessed_eog = preprocessed_continuous_eog.reshape(
            n_eog_channels, n_eog_epochs, eog_samples_per_epoch
        ).transpose(1, 0, 2)

        preprocessed_data['eog'] = preprocessed_eog

    # ==================== EMG (från iteration 3) ====================
    if config.CURRENT_ITERATION >= 3 and 'emg' in multi_channel_data:
        emg_data = multi_channel_data['emg']
        emg_fs = (channel_info.get('emg_fs') if channel_info is not None and 'emg_fs' in channel_info else 125)

        preprocessed_emg = np.zeros_like(emg_data)

        for ep in range(emg_data.shape[0]):
            signal = emg_data[ep, 0, :]
            # EMG: behåll mer högfrekvent innehåll → högre cutoff
            filtered_signal = lowpass_filter(signal, 70, emg_fs)
            preprocessed_emg[ep, 0, :] = filtered_signal

        preprocessed_data['emg'] = preprocessed_emg
        print("Multi-channel preprocessing applied to EEG + EOG + EMG")

    elif config.CURRENT_ITERATION >= 2:
        print("Iteration 2: Processing EEG + EOG channels")
    else:
        print("Iteration 1: Processing EEG channels only")

    # ==================== ITERATION 3: ARTEFAKTHANTERING ====================
    if config.CURRENT_ITERATION >= 3:
        # 1) EOG-regressionsborttagning (om EOG finns)
        if 'eog' in preprocessed_data:
            print("Iteration 3: Removing EOG artifacts from EEG using LinearRegression...")
            preprocessed_data['eeg'] = remove_eog_artifacts_from_eeg(
                preprocessed_data['eeg'],
                preprocessed_data['eog']
            )

        # 2) EMG-baserad adaptiv filtrering (om EMG finns)
        if 'emg' in preprocessed_data:
            print("Iteration 3: Applying EMG-based adaptive low-pass filtering on EEG...")
            preprocessed_data['eeg'] = apply_emg_adaptive_filtering(
                preprocessed_data['eeg'],
                preprocessed_data['emg'],
                eeg_fs=eeg_fs,
                emg_fs=(channel_info.get('emg_fs') if channel_info is not None and 'emg_fs' in channel_info else 125)
            )

    # TODO: Students kan lägga till:
    # - Signal quality indices
    # - Normalisering per kanaltyp
    # - Cross-channel artefaktdetektion (t.ex. stora spikes samtidigt i alla kanaler)

    return preprocessed_data


def preprocess_single_channel(data, config):
    """
    Backward compatibility for single-channel preprocessing.
    """
    if config.CURRENT_ITERATION == 1:
        fs = 125  # TODO: hämta från data/config
        preprocessed_data = lowpass_filter(data, config.LOW_PASS_FILTER_FREQ, fs)

    elif config.CURRENT_ITERATION == 2:
        print("TODO: Implement enhanced preprocessing for iteration 2")
        preprocessed_data = data  # Placeholder

    elif config.CURRENT_ITERATION >= 3:
        print("TODO: Students should use multi-channel data format for iteration 3+")
        preprocessed_data = data  # Placeholder

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return preprocessed_data
