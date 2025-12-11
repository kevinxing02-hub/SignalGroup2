# src/preprocessing.py
"""
Robust preprocessing pipeline for Iteration 4 (sleep scoring).

Features:
 - Accepts a flexible configuration (PreprocConfig instance OR project config module with PREPROCESS dict)
 - Continuous (per-group) zero-phase filtering using SOS + sosfiltfilt
 - Baseline removal (HPF), notch (base + harmonics), bandpass (HPF..LPF)
 - Optional EOG regression (per-epoch or global)
 - EMG-based epoch detection + optional epoch-wise gentle low-pass
 - Debug plots (PSD, edge plots) saved to outputs directory if enabled
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import os
import traceback

# scipy imports: use SOS designs for numerical stability
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from sklearn.linear_model import LinearRegression

# -------------------------
# Config / dataclass
# -------------------------
@dataclass
class PreprocConfig:
    # frequency cutoffs (Hz)
    highpass: float = 0.10             # keep low to preserve delta (0.5-4 Hz)
    lowpass: float = 40.0              # upper bound for sleep EEG

    # notch settings
    notch_freq: float = 50.0           # base (50 or 60)
    notch_Q: float = 30.0
    notch_harmonics: int = 2           # 1 => base only, 2 => base+2*base, etc.

    # filter orders (SOS)
    hp_order: int = 2
    bp_order: int = 4
    lp_order: int = 4

    # filtfilt padding options
    padtype: Optional[str] = 'odd'     # 'odd'|'even'|'constant'|None
    padlen: Optional[int] = None       # None -> scipy default

    # EOG regression options
    do_eog_regression: bool = True
    eog_regression_per_epoch: bool = True

    # EMG detection + optional processing
    detect_emg_epochs: bool = True
    emg_power_band: Tuple[float, float] = (20.0, 40.0)
    emg_power_pct_threshold: float = 75.0
    apply_emg_lowpass: bool = False
    emg_lowpass_cut: float = 30.0
    emg_lowpass_order: int = 4

    # debug plotting
    debug_plots: bool = False
    outputs_dir: str = "./outputs/preproc_debug"

# -------------------------
# Helper: coerce config
# -------------------------
def coerce_preproc_config(cfg_in: Any) -> PreprocConfig:
    """
    Accept:
      - PreprocConfig instance -> return directly
      - module-like config (e.g. import config) with PREPROCESS dict -> build from it
      - None or unknown -> default PreprocConfig
    """
    if isinstance(cfg_in, PreprocConfig):
        return cfg_in
    if cfg_in is None:
        return PreprocConfig()
    # try to read PREPROCESS dict
    try:
        # case: cfg_in is module with PREPROCESS dict
        if hasattr(cfg_in, "PREPROCESS"):
            p = getattr(cfg_in, "PREPROCESS")
            if isinstance(p, dict):
                return PreprocConfig(**{k: v for k, v in p.items() if k in PreprocConfig.__annotations__})
        # case: legacy config module with top-level keys
        # map some common keys
        kw = {}
        if hasattr(cfg_in, "LOW_PASS_FILTER_FREQ"):
            kw["lowpass"] = getattr(cfg_in, "LOW_PASS_FILTER_FREQ")
        if hasattr(cfg_in, "CURRENT_ITERATION"):
            # nothing to map directly
            pass
        # debug flags
        if hasattr(cfg_in, "DEBUG_PREPROC"):
            kw["debug_plots"] = getattr(cfg_in, "DEBUG_PREPROC")
        if hasattr(cfg_in, "OUTPUT_DIR"):
            kw["outputs_dir"] = getattr(cfg_in, "OUTPUT_DIR")
        if kw:
            return PreprocConfig(**{k:v for k,v in kw.items() if k in PreprocConfig.__annotations__})
    except Exception:
        pass
    # fallback default
    return PreprocConfig()

# -------------------------
# Filter builders (SOS)
# -------------------------
def sos_highpass(cutoff: float, fs: float, order: int = 2):
    # ensure cutoff < fs/2
    nyq = fs / 2.0
    if cutoff <= 0 or cutoff >= nyq:
        raise ValueError(f"Invalid highpass cutoff {cutoff} for fs={fs}")
    sos = butter(order, cutoff, btype='highpass', fs=fs, output='sos')
    return sos

def sos_lowpass(cutoff: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    if cutoff <= 0:
        raise ValueError("lowpass must be > 0")
    cutoff = min(cutoff, nyq*0.99)
    sos = butter(order, cutoff, btype='lowpass', fs=fs, output='sos')
    return sos

def sos_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    if lowcut <= 0 or highcut <= lowcut or highcut >= nyq:
        # clamp values but still attempt
        lowcut = max(0.001, lowcut)
        highcut = min(highcut, nyq*0.99)
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sos

def apply_sos_filtfilt(sos, signal: np.ndarray, padtype: Optional[str], padlen: Optional[int]):
    """
    Wrapper for sosfiltfilt with pad safety. scipy.signal.sosfiltfilt accepts padtype/padlen kwargs.
    """
    if padtype is None and padlen is None:
        return sosfiltfilt(sos, signal)
    # pass padtype/padlen if supported
    try:
        return sosfiltfilt(sos, signal, padtype=padtype, padlen=padlen)
    except TypeError:
        # older scipy versions may not accept padtype/padlen in sosfiltfilt signature
        # fall back to sosfiltfilt without those args
        return sosfiltfilt(sos, signal)

def apply_notch_freqs(signal: np.ndarray, fs: float, base_freq: float, Q: float, harmonics: int,
                      padtype: Optional[str], padlen: Optional[int]):
    """
    Apply IIR notch filters at base_freq and harmonics (if below Nyquist), using filtfilt.
    If filtfilt fails, try tf->sos + sosfiltfilt; on repeated failure return current signal.
    """
    nyq = fs / 2.0
    out = signal
    for k in range(1, max(1, int(harmonics))+1):
        f = base_freq * k
        if f >= nyq - 1e-6:
            break
        w0 = f / nyq
        try:
            b, a = iirnotch(w0=w0, Q=Q)
            # filtfilt supports padtype/padlen
            if padtype is None and padlen is None:
                out = filtfilt(b, a, out)
            else:
                out = filtfilt(b, a, out, padtype=padtype, padlen=padlen)
        except Exception:
            # try SOS fallback
            try:
                from scipy.signal import tf2sos
                sos = tf2sos(b, a)
                out = apply_sos_filtfilt(sos, out, padtype=padtype, padlen=padlen)
            except Exception:
                # give up on this notch
                break
    return out

# -------------------------
# EOG regression (per-epoch)
# -------------------------
def eog_regression_epochwise(eeg_epochs: np.ndarray, eog_epochs: np.ndarray) -> np.ndarray:
    n_epochs, n_eeg_ch, n_samples = eeg_epochs.shape
    _, n_eog_ch, n_samples_eog = eog_epochs.shape
    if n_samples != n_samples_eog:
        raise ValueError("EOG/EEG epoch size mismatch")

    cleaned = eeg_epochs.copy()
    lr = LinearRegression()
    for ep in range(n_epochs):
        X = eog_epochs[ep].T  # (n_samples, n_eog_ch)
        if np.allclose(X, 0):
            continue
        for ch in range(n_eeg_ch):
            y = eeg_epochs[ep, ch, :]
            try:
                lr.fit(X, y)
                yhat = lr.predict(X)
                cleaned[ep, ch, :] = y - yhat
            except Exception:
                # fallback: subtract projection using simple least squares
                try:
                    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                    cleaned[ep, ch, :] = y - X.dot(coef)
                except Exception:
                    pass
    return cleaned

# -------------------------
# EMG epoch detection
# -------------------------
def detect_emg_epochs(emg_epochs: np.ndarray, fs: float, band: Tuple[float,float], pct_threshold: float=75.0) -> Dict[str,Any]:
    n_epochs = emg_epochs.shape[0]
    powers = np.zeros(n_epochs)
    low, high = band
    sos = sos_bandpass(low, high, fs, order=2)
    for ep in range(n_epochs):
        s = emg_epochs[ep].mean(axis=0)
        try:
            filt = apply_sos_filtfilt(sos, s, padtype='odd', padlen=None)
            powers[ep] = np.mean(filt**2)
        except Exception:
            powers[ep] = np.mean(s**2)
    thresh = np.percentile(powers, pct_threshold)
    mask = powers > thresh
    return {"powers": powers, "threshold": thresh, "mask": mask}

# -------------------------
# Core pipeline
# -------------------------
def preprocess(multi_channel_data: Dict[str, np.ndarray], cfg: Any = None, channel_info: Dict[str,Any]=None) -> Dict[str, np.ndarray]:
    """
    Main entry point. Accepts:
      - multi_channel_data: dict with keys 'eeg','eog','emg','other' mapping to epoch arrays
        shapes: (n_epochs, n_channels, samples_per_epoch)
      - cfg: either PreprocConfig instance, or project config module (with PREPROCESS dict), or None
      - channel_info: dict from data_loader with per-group fs information
    Returns dict of preprocessed groups.
    """
    cfg_obj = coerce_preproc_config(cfg)
    # ensure outputs dir is set
    if not hasattr(cfg_obj, "outputs_dir") or cfg_obj.outputs_dir is None:
        cfg_obj.outputs_dir = "./outputs/preproc_debug"

    if channel_info is None:
        channel_info = {}

    out: Dict[str, np.ndarray] = {}
    emg_stats = None

    # create outputs dir if debug enabled
    if cfg_obj.debug_plots:
        os.makedirs(cfg_obj.outputs_dir, exist_ok=True)

    # ---------------- EEG ----------------
    if 'eeg' in multi_channel_data and multi_channel_data['eeg'] is not None:
        eeg = multi_channel_data['eeg']
        eeg_fs = float(channel_info.get('eeg_fs', channel_info.get('global_fs', 125.0)))
        n_epochs, n_eeg_ch, n_samp = eeg.shape

        # continuous concatenation to avoid epoch-edge transients
        eeg_cont = eeg.transpose(1,0,2).reshape(n_eeg_ch, -1)

        # 1) HPF baseline removal
        try:
            sos_hp = sos_highpass(cfg_obj.highpass, eeg_fs, order=cfg_obj.hp_order)
            for ch in range(n_eeg_ch):
                eeg_cont[ch] = apply_sos_filtfilt(sos_hp, eeg_cont[ch], padtype=cfg_obj.padtype, padlen=cfg_obj.padlen)
        except Exception:
            # fallback: detrend per-channel
            for ch in range(n_eeg_ch):
                eeg_cont[ch] = eeg_cont[ch] - np.mean(eeg_cont[ch])

        # 2) notch(s)
        for ch in range(n_eeg_ch):
            eeg_cont[ch] = apply_notch_freqs(eeg_cont[ch], eeg_fs, cfg_obj.notch_freq, cfg_obj.notch_Q, cfg_obj.notch_harmonics, cfg_obj.padtype, cfg_obj.padlen)

        # 3) bandpass (HP..LP)
        try:
            sos_bp = sos_bandpass(cfg_obj.highpass, cfg_obj.lowpass, eeg_fs, order=cfg_obj.bp_order)
            for ch in range(n_eeg_ch):
                eeg_cont[ch] = apply_sos_filtfilt(sos_bp, eeg_cont[ch], padtype=cfg_obj.padtype, padlen=cfg_obj.padlen)
        except Exception:
            # fallback: lowpass only, then demean
            try:
                sos_lp = sos_lowpass(cfg_obj.lowpass, eeg_fs, order=cfg_obj.lp_order)
                for ch in range(n_eeg_ch):
                    eeg_cont[ch] = apply_sos_filtfilt(sos_lp, eeg_cont[ch], padtype=cfg_obj.padtype, padlen=cfg_obj.padlen)
            except Exception:
                pass

        # reshape back to epochs
        eeg_pp = eeg_cont.reshape(n_eeg_ch, n_epochs, n_samp).transpose(1,0,2)
        out['eeg'] = eeg_pp

    # ---------------- EOG ----------------
    if 'eog' in multi_channel_data and multi_channel_data['eog'] is not None:
        eog = multi_channel_data['eog']
        eog_fs = float(channel_info.get('eog_fs', channel_info.get('global_fs', 125.0)))
        n_epochs_e, n_eog_ch, n_samp_eog = eog.shape
        eog_cont = eog.transpose(1,0,2).reshape(n_eog_ch, -1)
        # slightly stronger HP for EOG to avoid slow drifts
        hp_eog = max(cfg_obj.highpass, 0.2)
        try:
            sos_hp_eog = sos_highpass(hp_eog, eog_fs, order=cfg_obj.hp_order)
            sos_lp_eog = sos_lowpass(min(cfg_obj.lowpass, 30.0), eog_fs, order=cfg_obj.lp_order)
            for ch in range(n_eog_ch):
                s = eog_cont[ch]
                s = apply_sos_filtfilt(sos_hp_eog, s, padtype=cfg_obj.padtype, padlen=cfg_obj.padlen)
                s = apply_notch_freqs(s, eog_fs, cfg_obj.notch_freq, cfg_obj.notch_Q, cfg_obj.notch_harmonics, cfg_obj.padtype, cfg_obj.padlen)
                s = apply_sos_filtfilt(sos_lp_eog, s, padtype=cfg_obj.padtype, padlen=cfg_obj.padlen)
                eog_cont[ch] = s
        except Exception:
            for ch in range(n_eog_ch):
                eog_cont[ch] = eog_cont[ch] - np.mean(eog_cont[ch])

        eog_pp = eog_cont.reshape(n_eog_ch, n_epochs_e, n_samp_eog).transpose(1,0,2)
        out['eog'] = eog_pp

    # ---------------- EMG ----------------
    if 'emg' in multi_channel_data and multi_channel_data['emg'] is not None:
        emg = multi_channel_data['emg']
        emg_fs = float(channel_info.get('emg_fs', channel_info.get('global_fs', 125.0)))
        n_epochs_emg, n_emg_ch, n_samp_emg = emg.shape
        emg_pp = np.zeros_like(emg)
        try:
            sos_lp_emg = sos_lowpass(min(100.0, emg_fs*0.45), emg_fs, order=2)
            for ep in range(n_epochs_emg):
                for ch in range(n_emg_ch):
                    emg_pp[ep,ch,:] = apply_sos_filtfilt(sos_lp_emg, emg[ep,ch,:], padtype=cfg_obj.padtype, padlen=cfg_obj.padlen)
        except Exception:
            for ep in range(n_epochs_emg):
                for ch in range(n_emg_ch):
                    s = emg[ep,ch,:]
                    emg_pp[ep,ch,:] = s - np.mean(s)

        out['emg'] = emg_pp

        if cfg_obj.detect_emg_epochs:
            emg_stats = detect_emg_epochs(emg_pp, emg_fs, cfg_obj.emg_power_band, cfg_obj.emg_power_pct_threshold)
            if cfg_obj.apply_emg_lowpass and emg_stats is not None and 'eeg' in out:
                mask = emg_stats['mask']
                sos_emg_lp = sos_lowpass(cfg_obj.emg_lowpass_cut, eeg_fs, order=cfg_obj.emg_lowpass_order)
                eeg_tmp = out.get('eeg')
                for ep in np.where(mask)[0]:
                    for ch in range(eeg_tmp.shape[1]):
                        try:
                            eeg_tmp[ep,ch,:] = apply_sos_filtfilt(sos_emg_lp, eeg_tmp[ep,ch,:], padtype=cfg_obj.padtype, padlen=cfg_obj.padlen)
                        except Exception:
                            pass
                out['eeg'] = eeg_tmp

    # ---------------- EOG regression ----------------
    if cfg_obj.do_eog_regression and 'eog' in out and 'eeg' in out:
        try:
            eeg_pp = out['eeg']
            eog_pp = out['eog']
            # if sample lengths differ, resample EOG to EEG epoch length (simple interpolation)
            if eeg_pp.shape[2] != eog_pp.shape[2]:
                n_epochs_reg = min(eeg_pp.shape[0], eog_pp.shape[0])
                eog_resampled = np.zeros((n_epochs_reg, eog_pp.shape[1], eeg_pp.shape[2]), dtype=eog_pp.dtype)
                for ep in range(n_epochs_reg):
                    for ch in range(eog_pp.shape[1]):
                        old = eog_pp[ep,ch,:]
                        old_x = np.linspace(0,1,len(old))
                        new_x = np.linspace(0,1,eeg_pp.shape[2])
                        eog_resampled[ep,ch,:] = np.interp(new_x, old_x, old)
                eog_use = eog_resampled
                eeg_use = eeg_pp[:n_epochs_reg]
            else:
                eog_use = eog_pp
                eeg_use = eeg_pp

            if cfg_obj.eog_regression_per_epoch:
                cleaned = eog_regression_epochwise(eeg_use, eog_use)
            else:
                # global regression across concatenated signal
                n_epochs_r = eeg_use.shape[0]
                X = eog_use.transpose(1,0,2).reshape(eog_use.shape[1], -1).T  # (samples_total, n_eog_ch)
                Y = eeg_use.transpose(1,0,2).reshape(eeg_use.shape[1], -1).T
                lr = LinearRegression()
                lr.fit(X, Y)  # multioutput
                Yhat = lr.predict(X).T.reshape(eeg_use.shape[1], n_epochs_r, eeg_use.shape[2]).transpose(1,0,2)
                cleaned = eeg_use - Yhat

            out['eeg'][:cleaned.shape[0]] = cleaned
        except Exception as e:
            print("WARNING: EOG regression failed:", type(e).__name__, e)

    # ---------------- Debug plots ----------------
    if cfg_obj.debug_plots:
        try:
            _debug_plots(out, multi_channel_data, channel_info, cfg_obj, emg_stats)
        except Exception:
            print("WARNING: debug plots generation failed")
            traceback.print_exc()

    return out

# -------------------------
# Debugging / plotting helpers
# -------------------------
def _debug_plots(pp_data: Dict[str,np.ndarray], raw_data: Dict[str,np.ndarray], channel_info: Dict[str,Any], cfg: PreprocConfig, emg_stats: Optional[Dict]=None):
    """
    Save diagnostics for epoch 0, ch 0:
      - PSD before/after
      - First/last 1s overlay
    """
    os.makedirs(cfg.outputs_dir, exist_ok=True)
    fs = float(channel_info.get('eeg_fs', channel_info.get('global_fs', 125.0)))
    ch_idx = 0; ep_idx = 0

    try:
        raw = raw_data['eeg'][ep_idx, ch_idx, :].astype(float)
        proc = pp_data['eeg'][ep_idx, ch_idx, :].astype(float)
    except Exception as e:
        print("DEBUG: unable to extract raw/proc epoch for plotting:", e)
        return

    try:
        n = len(raw)
        freqs = np.fft.rfftfreq(n, 1.0/fs)
        raw_psd = np.abs(np.fft.rfft(raw))**2
        proc_psd = np.abs(np.fft.rfft(proc))**2

        import matplotlib.pyplot as plt
        plt.figure(figsize=(9,4))
        plt.semilogy(freqs, raw_psd, label='raw')
        plt.semilogy(freqs, proc_psd, label='proc')
        plt.xlim([0, min(60, fs/2)])
        plt.legend()
        plt.title(f'PSD epoch{ep_idx} ch{ch_idx}')
        p1 = os.path.join(cfg.outputs_dir, f'{channel_info.get("record_id","rec")}_ch{ch_idx}_ep{ep_idx}_psd.png')
        plt.savefig(p1, dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved PSD debug:", p1)
    except Exception as e:
        print("PSD debug failed:", e)

    try:
        n1 = int(fs*1.0)
        t = np.arange(n1)/fs
        raw_first = raw[:n1]; proc_first = proc[:n1]
        raw_last = raw[-n1:]; proc_last = proc[-n1:]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,3))
        plt.subplot(1,2,1)
        plt.plot(t, raw_first, 'b-', label='raw'); plt.plot(t, proc_first, 'r-', label='proc')
        plt.title('First 1s')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(t, raw_last, 'b-', label='raw'); plt.plot(t, proc_last, 'r-', label='proc')
        plt.title('Last 1s')
        p2 = os.path.join(cfg.outputs_dir, f'{channel_info.get("record_id","rec")}_ch{ch_idx}_ep{ep_idx}_edges.png')
        plt.savefig(p2, dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved edge debug:", p2)
    except Exception as e:
        print("Edge debug failed:", e)

# -------------------------
# Single-channel fallback
# -------------------------
def preprocess_single_channel(signal: np.ndarray, cfg: Any = None, fs: float = 125.0) -> np.ndarray:
    cfg_obj = coerce_preproc_config(cfg)
    sig = signal.copy()
    try:
        sos_hp = sos_highpass(cfg_obj.highpass, fs, order=cfg_obj.hp_order)
        sig = apply_sos_filtfilt(sos_hp, sig, padtype=cfg_obj.padtype, padlen=cfg_obj.padlen)
        sig = apply_notch_freqs(sig, fs, cfg_obj.notch_freq, cfg_obj.notch_Q, cfg_obj.notch_harmonics, cfg_obj.padtype, cfg_obj.padlen)
        sos_bp = sos_bandpass(cfg_obj.highpass, cfg_obj.lowpass, fs, order=cfg_obj.bp_order)
        sig = apply_sos_filtfilt(sos_bp, sig, padtype=cfg_obj.padtype, padlen=cfg_obj.padlen)
    except Exception:
        sig = sig - np.mean(sig)
    return sig
