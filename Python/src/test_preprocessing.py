"""
tests/test_preprocessing.py

Preprocessing validation test for Iteration 4 pipeline.

Saves diagnostics to: <config.OUTPUT_DIR>/preproc_validation/<record_id>_...

Usage:
    python tests/test_preprocessing.py
"""

from pathlib import Path
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# --- project-relative import setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # assume tests/ inside Python/
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config           # your project config
from data_loader import load_all_training_data, load_training_data
from preprocessing import preprocess, PreprocConfig, coerce_preproc_config

# -------------------------
# Helpers: PSD, delta power
# -------------------------
def compute_one_sided_psd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return freqs (>=0) and power (linear)."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, 1.0/fs)
    psd = np.abs(np.fft.rfft(x))**2 / n
    return freqs, psd

# -------------------------
# Validation functions
# -------------------------
def mean_check(raw_signal: np.ndarray, proc_signal: np.ndarray, threshold: float=1.0) -> Tuple[bool, float, float]:
    raw_mean = float(np.mean(raw_signal))
    proc_mean = float(np.mean(proc_signal))
    ok = abs(proc_mean) < threshold
    return ok, raw_mean, proc_mean

def delta_preservation_check(raw_signal: np.ndarray, proc_signal: np.ndarray, fs: float,
                             low: float=0.5, high: float=4.0, min_ratio: float=0.8):
    f_raw, p_raw = compute_one_sided_psd(raw_signal, fs)
    f_proc, p_proc = compute_one_sided_psd(proc_signal, fs)
    mask = (f_raw >= low) & (f_raw <= high)
    raw_delta = float(np.mean(p_raw[mask])) if np.any(mask) else 0.0
    proc_delta = float(np.mean(p_proc[mask])) if np.any(mask) else 0.0
    ratio = (proc_delta / raw_delta) if raw_delta > 0 else 1.0
    ok = ratio >= min_ratio
    return ok, ratio, (f_raw, p_raw, p_proc)

def phase_shift_check(raw_signal: np.ndarray, proc_signal: np.ndarray, fs: float, n_peaks_to_check:int=5):
    from scipy.signal import find_peaks
    # find peaks in absolute raw signal
    raw_peaks, _ = find_peaks(np.abs(raw_signal), height=np.std(raw_signal)*0.5, distance=int(fs*0.05))
    if len(raw_peaks) == 0:
        return True, 0.0, 0
    preserved = []
    for pk in raw_peaks[:n_peaks_to_check]:
        window = int(fs*0.05)
        lo = max(0, pk - window)
        hi = min(len(proc_signal), pk + window)
        if lo >= hi:
            preserved.append(False)
            continue
        local = proc_signal[lo:hi]
        local_pk = np.argmax(np.abs(local)) + lo
        dt = abs(local_pk - pk) / fs
        preserved.append(dt < 0.05)  # within 50 ms
    mean_shift = 0.0
    if len(raw_peaks)>0:
        # compute average absolute shift for first n_peaks_to_check
        shifts = []
        for pk in raw_peaks[:n_peaks_to_check]:
            window = int(fs*0.05)
            lo = max(0, pk - window)
            hi = min(len(proc_signal), pk + window)
            local = proc_signal[lo:hi]
            if len(local)==0:
                shifts.append(1.0)
                continue
            local_pk = np.argmax(np.abs(local)) + lo
            shifts.append(abs(local_pk - pk) / fs)
        mean_shift = float(np.mean(shifts))
    percent_preserved = float(np.mean(preserved))*100 if preserved else 100.0
    ok = np.mean(preserved) > 0.8 if preserved else True
    return ok, mean_shift, len(raw_peaks)

# -------------------------
# Plot helpers
# -------------------------
def save_power_plots(freqs, raw_psd, proc_psd, notch_freq, outpath):
    # full 0-60 Hz
    fig, ax = plt.subplots(2,1, figsize=(10,8))
    ax[0].semilogy(freqs[freqs<=60], raw_psd[freqs<=60], label='raw', linewidth=1.2)
    ax[0].semilogy(freqs[freqs<=60], proc_psd[freqs<=60], label='proc', linewidth=1.2)
    ax[0].axvline(notch_freq, color='r', linestyle='--', label=f'notch {notch_freq}Hz')
    if 2*notch_freq < freqs.max():
        ax[0].axvline(2*notch_freq, color='r', linestyle='--', label=f'notch {2*notch_freq}Hz')
    ax[0].set_xlim(0,60); ax[0].set_ylabel('Power'); ax[0].legend(); ax[0].grid(True)
    ax[0].set_title('PSD 0-60 Hz')

    # zoom around notch
    zlo, zhi = notch_freq-2.0, notch_freq+2.0
    mask = (freqs>=zlo) & (freqs<=zhi)
    ax[1].plot(freqs[mask], raw_psd[mask], 'b-o', label='raw')
    ax[1].plot(freqs[mask], proc_psd[mask], 'r-s', label='proc')
    ax[1].axvline(notch_freq, color='k', linestyle='--')
    ax[1].set_xlabel('Freq (Hz)'); ax[1].set_ylabel('Power'); ax[1].legend(); ax[1].grid(True)
    ax[1].set_title(f'Zoom around {notch_freq}Hz')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

def save_edge_plots(raw, proc, fs, outpath):
    n1 = int(fs*1.0)
    t = np.arange(n1)/fs
    fig, axs = plt.subplots(1,2, figsize=(12,3))
    axs[0].plot(t, raw[:n1], label='raw'); axs[0].plot(t, proc[:n1], label='proc')
    axs[0].set_title('First 1s'); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(t, raw[-n1:], label='raw'); axs[1].plot(t, proc[-n1:], label='proc')
    axs[1].set_title('Last 1s'); axs[1].legend(); axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

# -------------------------
# Main test runner
# -------------------------
def run_test():
    print("\n=== PREPROCESSING VALIDATION SUITE ===\n")

    # determine training dir from config
    training_dir = getattr(config, "TRAINING_DIR", None)
    data_dir = getattr(config, "DATA_DIR", None)
    if training_dir is None and data_dir is not None:
        training_dir = str(Path(data_dir) / "training")
    if training_dir is None:
        # fallback relative
        training_dir = str((PROJECT_ROOT / "data" / "training").resolve())

    training_dir = Path(training_dir)
    if not training_dir.exists():
        print(f"SKIP: training dir not found: {training_dir}")
        return

    # find first EDF+XML pair
    edf_files = sorted([p for p in training_dir.glob("*.edf")])
    if not edf_files:
        print(f"SKIP: no .edf files in {training_dir}")
        return

    chosen = edf_files[0]
    xml_candidate = chosen.with_suffix('.xml')
    if not xml_candidate.exists():
        # try uppercase/lowercase variants
        alt = list(training_dir.glob(chosen.stem + ".*"))
        xml_candidate = None
        for a in alt:
            if a.suffix.lower().startswith('.xml'):
                xml_candidate = a
                break
    if xml_candidate is None or not xml_candidate.exists():
        print(f"SKIP: Matching XML not found for {chosen}")
        return

    print(f"Using EDF: {chosen}")
    print(f"Using XML: {xml_candidate}")

    # load data
    try:
        data, labels, channel_info = load_training_data(str(chosen), str(xml_candidate))
    except Exception as e:
        print("ERROR loading recording:", type(e).__name__, e)
        return

    record_id = channel_info.get("record_id", chosen.stem)
    print(f"\nLoaded recording: {record_id}")
    print("Channel info (summary):")
    for k in sorted(channel_info.keys()):
        print(f"   {k} : {channel_info[k]}")
    print()

    # build preproc cfg: prefer config.PREPROCESS if present
    cfg_source = None
    if hasattr(config, "PREPROCESS"):
        cfg_source = config.PREPROCESS
        cfg = coerce_preproc_config(config)
        print("Using PREPROCESS from config module.")
    else:
        cfg = PreprocConfig()
        print("Using default PreprocConfig.")

    # force debug plots and outputs path
    cfg.debug_plots = True
    out_base = getattr(config, "OUTPUT_DIR", "./outputs")
    cfg.outputs_dir = str(Path(out_base) / "preproc_validation" / record_id)
    os.makedirs(cfg.outputs_dir, exist_ok=True)
    print(f"Debug outputs -> {cfg.outputs_dir}\n")

    # run preprocessing
    print("Running preprocess(...) ...")
    proc = preprocess(data, cfg, channel_info=channel_info)
    print("Preprocessing done.\n")

    # select first epoch/channel for checks
    fs = float(channel_info.get('eeg_fs', channel_info.get('global_fs', 125.0)))
    ep_idx = 0; ch_idx = 0
    raw_sig = data['eeg'][ep_idx, ch_idx, :].astype(float)
    proc_sig = proc['eeg'][ep_idx, ch_idx, :].astype(float)

    # Mean check
    mean_ok, raw_mean, proc_mean = mean_check(raw_sig, proc_sig, threshold=1.0)
    print(f"[Mean Check] raw_mean={raw_mean:.4e}, pre_mean={proc_mean:.4e} -> {'PASS' if mean_ok else 'FAIL'}")

    # Delta preservation
    delta_ok, ratio, (freqs, raw_psd, proc_psd) = delta_preservation_check(raw_sig, proc_sig, fs, low=0.5, high=4.0, min_ratio=0.8)
    print(f"[Delta Preservation] preservation_ratio={ratio:.3f} -> {'PASS' if delta_ok else 'FAIL'}")

    # Save PSD and edge plots
    ps_out = Path(cfg.outputs_dir) / f"{record_id}_ch{ch_idx}_ep{ep_idx}_ps.png"
    save_power_plots(freqs, raw_psd, proc_psd, cfg.notch_freq, str(ps_out))
    edges_out = Path(cfg.outputs_dir) / f"{record_id}_ch{ch_idx}_ep{ep_idx}_edges.png"
    save_edge_plots(raw_sig, proc_sig, fs, str(edges_out))
    print(f"Saved power spectrum comparison to: {ps_out}")
    print(f"Saved edge effects plot to: {edges_out}")

    # Phase shift check
    phase_ok, mean_shift, n_peaks = phase_shift_check(raw_sig, proc_sig, fs)
    print(f"[Phase Shift] peaks_found={n_peaks}, mean_shift={mean_shift:.4f}s -> {'PASS' if phase_ok else 'FAIL'}")

    # summary
    print("\n=== SUMMARY ===")
    print(f"Mean check:           {'PASS' if mean_ok else 'FAIL'}")
    print(f"Delta preservation:   {'PASS' if delta_ok else 'FAIL'}")
    print(f"Power spectrum plot:  Saved -> {ps_out}")
    print(f"Edge effects plot:    Saved -> {edges_out}")
    print(f"Phase-shift check:    {'PASS' if phase_ok else 'FAIL'}")
    print(f"\nDiagnostics saved to: {cfg.outputs_dir}")

    if not (mean_ok and delta_ok):
        print("\nOne or more critical validations failed.")
        print("Hints:")
        print(" - Try lowering hp cutoff (cfg.highpass) toward 0.05-0.15 Hz to preserve delta band.")
        print(" - Try different padtype ('odd' vs None) or padlen in PreprocConfig.")
        print(" - If hardware already highpassed at 0.15Hz, consider hp ~0.1 or skip HPF.")
    else:
        print("\nAll required validations PASSED.")

if __name__ == "__main__":
    run_test()
"""
tests/test_preprocessing.py

Preprocessing validation test for Iteration 4 pipeline.

Saves diagnostics to: <config.OUTPUT_DIR>/preproc_validation/<record_id>_...

Usage:
    python tests/test_preprocessing.py
"""

from pathlib import Path
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# --- project-relative import setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # assume tests/ inside Python/
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config           # your project config
from data_loader import load_all_training_data, load_training_data
from preprocessing import preprocess, PreprocConfig, coerce_preproc_config

# -------------------------
# Helpers: PSD, delta power
# -------------------------
def compute_one_sided_psd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return freqs (>=0) and power (linear)."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, 1.0/fs)
    psd = np.abs(np.fft.rfft(x))**2 / n
    return freqs, psd

# -------------------------
# Validation functions
# -------------------------
def mean_check(raw_signal: np.ndarray, proc_signal: np.ndarray, threshold: float=1.0) -> Tuple[bool, float, float]:
    raw_mean = float(np.mean(raw_signal))
    proc_mean = float(np.mean(proc_signal))
    ok = abs(proc_mean) < threshold
    return ok, raw_mean, proc_mean

def delta_preservation_check(raw_signal: np.ndarray, proc_signal: np.ndarray, fs: float,
                             low: float=0.5, high: float=4.0, min_ratio: float=0.8):
    f_raw, p_raw = compute_one_sided_psd(raw_signal, fs)
    f_proc, p_proc = compute_one_sided_psd(proc_signal, fs)
    mask = (f_raw >= low) & (f_raw <= high)
    raw_delta = float(np.mean(p_raw[mask])) if np.any(mask) else 0.0
    proc_delta = float(np.mean(p_proc[mask])) if np.any(mask) else 0.0
    ratio = (proc_delta / raw_delta) if raw_delta > 0 else 1.0
    ok = ratio >= min_ratio
    return ok, ratio, (f_raw, p_raw, p_proc)

def phase_shift_check(raw_signal: np.ndarray, proc_signal: np.ndarray, fs: float, n_peaks_to_check:int=5):
    from scipy.signal import find_peaks
    # find peaks in absolute raw signal
    raw_peaks, _ = find_peaks(np.abs(raw_signal), height=np.std(raw_signal)*0.5, distance=int(fs*0.05))
    if len(raw_peaks) == 0:
        return True, 0.0, 0
    preserved = []
    for pk in raw_peaks[:n_peaks_to_check]:
        window = int(fs*0.05)
        lo = max(0, pk - window)
        hi = min(len(proc_signal), pk + window)
        if lo >= hi:
            preserved.append(False)
            continue
        local = proc_signal[lo:hi]
        local_pk = np.argmax(np.abs(local)) + lo
        dt = abs(local_pk - pk) / fs
        preserved.append(dt < 0.05)  # within 50 ms
    mean_shift = 0.0
    if len(raw_peaks)>0:
        # compute average absolute shift for first n_peaks_to_check
        shifts = []
        for pk in raw_peaks[:n_peaks_to_check]:
            window = int(fs*0.05)
            lo = max(0, pk - window)
            hi = min(len(proc_signal), pk + window)
            local = proc_signal[lo:hi]
            if len(local)==0:
                shifts.append(1.0)
                continue
            local_pk = np.argmax(np.abs(local)) + lo
            shifts.append(abs(local_pk - pk) / fs)
        mean_shift = float(np.mean(shifts))
    percent_preserved = float(np.mean(preserved))*100 if preserved else 100.0
    ok = np.mean(preserved) > 0.8 if preserved else True
    return ok, mean_shift, len(raw_peaks)

# -------------------------
# Plot helpers
# -------------------------
def save_power_plots(freqs, raw_psd, proc_psd, notch_freq, outpath):
    # full 0-60 Hz
    fig, ax = plt.subplots(2,1, figsize=(10,8))
    ax[0].semilogy(freqs[freqs<=60], raw_psd[freqs<=60], label='raw', linewidth=1.2)
    ax[0].semilogy(freqs[freqs<=60], proc_psd[freqs<=60], label='proc', linewidth=1.2)
    ax[0].axvline(notch_freq, color='r', linestyle='--', label=f'notch {notch_freq}Hz')
    if 2*notch_freq < freqs.max():
        ax[0].axvline(2*notch_freq, color='r', linestyle='--', label=f'notch {2*notch_freq}Hz')
    ax[0].set_xlim(0,60); ax[0].set_ylabel('Power'); ax[0].legend(); ax[0].grid(True)
    ax[0].set_title('PSD 0-60 Hz')

    # zoom around notch
    zlo, zhi = notch_freq-2.0, notch_freq+2.0
    mask = (freqs>=zlo) & (freqs<=zhi)
    ax[1].plot(freqs[mask], raw_psd[mask], 'b-o', label='raw')
    ax[1].plot(freqs[mask], proc_psd[mask], 'r-s', label='proc')
    ax[1].axvline(notch_freq, color='k', linestyle='--')
    ax[1].set_xlabel('Freq (Hz)'); ax[1].set_ylabel('Power'); ax[1].legend(); ax[1].grid(True)
    ax[1].set_title(f'Zoom around {notch_freq}Hz')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

def save_edge_plots(raw, proc, fs, outpath):
    n1 = int(fs*1.0)
    t = np.arange(n1)/fs
    fig, axs = plt.subplots(1,2, figsize=(12,3))
    axs[0].plot(t, raw[:n1], label='raw'); axs[0].plot(t, proc[:n1], label='proc')
    axs[0].set_title('First 1s'); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(t, raw[-n1:], label='raw'); axs[1].plot(t, proc[-n1:], label='proc')
    axs[1].set_title('Last 1s'); axs[1].legend(); axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

# -------------------------
# Main test runner
# -------------------------
def run_test():
    print("\n=== PREPROCESSING VALIDATION SUITE ===\n")

    # determine training dir from config
    training_dir = getattr(config, "TRAINING_DIR", None)
    data_dir = getattr(config, "DATA_DIR", None)
    if training_dir is None and data_dir is not None:
        training_dir = str(Path(data_dir) / "training")
    if training_dir is None:
        # fallback relative
        training_dir = str((PROJECT_ROOT / "data" / "training").resolve())

    training_dir = Path(training_dir)
    if not training_dir.exists():
        print(f"SKIP: training dir not found: {training_dir}")
        return

    # find first EDF+XML pair
    edf_files = sorted([p for p in training_dir.glob("*.edf")])
    if not edf_files:
        print(f"SKIP: no .edf files in {training_dir}")
        return

    chosen = edf_files[0]
    xml_candidate = chosen.with_suffix('.xml')
    if not xml_candidate.exists():
        # try uppercase/lowercase variants
        alt = list(training_dir.glob(chosen.stem + ".*"))
        xml_candidate = None
        for a in alt:
            if a.suffix.lower().startswith('.xml'):
                xml_candidate = a
                break
    if xml_candidate is None or not xml_candidate.exists():
        print(f"SKIP: Matching XML not found for {chosen}")
        return

    print(f"Using EDF: {chosen}")
    print(f"Using XML: {xml_candidate}")

    # load data
    try:
        data, labels, channel_info = load_training_data(str(chosen), str(xml_candidate))
    except Exception as e:
        print("ERROR loading recording:", type(e).__name__, e)
        return

    record_id = channel_info.get("record_id", chosen.stem)
    print(f"\nLoaded recording: {record_id}")
    print("Channel info (summary):")
    for k in sorted(channel_info.keys()):
        print(f"   {k} : {channel_info[k]}")
    print()

    # build preproc cfg: prefer config.PREPROCESS if present
    cfg_source = None
    if hasattr(config, "PREPROCESS"):
        cfg_source = config.PREPROCESS
        cfg = coerce_preproc_config(config)
        print("Using PREPROCESS from config module.")
    else:
        cfg = PreprocConfig()
        print("Using default PreprocConfig.")

    # force debug plots and outputs path
    cfg.debug_plots = True
    out_base = getattr(config, "OUTPUT_DIR", "./outputs")
    cfg.outputs_dir = str(Path(out_base) / "preproc_validation" / record_id)
    os.makedirs(cfg.outputs_dir, exist_ok=True)
    print(f"Debug outputs -> {cfg.outputs_dir}\n")

    # run preprocessing
    print("Running preprocess(...) ...")
    proc = preprocess(data, cfg, channel_info=channel_info)
    print("Preprocessing done.\n")

    # select first epoch/channel for checks
    fs = float(channel_info.get('eeg_fs', channel_info.get('global_fs', 125.0)))
    ep_idx = 0; ch_idx = 0
    raw_sig = data['eeg'][ep_idx, ch_idx, :].astype(float)
    proc_sig = proc['eeg'][ep_idx, ch_idx, :].astype(float)

    # Mean check
    mean_ok, raw_mean, proc_mean = mean_check(raw_sig, proc_sig, threshold=1.0)
    print(f"[Mean Check] raw_mean={raw_mean:.4e}, pre_mean={proc_mean:.4e} -> {'PASS' if mean_ok else 'FAIL'}")

    # Delta preservation
    delta_ok, ratio, (freqs, raw_psd, proc_psd) = delta_preservation_check(raw_sig, proc_sig, fs, low=0.5, high=4.0, min_ratio=0.8)
    print(f"[Delta Preservation] preservation_ratio={ratio:.3f} -> {'PASS' if delta_ok else 'FAIL'}")

    # Save PSD and edge plots
    ps_out = Path(cfg.outputs_dir) / f"{record_id}_ch{ch_idx}_ep{ep_idx}_ps.png"
    save_power_plots(freqs, raw_psd, proc_psd, cfg.notch_freq, str(ps_out))
    edges_out = Path(cfg.outputs_dir) / f"{record_id}_ch{ch_idx}_ep{ep_idx}_edges.png"
    save_edge_plots(raw_sig, proc_sig, fs, str(edges_out))
    print(f"Saved power spectrum comparison to: {ps_out}")
    print(f"Saved edge effects plot to: {edges_out}")

    # Phase shift check
    phase_ok, mean_shift, n_peaks = phase_shift_check(raw_sig, proc_sig, fs)
    print(f"[Phase Shift] peaks_found={n_peaks}, mean_shift={mean_shift:.4f}s -> {'PASS' if phase_ok else 'FAIL'}")

    # summary
    print("\n=== SUMMARY ===")
    print(f"Mean check:           {'PASS' if mean_ok else 'FAIL'}")
    print(f"Delta preservation:   {'PASS' if delta_ok else 'FAIL'}")
    print(f"Power spectrum plot:  Saved -> {ps_out}")
    print(f"Edge effects plot:    Saved -> {edges_out}")
    print(f"Phase-shift check:    {'PASS' if phase_ok else 'FAIL'}")
    print(f"\nDiagnostics saved to: {cfg.outputs_dir}")

    if not (mean_ok and delta_ok):
        print("\nOne or more critical validations failed.")
        print("Hints:")
        print(" - Try lowering hp cutoff (cfg.highpass) toward 0.05-0.15 Hz to preserve delta band.")
        print(" - Try different padtype ('odd' vs None) or padlen in PreprocConfig.")
        print(" - If hardware already highpassed at 0.15Hz, consider hp ~0.1 or skip HPF.")
    else:
        print("\nAll required validations PASSED.")

if __name__ == "__main__":
    run_test()
