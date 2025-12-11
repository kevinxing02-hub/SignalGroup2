"""
Enhanced Test script for data loader functionality.
Now uses config.DATA_DIR / config.TRAINING_DIR, so paths are always correct.
"""

import os
import sys
from pathlib import Path
import numpy as np

# ------------------------------------------------------------
# 1) Import config and data_loader using project-relative paths
# ------------------------------------------------------------
# Assume script is inside project/Python/tests/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
from data_loader import load_training_data, load_all_training_data
from xml_parser import parse_xml_annotations

# ------------------------------------------------------------
# 2) Load data directory from config
# ------------------------------------------------------------
DATA_DIR = Path(config.DATA_DIR)        # e.g. S:/SignalGoupWork/
TRAINING_DIR = Path(config.TRAINING_DIR)  # e.g. S:/SignalGoupWork/training/

print("=== PATH SETTINGS ===")
print(f"DATA_DIR     = {DATA_DIR}")
print(f"TRAINING_DIR = {TRAINING_DIR}")
print("=====================\n")

# ------------------------------------------------------------
# 3) Optional plotting
# ------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

OUTDIR = PROJECT_ROOT / "outputs" / "loader_visuals"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------- Helper printing ------------------

def safe_print_channel_info(info):
    print("  Channel info keys:", sorted(info.keys()))
    for grp in ['eeg', 'eog', 'emg', 'other']:
        names = info.get(f"{grp}_names")
        fs = info.get(f"{grp}_fs")
        orig = info.get(f"{grp}_orig_fs")
        res = info.get(f"{grp}_resampled")
        se = info.get(f"{grp}_samps_per_epoch")
        if names:
            print(f"    {grp.upper()}: fs={fs}, orig_fs={orig}, resampled={res}, samp/epoch={se}, names={names}")

# --------------- Plotting ----------------------

def plot_and_save_examples(record_id, multi_data, info):
    if not HAS_PLOTTING:
        return

    for grp, arr in multi_data.items():
        if arr is None or arr.size == 0:
            continue

        fs = info.get(f"{grp}_fs", info.get("global_fs", 125))
        n_epochs, n_ch, n_samp = arr.shape
        t = (np.arange(n_samp) / fs)

        # only plot first epoch
        data = arr[0]

        # waveform
        plt.figure(figsize=(10, 3))
        for ch in range(min(n_ch, 3)):
            plt.plot(t, data[ch], label=f"{grp.upper()} ch{ch}")
        plt.title(f"{record_id} - {grp.upper()} waveform (first epoch)")
        plt.xlabel("Time (s)")
        plt.legend()
        fp = OUTDIR / f"{record_id}_{grp}_wave.png"
        plt.savefig(fp)
        plt.close()
        print(f"  Saved waveform → {fp}")

        # PSD
        plt.figure(figsize=(10, 3))
        for ch in range(min(n_ch, 3)):
            f, Pxx = welch(data[ch], fs=fs)
            plt.semilogy(f, Pxx, label=f"{grp.upper()} ch{ch}")
        plt.title(f"{record_id} - {grp.upper()} PSD (first epoch)")
        plt.xlabel("Frequency (Hz)")
        plt.legend()
        fp = OUTDIR / f"{record_id}_{grp}_psd.png"
        plt.savefig(fp)
        plt.close()
        print(f"  Saved PSD → {fp}")

# ------------------------------------------------------------
# TEST 1 — XML parser
# ------------------------------------------------------------
def test_xml_parser():
    print("="*60)
    print("TEST 1: XML Parser")
    print("="*60)

    xml_file = TRAINING_DIR / "R1.xml"
    if not xml_file.exists():
        print(f"SKIP: XML not found: {xml_file}")
        return

    result = parse_xml_annotations(str(xml_file))
    print("✓ XML parsed.")
    print(f"  stages: {len(result.get('stages', []))}")
    print(f"  events: {len(result.get('events', []))}\n")

# ------------------------------------------------------------
# TEST 2 — Single recording
# ------------------------------------------------------------
def test_single_recording():
    print("="*60)
    print("TEST 2: Single Recording Load")
    print("="*60)

    edf = TRAINING_DIR / "R1.edf"
    xml = TRAINING_DIR / "R1.xml"

    if not edf.exists():
        print(f"SKIP: EDF not found: {edf}")
        return

    data, labels, info = load_training_data(str(edf), str(xml))
    print(f"✓ Loaded {edf.name}")
    print(f"  Epochs: {len(labels)}")
    safe_print_channel_info(info)

    plot_and_save_examples(info["record_id"], data, info)

# ------------------------------------------------------------
# TEST 3 — Load ALL recordings
# ------------------------------------------------------------
def test_all():
    print("="*60)
    print("TEST 3: Load All Recordings")
    print("="*60)

    if not TRAINING_DIR.exists():
        print(f"SKIP: Training dir not found: {TRAINING_DIR}")
        return

    data, labels, rec_ids, info = load_all_training_data(str(TRAINING_DIR))

    print(f"✓ Loaded {len(labels)} epochs from {len(set(rec_ids))} subjects")
    for k,v in data.items():
        print(f"  {k}: {v.shape}")

    print("\nCanonical channel info:")
    safe_print_channel_info(info)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("\n=== ENHANCED DATA LOADER TEST SUITE START ===\n")
    test_xml_parser()
    test_single_recording()
    test_all()
    print("\n=== TESTS COMPLETED ===\n")


if __name__ == "__main__":
    main()
