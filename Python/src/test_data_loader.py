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

# --------------- Utilities ----------------------

def _sanitize_filename(s: str) -> str:
    """Make a safe filename from an arbitrary string."""
    keep = (" ", ".", "_", "-")
    out = "".join(c if c.isalnum() or c in keep else "_" for c in s)
    # shorten long names a little
    if len(out) > 200:
        out = out[:200]
    return out

# --------------- Plotting ----------------------

def plot_and_save_examples(record_id, multi_data, info):
    """
    Plot waveform and PSD examples (first epoch) and save to OUTDIR.
    Ensures labels/axis are not clipped by using tight_layout/bbox_inches.
    """
    if not HAS_PLOTTING:
        return

    # Ensure record_id exists for filenames
    r_id = str(record_id) if record_id is not None else "record"
    r_id = _sanitize_filename(r_id)

    for grp, arr in multi_data.items():
        if arr is None or arr.size == 0:
            continue

        fs = info.get(f"{grp}_fs", info.get("global_fs", 125))
        # Expect shape: (n_epochs, n_ch, n_samp)
        try:
            n_epochs, n_ch, n_samp = arr.shape
        except Exception:
            # Be defensive if shape is different
            data = np.asarray(arr)
            if data.ndim == 2:
                # (n_ch, n_samp) treat as single epoch
                n_epochs = 1
                n_ch, n_samp = data.shape
                arr = data.reshape((1, n_ch, n_samp))
            else:
                # fallback
                continue

        t = (np.arange(n_samp) / float(fs))

        # only plot first epoch
        data_epoch = arr[0]

        # limit channels to a reasonable number for plotting
        max_plot_ch = min(n_ch, 6)

        # ---------- Waveform ----------
        fig, ax = plt.subplots(figsize=(10, 3))
        for ch in range(max_plot_ch):
            ax.plot(t, data_epoch[ch], label=f"{grp.upper()} ch{ch}")
        ax.set_title(f"{r_id} - {grp.upper()} waveform (first epoch)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize="small", ncol=1)
        # give a little extra space so xlabel/legend won't be clipped
        plt.subplots_adjust(bottom=0.18, right=0.95)
        fig.tight_layout()
        fp = OUTDIR / f"{r_id}_{grp}_wave.png"
        try:
            fig.savefig(fp, dpi=150, bbox_inches="tight", pad_inches=0.08)
            print(f"  Saved waveform → {fp}")
        finally:
            plt.close(fig)

        # ---------- PSD ----------
        fig, ax = plt.subplots(figsize=(10, 3))
        for ch in range(max_plot_ch):
            try:
                f, Pxx = welch(data_epoch[ch], fs=fs)
                ax.semilogy(f, Pxx, label=f"{grp.upper()} ch{ch}")
            except Exception as e:
                # If Welch fails for some channel, skip it
                print(f"    Warning: PSD failed for {grp} ch{ch}: {e}")
                continue
        ax.set_title(f"{r_id} - {grp.upper()} PSD (first epoch)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.legend(loc="upper right", fontsize="small", ncol=1)
        plt.subplots_adjust(bottom=0.18, right=0.95)
        fig.tight_layout()
        fp = OUTDIR / f"{r_id}_{grp}_psd.png"
        try:
            fig.savefig(fp, dpi=150, bbox_inches="tight", pad_inches=0.08)
            print(f"  Saved PSD → {fp}")
        finally:
            plt.close(fig)


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

    plot_and_save_examples(info.get("record_id", edf.stem), data, info)

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
    for k, v in data.items():
        try:
            shape = v.shape
        except Exception:
            shape = str(type(v))
        print(f"  {k}: {shape}")

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
