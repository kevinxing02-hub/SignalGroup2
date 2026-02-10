# src/create_features_npz.py
import os
import sys
from pathlib import Path
import numpy as np

# assume this script is in project/src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
from data_loader import load_all_training_data, load_training_data
from preprocessing import preprocess, PreprocConfig
from feature_extraction import extract_features

OUTPUTS_DIR = Path(config.OUTPUT_DIR) / "features_dataset"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
OUTPATH = OUTPUTS_DIR / "features_dataset.npz"

def main():
    print("Creating features .npz for gridsearch ...")
    training_dir = Path(config.TRAINING_DIR)

    if not training_dir.exists():
        raise FileNotFoundError(f"Training dir not found: {training_dir}")

    # Load all training data (this may take time)
    print("Loading all training data (this may take a while)...")
    combined, labels, record_ids, channel_info = load_all_training_data(str(training_dir), epoch_length=30, target_fs=125, canonical_channel_lists=None)

    if 'eeg' not in combined:
        raise RuntimeError("No EEG found in combined data")

    # Preprocess - use default PreprocConfig or from config if you added one
    cfg = PreprocConfig()  # adjust attributes if desired
    cfg.debug_plots = False  # don't save debug plots here (set True to inspect)

    print("Running preprocessing (multi-channel)...")
    preproc = preprocess(combined, cfg, channel_info=channel_info)

    print("Extracting features (may be slow)...")
    features = extract_features(preproc, config)

    # features might be 2D numeric or object dtype; ensure numeric array
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(record_ids)

    print(f"Features shape: {X.shape}, labels: {y.shape}, groups: {groups.shape}")

    print(f"Saving to {OUTPATH} ...")
    np.savez_compressed(OUTPATH, features=X, labels=y, record_ids=groups)
    print("Saved. You can now run the gridsearch runner.")

if __name__ == "__main__":
    main()
