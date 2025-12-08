import os
import sys

# Add the parent directory to Python path so we can import config and src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

import config
from src.data_loader import load_all_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features


def validate_features(features: np.ndarray, labels: np.ndarray):
    """
    Run basic validation checks on the extracted features:
    - NaN / Inf check
    - Value range sanity check
    - Distribution plots per sleep stage
    """

    print("\n=== FEATURE VALIDATION ===")

    # ---------- 1. NaN / Inf check ----------
    n_nan = np.isnan(features).sum()
    n_inf = np.isinf(features).sum()

    print("\n[VALIDATION] Feature Integrity Check:")
    print(f"  NaN values: {n_nan}")
    print(f"  Inf values: {n_inf}")

    if n_nan > 0 or n_inf > 0:
        print(" WARNING: Features contain NaN or Inf values. Check feature extraction!")

    # ---------- 2. Basic range statistics ----------
    feature_min = np.min(features, axis=0)
    feature_max = np.max(features, axis=0)
    feature_mean = np.mean(features, axis=0)

    print("\n[VALIDATION] Feature Range Summary (first 10 features):")
    n_show = min(10, features.shape[1])
    for i in range(n_show):
        print(
            f"  Feature {i:02d}: "
            f"min={feature_min[i]:.3f}, "
            f"mean={feature_mean[i]:.3f}, "
            f"max={feature_max[i]:.3f}"
        )

    too_large = np.any(np.abs(features) > 1e6)
    if too_large:
        print(" WARNING: Some feature values exceed |1e6|. Check scaling / implementation.")

    # ---------- 3. Distribution plots per sleep stage ----------
    print("\n[VALIDATION] Plotting feature distributions by sleep stage...")

    stage_names = {
        0: "Wake",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM",
    }

    unique_stages = np.unique(labels)
    n_plot_features = min(4, features.shape[1])  # just first few features for visualization

    # Optionally save plots instead of only showing them
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    for feat_idx in range(n_plot_features):
        plt.figure(figsize=(8, 5))

        for stage in unique_stages:
            mask = labels == stage
            if np.sum(mask) == 0:
                continue

            plt.hist(
                features[mask, feat_idx],
                bins=50,
                alpha=0.5,
                density=True,
                label=stage_names.get(int(stage), f"Stage {stage}")
            )

        plt.title(f"Feature {feat_idx} Distribution Across Sleep Stages")
        plt.xlabel(f"Feature {feat_idx} value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        # Save and show
        plot_path = os.path.join(output_dir, f"feature_{feat_idx}_by_stage.png")
        plt.savefig(plot_path)
        print(f"  Saved plot: {plot_path}")
        plt.show()
        plt.close()


def main():
    print("\n=== FEATURE VALIDATION PIPELINE ===")
    print(f"Iteration: {config.CURRENT_ITERATION}")

    # ---------- 1. Load data ----------
    print("\nSTEP 1: DATA LOADING")
    use_single_recording = (config.CURRENT_ITERATION == 1)
    multi_channel_data, labels, record_ids, channel_info = load_all_training_data(
        config.TRAINING_DIR,
        use_single_recording=use_single_recording
    )

    print(f"  EEG shape: {multi_channel_data['eeg'].shape}")
    print(f"  Labels shape: {labels.shape}")

    # (Optional) for faster debugging: limit number of epochs
    # max_epochs = 1000
    # multi_channel_data['eeg'] = multi_channel_data['eeg'][:max_epochs]
    # labels = labels[:max_epochs]
    # print(f"  Using only first {max_epochs} epochs for validation")

    # ---------- 2. Preprocessing ----------
    print("\nSTEP 2: PREPROCESSING")
    preprocessed_data = preprocess(multi_channel_data, config, channel_info=channel_info)
    if isinstance(preprocessed_data, dict) and 'eeg' in preprocessed_data:
        print(f"  Preprocessed EEG shape: {preprocessed_data['eeg'].shape}")
    else:
        print("  Preprocessed data is not in expected dict format with 'eeg' key.")
        return

    # ---------- 3. Feature Extraction ----------
    print("\nSTEP 3: FEATURE EXTRACTION")
    features = extract_features(preprocessed_data, config)
    print(f"  Features shape: {features.shape}")

    if features.shape[0] != len(labels):
        print(
            f" WARNING: Features ({features.shape[0]}) and labels ({len(labels)}) "
            f"do not match. Validation may be misleading."
        )

    # ---------- 4. Validation ----------
    validate_features(features, labels)

    print("\n=== VALIDATION FINISHED ===")


if __name__ == "__main__":
    main()