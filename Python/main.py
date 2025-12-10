import config
from src.data_loader import load_all_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.classification_cnn import train_classifier
from src.visualization import visualize_results
from src.report import generate_report
from src.utils import save_cache, load_cache
# add this near your other imports
from visualization_helpers import save_iteration_performance, plot_overall_performance

import os
import sys
import io
import numpy as np

# set default output dir if not defined in config
if not hasattr(config, 'OUTPUT_DIR'):
    config.OUTPUT_DIR = 'outputs'
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# utils.py (add)
import numpy as np

# Subject-level normalization (per-subject z-score of features)
# Normalize features within each recording (subject) before pooling for training.
def subject_zscore(features, record_ids, eps=1e-9):
    """Z-score features per subject (rows match epochs). Returns array same shape as features."""
    X = np.asarray(features, dtype=float)
    out = np.empty_like(X)
    unique = np.unique(record_ids)
    for r in unique:
        mask = (np.array(record_ids) == r)
        sub = X[mask]
        mean = np.mean(sub, axis=0)
        std = np.std(sub, axis=0)
        std = np.where(std < eps, 1.0, std)
        out[mask] = (sub - mean) / std
    return out


class TeeOutput:
    """Class that writes to both terminal and buffer simultaneously."""

    def __init__(self, terminal, buffer):
        self.terminal = terminal
        self.buffer = buffer

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  # Flush terminal immediately
        self.buffer.write(message)

    def flush(self):
        self.terminal.flush()
        self.buffer.flush()

    def __getattr__(self, name):
        # Forward any other attributes to terminal (for compatibility)
        return getattr(self.terminal, name)


def main():
    # Create a string buffer for logging
    stdout_buffer = io.StringIO()

    # Save the original stdout
    original_stdout = sys.stdout

    # Create a Tee output that writes to both terminal and buffer
    sys.stdout = TeeOutput(original_stdout, stdout_buffer)

    print("\n=== PROCESSING LOG ===")
    print(f"--- Sleep Scoring Pipeline - Iteration {config.CURRENT_ITERATION} ---")

    # ============================================================
    # STEP 1: DATA LOADING
    # ============================================================
    print("\n=== STEP 1: DATA LOADING ===")
    # Iteration 1: EEG only (use_single_recording=True)
    # Iterations 2–4: multi-channel (use_single_recording=False)
    use_single_recording = (config.CURRENT_ITERATION == 1)

    multi_channel_data, labels, record_ids, channel_info = load_all_training_data(
        config.TRAINING_DIR,
        use_single_recording=use_single_recording
    )

    print("\nData loading summary:")
    if 'eeg' in multi_channel_data:
        print(f"  EEG: {multi_channel_data['eeg'].shape}")
    if 'eog' in multi_channel_data:
        print(f"  EOG: {multi_channel_data['eog'].shape}")
    if 'emg' in multi_channel_data:
        print(f"  EMG: {multi_channel_data['emg'].shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Unique recordings (subjects): {len(np.unique(record_ids))}")
    print(f"  Total epochs: {len(labels)}")

    # Make record_ids available to the classification module (for LOSO / grouped CV)
    config.record_ids = record_ids

    # ============================================================
    # STEP 2: PREPROCESSING
    # ============================================================
    print("\n=== STEP 2: PREPROCESSING ===")
    preprocessed_data = None
    cache_filename_preprocess = f"preprocessed_data_iter{config.CURRENT_ITERATION}.joblib"

    if config.USE_CACHE:
        preprocessed_data = load_cache(cache_filename_preprocess, config.CACHE_DIR)
        if preprocessed_data is not None:
            print("Loaded preprocessed data from cache")
            # Validate cached preprocessed data matches current dataset
            if isinstance(preprocessed_data, dict) and 'eeg' in preprocessed_data:
                cached_epochs = preprocessed_data['eeg'].shape[0]
                if cached_epochs != len(labels):
                    print(
                        f"WARNING: Cached preprocessed data ({cached_epochs} epochs) "
                        f"does not match current labels ({len(labels)} epochs)."
                    )
                    print("Clearing cache and re-running preprocessing...")
                    cache_file = os.path.join(config.CACHE_DIR, cache_filename_preprocess)
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                    preprocessed_data = None

    if preprocessed_data is None:
        preprocessed_data = preprocess(multi_channel_data, config, channel_info=channel_info)

        if isinstance(preprocessed_data, dict) and 'eeg' in preprocessed_data:
            print(f"Preprocessed EEG shape: {preprocessed_data['eeg'].shape}")
            # Validate preprocessed data matches labels
            if preprocessed_data['eeg'].shape[0] != len(labels):
                raise ValueError(
                    f"Preprocessing mismatch: preprocessed data has "
                    f"{preprocessed_data['eeg'].shape[0]} epochs but labels has "
                    f"{len(labels)} epochs. Check preprocessing code."
                )
        else:
            print("Preprocessed data ready")

        if config.USE_CACHE:
            save_cache(preprocessed_data, cache_filename_preprocess, config.CACHE_DIR)
            print("Saved preprocessed data to cache")

    # ============================================================
    # STEP 3: FEATURE EXTRACTION
    # ============================================================
    print("\n=== STEP 3: FEATURE EXTRACTION ===")
    features = None
    cache_filename_features = f"features_iter{config.CURRENT_ITERATION}.joblib"

    if config.USE_CACHE:
        features = load_cache(cache_filename_features, config.CACHE_DIR)
        if features is not None:
            print("Loaded features from cache")

    if features is None:
        features = extract_features(preprocessed_data, config)
        print(f"Extracted features shape: {features.shape}")
        if features.shape[1] == 0:
            print("WARNING: No features extracted! Students must implement feature extraction.")

        # Validate features match labels before caching
        if features.shape[0] != len(labels):
            raise ValueError(
                f"Feature extraction mismatch: features has {features.shape[0]} samples "
                f"but labels has {len(labels)} samples. Check feature extraction code."
            )

        if config.USE_CACHE:
            save_cache(features, cache_filename_features, config.CACHE_DIR)
            print("Saved features to cache")
    else:
        # Validate cached features match current labels
        if features.shape[0] != len(labels):
            print(
                f"WARNING: Cached features ({features.shape[0]} samples) do not match "
                f"current labels ({len(labels)} samples)."
            )
            print("Clearing cache and re-extracting features...")
            cache_file = os.path.join(config.CACHE_DIR, cache_filename_features)
            if os.path.exists(cache_file):
                os.remove(cache_file)
            features = extract_features(preprocessed_data, config)
            print(f"Re-extracted features shape: {features.shape}")
            if config.USE_CACHE:
                save_cache(features, cache_filename_features, config.CACHE_DIR)
                print("Saved features to cache")

    print("\n=== DEBUG: Raw features shape ===", features.shape)

    #features = subject_zscore(features, record_ids)

    # ============================================================
    # STEP 4: FEATURE SELECTION
    # ============================================================
    print("\n=== STEP 4: FEATURE SELECTION ===")
    # Validate features and labels match before feature selection
    if features.shape[0] != len(labels):
        raise ValueError(
            f"Cannot proceed: features ({features.shape[0]} samples) and labels "
            f"({len(labels)} samples) do not match. Clear cache and rerun."
        )

    # Now select_features returns (selected_features, selector_obj)
    selected_features, selector_obj = select_features(features, labels, config)
    print(f"Selected features shape: {selected_features.shape}")

    # keep selector_obj available for saving with the model bundle
    # store it somewhere local so we can save in the model bundle below
    selected_feature_selector = selector_obj

    # Validate feature selection did not change the number of samples
    if selected_features.shape[0] != len(labels):
        raise ValueError(
            f"Feature selection error: selected_features ({selected_features.shape[0]} samples) "
            f"and labels ({len(labels)} samples) do not match."
        )

    # ============================================================
    # STEP 5: CLASSIFICATION
    # ============================================================
    print("\n=== STEP 5: CLASSIFICATION ===")
    if selected_features.shape[1] > 0:
        # model, scaler = train_classifier(selected_features, labels, config)
        model, scaler, loso_aggregated = train_classifier(selected_features, labels, config)

        print(f"Trained {config.CLASSIFIER_TYPE} classifier")
    else:
        print("WARNING: Cannot train classifier - no features available!")
        print("Students must implement feature extraction first.")
        model = None
        scaler = None
        loso_aggregated = None

    if model is not None:
        # helper: try multiple attribute names / methods on selector
        def _get_selector_indices(selector):
            if selector is None:
                return None
            # If selector already exposes get_support(indices=True)
            try:
                idx = selector.get_support(indices=True)
                if idx is not None:
                    return np.asarray(idx, dtype=int)
            except Exception:
                pass
            # Try common attribute names
            for attr in ("indices_", "selected_indices", "indices", "support_", "get_support"):
                if hasattr(selector, attr) and not callable(getattr(selector, attr)):
                    try:
                        val = getattr(selector, attr)
                        if val is None:
                            continue
                        arr = np.asarray(val, dtype=int)
                        return arr
                    except Exception:
                        continue
            # last attempt: if selector has attribute 'get_support' callable that accepts indices=True
            if hasattr(selector, "get_support") and callable(selector.get_support):
                try:
                    val = selector.get_support(indices=True)
                    return np.asarray(val, dtype=int)
                except Exception:
                    pass
            return None

        sel_indices = _get_selector_indices(selected_feature_selector)
        # normalize for storage: prefer python list (safer for printing / older picklers)
        sel_indices_list = None if sel_indices is None else sel_indices.tolist()

        model_bundle = {
            "model": model,
            "scaler": scaler,
            "feature_selector": selected_feature_selector,
            "selected_indices": sel_indices_list,
        }

        model_cache_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
        save_cache(model_bundle, model_cache_filename, config.CACHE_DIR)
        print(f"Saved trained model, scaler and selector to cache: {model_cache_filename}")

    # ============================================================
    # STEP 6: VISUALIZATION
    # ============================================================
    print("\n=== STEP 6: VISUALIZATION ===")
    if model is not None:
        # Create a consistent visuals folder for this iteration and timestamp if desired
        visuals_dir = os.path.join(config.OUTPUT_DIR if hasattr(config, 'OUTPUT_DIR') else '.', f"visuals_iter{config.CURRENT_ITERATION}")
        os.makedirs(visuals_dir, exist_ok=True)

        # Optionally store some config pointers for visualization functions (hypnogram)
        # config.ANNOTATION_XML = '/path/to/annotations.xml'
        # config.SAMPLE_EDF = '/path/to/sample.edf'

        visualize_results(
            model=model,
            features=selected_features,
            labels=labels,
            config=config,
            scaler=scaler,
            loso_aggregated=loso_aggregated,
            output_dir=visuals_dir,
            cmap='viridis',
            save_all=True
        )
    else:
        # still ensure visuals_dir exists for STEP 888 if no model
        visuals_dir = os.path.join(config.OUTPUT_DIR if hasattr(config, 'OUTPUT_DIR') else '.', f"visuals_iter{config.CURRENT_ITERATION}")
        os.makedirs(visuals_dir, exist_ok=True)
        print("Skipping visualize_results - no trained model")

    # ============================================================
    # STEP 7: REPORT GENERATION
    # ============================================================
    print("\n=== STEP 7: PROCESSING LOG & REPORT GENERATION ===")

    # Restore the original stdout
    sys.stdout = original_stdout

    # Get the captured output from the buffer
    processing_log = stdout_buffer.getvalue()

    if model is not None:
        generate_report(model, selected_features, labels, config, processing_log)
    else:
        print("Skipping report - no trained model")

    print("\n" + "=" * 50)
    print("PIPELINE FINISHED")
    if model is None:
        print("WARNING: Students need to implement missing components.")
    print("=" * 50)

    # ============================================================
    # STEP 888: SAVE PER-ITERATION METRICS & PLOT OVERALL
    # ============================================================
    # ensure visuals_dir exists (already created above but keep safe)
    visuals_dir = os.path.join(config.OUTPUT_DIR if hasattr(config, 'OUTPUT_DIR') else '.', f"visuals_iter{config.CURRENT_ITERATION}")
    os.makedirs(visuals_dir, exist_ok=True)

    cluster_name = getattr(config, 'PICTURE_CLUSTER_NAME', None) or 'default'

    try:
        # Save per-iteration metrics (this function will try to compute LOSO if missing)
        save_iteration_performance(
            config=config,
            iteration=config.CURRENT_ITERATION,
            model=model,
            scaler=scaler,
            features=selected_features,
            labels=labels,
            loso_aggregated=loso_aggregated,
            output_dir=visuals_dir,
            cluster_name=cluster_name
        )
        print("Performance recorded for this iteration.")
    except Exception as e:
        # do not crash the pipeline if saving fails
        print(f"WARNING: Failed to save iteration performance: {e}")

    # generate aggregated plot for the current visuals folder (avoids recursive-search issues)
    try:
        overall_path = plot_overall_performance(
            output_dir=visuals_dir,       # point directly at folder containing the JSON(s)
            cluster_name=cluster_name,
            save_png=True,
            show_fig=False
        )
        print(f"Overall performance plot generated: {overall_path}")
    except FileNotFoundError:
        # No files yet (e.g. first run) — not an error
        print("No overall performance files found to plot (this is normal on the first run).")
    except Exception as e:
        print(f"WARNING: Failed to generate overall performance plot: {e}")


if __name__ == "__main__":
    main()
