import config
from src.data_loader_ite2 import load_all_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection_ite2 import select_features
from src.classification import train_classifier
from src.visualization_ite2 import visualize_results
from src.report_ite2 import generate_report
from src.utils_ite2 import save_cache, load_cache
import os
import sys
import io
import numpy as np


# SHIT WEEK, ADDING CACHE STORED CODE AFTER 5.CLASSIDICTAION
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

    # 1. Load Data
    # Load ALL training recordings (R1-R10)
    # Iteration 1: EEG only (use_single_recording=True)
    # Iterations 2-4: Multi-channel EEG+EOG+EMG (use_single_recording=False)
    print("\n=== STEP 1: DATA LOADING ===")
    use_single_recording = (config.CURRENT_ITERATION == 1)
    # Refine,add 'record_ids'！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    multi_channel_data, labels, record_ids, channel_info = load_all_training_data(
        config.TRAINING_DIR,
        use_single_recording=use_single_recording
    )

    print(f"\nData loading summary:")
    if 'eeg' in multi_channel_data:
        print(f"  EEG: {multi_channel_data['eeg'].shape}")
    if 'eog' in multi_channel_data:
        print(f"  EOG: {multi_channel_data['eog'].shape}")
    if 'emg' in multi_channel_data:
        print(f"  EMG: {multi_channel_data['emg'].shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Unique recordings: {len(np.unique(record_ids))}")
    print(f"  Total epochs: {len(labels)}")

    # 2. Preprocessing
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
                    print(f"⚠️  WARNING: Cached preprocessed data ({cached_epochs} epochs) "
                          f"doesn't match current labels ({len(labels)} epochs).")
                    print("Clearing cache and re-preprocessing...")
                    cache_file = os.path.join(config.CACHE_DIR, cache_filename_preprocess)
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                    preprocessed_data = None

    if preprocessed_data is None:
        preprocessed_data = preprocess(multi_channel_data, config, channel_info=channel_info)
        # Expect dict with 'eeg'
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
            print(f"Preprocessed data ready")
        if config.USE_CACHE:
            save_cache(preprocessed_data, cache_filename_preprocess, config.CACHE_DIR)
            print("Saved preprocessed data to cache")

    # 3. Feature Extraction
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
            print("⚠️  WARNING: No features extracted! Students must implement feature extraction.")

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
            print(f"⚠️  WARNING: Cached features ({features.shape[0]} samples) don't match "
                  f"current labels ({len(labels)} samples).")
            print("Clearing cache and re-extracting features...")
            # Clear the cache file
            cache_file = os.path.join(config.CACHE_DIR, cache_filename_features)
            if os.path.exists(cache_file):
                os.remove(cache_file)
            # Re-extract features
            features = extract_features(preprocessed_data, config)
            print(f"Re-extracted features shape: {features.shape}")
            if config.USE_CACHE:
                save_cache(features, cache_filename_features, config.CACHE_DIR)
                print("Saved features to cache")
    print("\n=== DEBUG: Raw features shape ===", features.shape)

    # 4. Feature Selection
    print("\n=== STEP 4: FEATURE SELECTION ===")
    # Validate features and labels match before feature selection
    if features.shape[0] != len(labels):
        raise ValueError(
            f"Cannot proceed: features ({features.shape[0]} samples) and labels "
            f"({len(labels)} samples) don't match. Clear cache and rerun."
        )
    selected_features = select_features(features, labels, config)
    print(f"Selected features shape: {selected_features.shape}")

    # Validate feature selection didn't change number of samples
    if selected_features.shape[0] != len(labels):
        raise ValueError(
            f"Feature selection error: selected_features ({selected_features.shape[0]} samples) "
            f"and labels ({len(labels)} samples) don't match."
        )
    ### newly added ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    config.record_ids = record_ids
    ### test
    # === DEBUG: Check variance of final features ===
    '''print("\n=== DEBUG: Feature Variance Check (first 30 features) ===")
    print(np.var(selected_features, axis=0)[:30])
    print("Min variance:", np.min(np.var(selected_features, axis=0)))
    print("Max variance:", np.max(np.var(selected_features, axis=0)))'''

    '''print("\nDEBUG: record_ids distribution:")
    unique_ids, counts = np.unique(record_ids, return_counts=True)
    for uid, c in zip(unique_ids, counts):
        print(f"  Subject {uid}: {c} epochs")'''

    # 5. Classification
    print("\n=== STEP 5: CLASSIFICATION ===")
    if selected_features.shape[1] > 0:
        model, scaler = train_classifier(selected_features, labels,
                                         config)  ############################refine，add scaler
        print(f"Trained {config.CLASSIFIER_TYPE} classifier")
    else:
        print("⚠️  WARNING: Cannot train classifier - no features available!")
        print("Students must implement feature extraction first.")
        model = None

        # 5. Classification
    print("\n=== STEP 5: CLASSIFICATION ===")
    if selected_features.shape[1] > 0:
        model, scaler = train_classifier(selected_features, labels,
                                         config)  ############################refine，add scaler
        print(f"Trained {config.CLASSIFIER_TYPE} classifier")
    else:
        print("⚠️  WARNING: Cannot train classifier - no features available!")
        print("Students must implement feature extraction first.")
        model = None
    ## NEWLY ADDED
    # === SAVE MODEL & SCALER TO CACHE FOR INFERENCE ===
    if model is not None:
        model_bundle = {
            "model": model,
            "scaler": scaler
        }
        model_cache_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
        save_cache(model_bundle, model_cache_filename, config.CACHE_DIR)
        print(f"Saved trained model & scaler to cache: {model_cache_filename}")

    # 6. Visualization
    print("\n=== STEP 6: VISUALIZATION ===")
    if model is not None:
        visualize_results(
            model,
            selected_features,  # features
            labels,  # labels
            config,  # config-modulen
            scaler=scaler,  # bara används i iter 2, ok att skicka med
            loso_aggregated=None
        )
    else:
        print("Skipping visualization - no trained model")

    # 7. Report Generation
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
        print("⚠️  Students need to implement missing components!")
    print("=" * 50)


if __name__ == "__main__":
    main()