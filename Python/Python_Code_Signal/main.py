#!/usr/bin/env python3
"""
Main pipeline runner (cleaned / robustified).
Place at project root and run with `python main.py`.
"""
import os
import sys
import io
import time
from pathlib import Path
import numpy as np

# Make sure src is importable (assume main.py sits in project root)
PROJECT_ROOT = Path(__file__).resolve().parents[0]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---- project imports (may raise if missing - handled later) ----
import config

# try to import pipeline modules; if missing, we'll surface warnings but not crash early
try:
    from data_loader import load_all_training_data
except Exception as e:
    load_all_training_data = None
    print("WARNING: data_loader.load_all_training_data not available:", e)

try:
    from preprocessing import preprocess, PreprocConfig
except Exception as e:
    preprocess = None
    PreprocConfig = None
    print("WARNING: preprocessing.preprocess not available:", e)

try:
    from feature_extraction import extract_features
except Exception as e:
    extract_features = None
    print("WARNING: feature_extraction.extract_features not available:", e)

try:
    from feature_selection import select_features
except Exception as e:
    select_features = None
    print("WARNING: feature_selection.select_features not available:", e)

try:
    from classification_cnn import train_classifier
except Exception as e:
    train_classifier = None
    print("WARNING: classification_cnn.train_classifier not available:", e)

try:
    from visualization import visualize_results
except Exception as e:
    visualize_results = None
    print("WARNING: visualization.visualize_results not available:", e)

try:
    from report import generate_report
except Exception as e:
    generate_report = None
    print("WARNING: report.generate_report not available:", e)

try:
    from utils import save_cache, load_cache
except Exception as e:
    # provide minimal fallback cache (joblib-based) if utils missing
    save_cache = None
    load_cache = None
    print("WARNING: utils.save_cache/load_cache not available:", e)

# add this near your other imports (helpers for visualization storage; may be optional)
try:
    from visualization_helpers import save_iteration_performance, plot_overall_performance
except Exception:
    save_iteration_performance = None
    plot_overall_performance = None

# -----------------------------
# Fallback defaults for config
# -----------------------------
if not hasattr(config, "OUTPUT_DIR"):
    config.OUTPUT_DIR = "./outputs"
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

if not hasattr(config, "CACHE_DIR"):
    config.CACHE_DIR = "./cache"
os.makedirs(config.CACHE_DIR, exist_ok=True)

if not hasattr(config, "USE_CACHE"):
    config.USE_CACHE = False

# -----------------------------
# Local helper utilities
# -----------------------------
def subject_zscore(features, record_ids, eps=1e-9):
    """Z-score features per subject (rows match epochs)."""
    X = np.asarray(features, dtype=float)
    out = np.empty_like(X)
    unique = np.unique(record_ids)
    for r in unique:
        mask = (np.array(record_ids) == r)
        sub = X[mask]
        if sub.size == 0:
            continue
        mean = np.mean(sub, axis=0)
        std = np.std(sub, axis=0)
        std = np.where(std < eps, 1.0, std)
        out[mask] = (sub - mean) / std
    return out

class TeeOutput:
    """Writes to both terminal and buffer (used to capture processing log)."""
    def __init__(self, terminal, buffer):
        self.terminal = terminal
        self.buffer = buffer

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.buffer.write(message)

    def flush(self):
        self.terminal.flush()
        self.buffer.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)

# minimal joblib-based cache helpers if real utils missing
def _save_cache_fallback(obj, fname, cache_dir):
    import joblib
    path = os.path.join(cache_dir, fname)
    joblib.dump(obj, path)

def _load_cache_fallback(fname, cache_dir):
    import joblib
    path = os.path.join(cache_dir, fname)
    if os.path.exists(path):
        return joblib.load(path)
    return None

if save_cache is None or load_cache is None:
    save_cache = _save_cache_fallback
    load_cache = _load_cache_fallback

# -----------------------------
# Main
# -----------------------------
def main():
    stdout_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(original_stdout, stdout_buffer)

    start_time = time.time()
    print("\n=== PROCESSING LOG ===")
    print(f"--- Sleep Scoring Pipeline - Iteration {getattr(config, 'CURRENT_ITERATION', 'N/A')} ---")

    # ---------------- STEP 1: LOAD ----------------
    print("\n=== STEP 1: DATA LOADING ===")
    use_single_recording = (getattr(config, "CURRENT_ITERATION", 1) == 1)
    if load_all_training_data is None:
        print("ERROR: load_all_training_data is not available. Aborting data loading.")
        sys.stdout = original_stdout
        return

    try:
        multi_channel_data, labels, record_ids, channel_info = load_all_training_data(
            getattr(config, "TRAINING_DIR", getattr(config, "DATA_DIR", "./data") + "/training/"),
            use_single_recording=use_single_recording
        )
    except TypeError:
        # older load_all_training_data might have different signature
        multi_channel_data, labels, record_ids, channel_info = load_all_training_data()

    # Basic summary
    print("\nData loading summary:")
    if isinstance(multi_channel_data, dict):
        for k in ("eeg", "eog", "emg", "other"):
            if k in multi_channel_data and multi_channel_data[k] is not None:
                print(f"  {k.upper()}: {multi_channel_data[k].shape}")
    print(f"  Labels: {np.asarray(labels).shape}")
    print(f"  Unique recordings (subjects): {len(np.unique(record_ids))}")
    print(f"  Total epochs: {len(labels)}")

    # make record_ids accessible on config
    config.record_ids = record_ids

    # ---------------- STEP 2: PREPROCESS ----------------
    print("\n=== STEP 2: PREPROCESSING ===")
    preprocessed_data = None
    cache_filename_preprocess = f"preprocessed_data_iter{getattr(config,'CURRENT_ITERATION',1)}.joblib"

    if config.USE_CACHE:
        preprocessed_data = load_cache(cache_filename_preprocess, config.CACHE_DIR)
        if preprocessed_data is not None:
            print("Loaded preprocessed data from cache")
            # simple sanity check
            if isinstance(preprocessed_data, dict) and 'eeg' in preprocessed_data:
                if preprocessed_data['eeg'].shape[0] != len(labels):
                    print("Cached preprocessed epoch count mismatch -> will re-run preprocessing")
                    preprocessed_data = None

    if preprocessed_data is None:
        if preprocess is None:
            print("ERROR: preprocess function not available. Aborting.")
            sys.stdout = original_stdout
            return
        try:
            # if your preprocess expects config object, pass it; else assume simpler call
            preprocessed_data = preprocess(multi_channel_data, getattr(config, "PreprocConfig", config) , channel_info=channel_info)
        except TypeError:
            preprocessed_data = preprocess(multi_channel_data, config, channel_info=channel_info)

        if isinstance(preprocessed_data, dict) and 'eeg' in preprocessed_data:
            print(f"Preprocessed EEG shape: {preprocessed_data['eeg'].shape}")
            if preprocessed_data['eeg'].shape[0] != len(labels):
                print("ERROR: Preprocessing output epoch count doesn't match labels. Aborting.")
                sys.stdout = original_stdout
                return

        if config.USE_CACHE:
            save_cache(preprocessed_data, cache_filename_preprocess, config.CACHE_DIR)
            print("Saved preprocessed data to cache")

    # ---------------- STEP 3: FEATURE EXTRACTION ----------------
    print("\n=== STEP 3: FEATURE EXTRACTION ===")
    features = None
    cache_filename_features = f"features_iter{getattr(config,'CURRENT_ITERATION',1)}.joblib"

    if config.USE_CACHE:
        features = load_cache(cache_filename_features, config.CACHE_DIR)
        if features is not None:
            print("Loaded features from cache")

    if features is None:
        if extract_features is None:
            print("ERROR: extract_features not available. Aborting.")
            sys.stdout = original_stdout
            return
        features = extract_features(preprocessed_data, config)
        print(f"Extracted features shape: {features.shape}")

        if features.shape[0] != len(labels):
            print("ERROR: Feature rows don't match label count. Aborting.")
            sys.stdout = original_stdout
            return

        if config.USE_CACHE:
            save_cache(features, cache_filename_features, config.CACHE_DIR)
            print("Saved features to cache")

    print("\n=== DEBUG: Raw features shape ===", np.asarray(features).shape)

    # optional subject-level normalization (uncomment if desired)
    if getattr(config, "APPLY_SUBJECT_NORMALIZATION", False):
        features = subject_zscore(features, record_ids)
        print("Applied subject-level z-scoring to features")

    # ---------------- STEP 4: FEATURE SELECTION ----------------
    print("\n=== STEP 4: FEATURE SELECTION ===")
    if select_features is None:
        print("WARNING: select_features not available - skipping selection and using raw features")
        selected_features = features
        selector_obj = None
    else:
        # call select_features with groups if function supports it (we assume signature select_features(features, labels, config, groups=None))
        try:
            selected_features, selector_obj = select_features(features, labels, config, groups=record_ids)
        except TypeError:
            # older signature: select_features(features, labels, config)
            selected_features, selector_obj = select_features(features, labels, config)
        print(f"Selected features shape: {selected_features.shape}")

    # ---------------- STEP 5: CLASSIFICATION ----------------
    print("\n=== STEP 5: CLASSIFICATION ===")
    model = None
    scaler = None
    loso_aggregated = None

    if selected_features is None or selected_features.shape[1] == 0:
        print("WARNING: No features available - skipping classifier training")
    else:
        if train_classifier is None:
            print("WARNING: train_classifier not available - skipping classifier step")
        else:
            try:
                # train_classifier may return (model, scaler, diagnostics) OR (model, scaler)
                out = train_classifier(selected_features, labels, config)
                # normalize different return signatures
                if isinstance(out, tuple):
                    if len(out) == 3:
                        model, scaler, loso_aggregated = out
                    elif len(out) == 2:
                        model, scaler = out
                        loso_aggregated = None
                    else:
                        model = out[0]
                else:
                    model = out
                print(f"Trained classifier: {getattr(config,'CLASSIFIER_TYPE', 'unknown')}")
            except Exception as e:
                print("ERROR during train_classifier:", type(e).__name__, e)
                model = None

    # Save model bundle (if model exists)
    if model is not None:
        # best-effort extraction of selector indices for saving
        def _get_selector_indices(selector):
            if selector is None:
                return None
            try:
                return np.asarray(selector.get_support(indices=True), dtype=int)
            except Exception:
                for attr in ("indices_", "selected_indices", "indices", "support_"):
                    if hasattr(selector, attr):
                        val = getattr(selector, attr)
                        try:
                            return np.asarray(val, dtype=int)
                        except Exception:
                            continue
            return None

        sel_indices = _get_selector_indices(selector_obj)
        model_bundle = {"model": model, "scaler": scaler, "feature_selector": selector_obj, "selected_indices": None if sel_indices is None else sel_indices.tolist()}
        model_cache_filename = f"model_iter{getattr(config,'CURRENT_ITERATION',1)}.joblib"
        save_cache(model_bundle, model_cache_filename, config.CACHE_DIR)
        print(f"Saved model bundle to cache: {model_cache_filename}")

    # ---------------- STEP 6: VISUALIZATION ----------------
    print("\n=== STEP 6: VISUALIZATION ===")
    visuals_dir = os.path.join(config.OUTPUT_DIR, f"visuals_iter{getattr(config,'CURRENT_ITERATION',1)}")
    os.makedirs(visuals_dir, exist_ok=True)
    if model is not None and visualize_results is not None:
        try:
            visualize_results(model=model,
                              features=selected_features,
                              labels=labels,
                              config=config,
                              scaler=scaler,
                              loso_aggregated=loso_aggregated,
                              output_dir=visuals_dir,
                              cmap='viridis',
                              save_all=True)
        except Exception as e:
            print("WARNING: visualize_results failed:", type(e).__name__, e)
    else:
        print("Skipping visualization (missing model or visualize_results)")

    # ---------------- STEP 7: REPORT ----------------
    print("\n=== STEP 7: REPORT GENERATION ===")
    sys.stdout = original_stdout  # restore for report printing
    processing_log = stdout_buffer.getvalue()
    if model is not None and generate_report is not None:
        try:
            generate_report(model, selected_features, labels, config, processing_log)
            print("Report generated.")
        except Exception as e:
            print("WARNING: generate_report failed:", type(e).__name__, e)
    else:
        print("Skipping report generation (missing model or generate_report)")

    # ---------------- STEP 888: Save iteration performance & overall plot (best-effort)
    try:
        if save_iteration_performance is not None:
            save_iteration_performance(config=config, iteration=getattr(config,'CURRENT_ITERATION',1), model=model, scaler=scaler, features=selected_features, labels=labels, loso_aggregated=loso_aggregated, output_dir=visuals_dir, cluster_name=getattr(config,'PICTURE_CLUSTER_NAME','default'))
            print("Saved iteration performance.")
    except Exception as e:
        print("WARNING: save_iteration_performance failed:", type(e).__name__, e)

    try:
        if plot_overall_performance is not None:
            overall_path = plot_overall_performance(output_dir=visuals_dir, cluster_name=getattr(config,'PICTURE_CLUSTER_NAME','default'), save_png=True, show_fig=False)
            print(f"Overall performance plot generated: {overall_path}")
    except FileNotFoundError:
        print("No overall performance files found to plot (normal on first run).")
    except Exception as e:
        print("WARNING: plot_overall_performance failed:", type(e).__name__, e)

    # final prints
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"PIPELINE FINISHED (time: {total_time:.1f}s)")
    if model is None:
        print("WARNING: pipeline finished without a trained model - check earlier warnings/errors.")
    print("=" * 50)

    # restore stdout if not already
    try:
        sys.stdout = original_stdout
    except Exception:
        pass

if __name__ == "__main__":
    main()
