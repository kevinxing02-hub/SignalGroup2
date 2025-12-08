import os
import traceback
import numpy as np
import config

from src.data_loader_ite2 import load_holdout_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.inference_ite2 import make_inference, generate_submission_file
from src.utils_ite2 import load_cache, save_cache


def normalize_preprocessed_output(obj):
    """
    Ensure preprocess() output is compatible with extract_features().
    - Prefer dict with 'eeg' key
    - Accept ndarray (single-channel)
    - If tuple/list: try to extract dict or ndarray
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "shape"):
        return obj

    if isinstance(obj, (tuple, list)):
        # Prefer dict with 'eeg'
        for item in obj:
            if isinstance(item, dict) and 'eeg' in item:
                return item
        # Prefer ndarray
        for item in obj:
            if hasattr(item, "shape"):
                return item
        # fallback: first element
        return obj[0]

    return obj


def _ensure_channel_info(record_info):
    """
    Ensure record_info contains minimal channel_info keys expected by preprocess.
    Returns a dict with keys like 'eeg_fs', 'eog_fs', 'emg_fs', 'epoch_length', and channel names.
    """
    if record_info is None:
        return {
            "epoch_length": 30,
            "eeg_fs": 125,
            "eog_fs": 125,
            "emg_fs": 125,
            "eeg_names": [],
            "eog_names": [],
            "emg_names": []
        }

    # copy and fill defaults
    info = dict(record_info)
    info.setdefault("epoch_length", record_info.get("epoch_length", 30))
    # sampling_rates in your loader are under 'sampling_rates'
    sr = record_info.get("sampling_rates", {})
    info.setdefault("eeg_fs", sr.get("eeg", record_info.get("eeg_fs", 125)))
    info.setdefault("eog_fs", sr.get("eog", record_info.get("eog_fs", 125)))
    info.setdefault("emg_fs", sr.get("emg", record_info.get("emg_fs", 125)))

    # channel lists
    info.setdefault("eeg_names", record_info.get("channels", []) if record_info.get("channels") else record_info.get("eeg_names", []))
    info.setdefault("eog_names", record_info.get("eog_names", []))
    info.setdefault("emg_names", record_info.get("emg_names", []))

    return info


def run_inference():
    print(f"\n--- Sleep Scoring Inference - Iteration {config.CURRENT_ITERATION} ---")

    # === Load trained model bundle ===
    model_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
    model_bundle = load_cache(model_filename, config.CACHE_DIR)

    if model_bundle is None:
        print("❌ Error: No trained model found. Run main.py first.")
        return

    model = model_bundle.get("model")
    scaler = model_bundle.get("scaler")

    print(f"Loaded trained model & scaler from: {model_filename}\n")

    # === Prepare final submission lists ===
    all_predictions = []
    all_record_ids = []
    all_epoch_ids = []

    # === Scan holdout directory ===
    print(f"Scanning holdout directory: {config.HOLDOUT_DIR}\n")
    edf_files = sorted([f for f in os.listdir(config.HOLDOUT_DIR) if f.lower().endswith(".edf")])

    if not edf_files:
        print("❌ No EDF files found in holdout directory.")
        return

    print(f"Found EDF files: {edf_files}\n")

    # ============================================================
    #   MAIN LOOP — PROCESS EACH HOLDOUT RECORD
    # ============================================================
    for edf_name in edf_files:
        record_id = os.path.splitext(edf_name)[0]
        edf_path = os.path.join(config.HOLDOUT_DIR, edf_name)

        print(f"\n=== Processing {record_id} ===")

        try:
            # ---------------------------------------------------------
            # 1. Load holdout EDF (returns tuple: (multi_channel_data, record_info))
            # ---------------------------------------------------------
            multi_channel_data, record_info = load_holdout_data(edf_path)

            if not isinstance(multi_channel_data, dict):
                raise ValueError(
                    f"load_holdout_data() returned unexpected type for data: {type(multi_channel_data)}"
                )

            # Ensure we have a usable channel_info for preprocess
            channel_info = _ensure_channel_info(record_info)

            # ---------------------------------------------------------
            # 2. Preprocessing
            # ---------------------------------------------------------
            cache_pre = f"preprocessed_{record_id}_iter{config.CURRENT_ITERATION}.joblib"
            preprocessed = load_cache(cache_pre, config.CACHE_DIR) if config.USE_CACHE else None

            if preprocessed is None:
                # pass channel_info into preprocess
                preprocessed = preprocess(multi_channel_data, config, channel_info=channel_info)
                if config.USE_CACHE:
                    save_cache(preprocessed, cache_pre, config.CACHE_DIR)

            # Normalize for feature extractor
            preprocessed = normalize_preprocessed_output(preprocessed)

            # ---------------------------------------------------------
            # 3. Feature extraction
            # ---------------------------------------------------------
            cache_feat = f"features_{record_id}_iter{config.CURRENT_ITERATION}.joblib"
            features = load_cache(cache_feat, config.CACHE_DIR) if config.USE_CACHE else None

            if features is None:
                features = extract_features(preprocessed, config)
                if config.USE_CACHE:
                    save_cache(features, cache_feat, config.CACHE_DIR)

            if not hasattr(features, "shape") or len(features.shape) != 2:
                raise RuntimeError(
                    f"Feature extraction failed for {record_id}. Got shape: {getattr(features, 'shape', None)}"
                )

            # ---------------------------------------------------------
            # 4. Scale features
            # ---------------------------------------------------------
            if scaler is not None:
                features_scaled = scaler.transform(features)
            else:
                print("⚠️  Warning: No scaler found — using raw features.")
                features_scaled = features

            # ---------------------------------------------------------
            # 5. Predict
            # ---------------------------------------------------------
            predictions = make_inference(model, features_scaled, config)
            predictions = list(predictions)

            # ---------------------------------------------------------
            # 6. Store results
            # ---------------------------------------------------------
            n_epochs = len(predictions)
            all_predictions.extend(predictions)
            all_record_ids.extend([record_id] * n_epochs)
            all_epoch_ids.extend(list(range(n_epochs)))

            print(f"✔ Finished {record_id}: {n_epochs} epochs")

        except Exception as e:
            print(f"\n❌ ERROR processing {record_id}: {e}")
            traceback.print_exc()
            print("Skipping this record and continuing...\n")
            continue

    # ============================================================
    #   STEP 7 — Generate Submission
    # ============================================================
    if not all_predictions:
        print("❌ No predictions generated. Submission aborted.")
        return

    print("\nGenerating submission file...\n")

    generate_submission_file(
        all_predictions,
        all_record_ids,
        all_epoch_ids,
        config
    )

    print("\n--- Inference Completed Successfully ---\n")


if __name__ == "__main__":
    run_inference()
