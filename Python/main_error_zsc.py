#!/usr/bin/env python3
"""
main.py - pipeline runner with corrected ordering and normalization toggle.

Order:
  raw_features
    -> variance_threshold (stage1)
    -> correlation_filter (stage2)
    -> subject-level normalization (config.NORMALIZATION)
    -> mutual information selection (stage3)
    -> classifier training / visualization / report

Normalization options (set in config.py):
  NORMALIZATION = 'none'   # no subject normalization
  NORMALIZATION = 'zscore' # subject mean/std z-score
  NORMALIZATION = 'mean'   # subject mean removal only
  NORMALIZATION = 'robust' # subject median / IQR robust scaling
"""

import os
import sys
import io
import time
import json
import numpy as np

import config
from src.data_loader import load_all_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.classification import train_classifier
from src.visualization import visualize_results
from src.report import generate_report
from src.utils import save_cache, load_cache
from visualization_helpers import save_iteration_performance, plot_overall_performance

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Set default output dir if not present
if not hasattr(config, 'OUTPUT_DIR'):
    config.OUTPUT_DIR = 'outputs'
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


# -------------------------
# Normalization helpers
# -------------------------
def subject_zscore(features, record_ids, eps=1e-9, verbose=False, max_report_subjects=10):
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError("features must be a 2D array (n_epochs, n_features)")
    r = np.asarray(record_ids)
    if r.shape[0] != X.shape[0]:
        raise ValueError(f"record_ids length ({r.shape[0]}) != features rows ({X.shape[0]})")
    out = np.empty_like(X)
    unique, inverse = np.unique(r, return_inverse=True)
    diagnostics = {"n_subjects": int(len(unique)), "subjects": {}}
    for i, uid in enumerate(unique):
        mask = (inverse == i)
        sub = X[mask]
        n_epochs = int(mask.sum())
        if n_epochs == 0:
            diagnostics["subjects"][str(uid)] = {"n_epochs": 0, "mean_abs_mean": None, "mean_std": None, "n_zero_std_feats": None}
            continue
        mean = np.mean(sub, axis=0)
        std = np.std(sub, axis=0)
        n_zero_std = int(np.sum(std < eps))
        std_safe = np.where(std < eps, eps, std)
        out[mask] = (sub - mean) / std_safe
        diagnostics["subjects"][str(uid)] = {
            "n_epochs": n_epochs,
            "mean_abs_mean": float(np.mean(np.abs(mean))),
            "mean_std": float(np.mean(std)),
            "n_zero_std_feats": n_zero_std
        }
    if verbose:
        print(f"[subject_zscore] subjects: {diagnostics['n_subjects']}")
        sample_keys = list(diagnostics["subjects"].keys())[:max_report_subjects]
        for k in sample_keys:
            v = diagnostics["subjects"][k]
            print(f"  Subject {k}: epochs={v['n_epochs']}, mean_abs_mean={v['mean_abs_mean']:.3g}, mean_std={v['mean_std']:.3g}, n_zero_std_feats={v['n_zero_std_feats']}")
    return out


def subject_mean_remove(features, record_ids):
    X = np.asarray(features, dtype=float)
    r = np.asarray(record_ids)
    out = np.empty_like(X)
    unique, inverse = np.unique(r, return_inverse=True)
    for i, uid in enumerate(unique):
        mask = (inverse == i)
        sub = X[mask]
        mean = np.mean(sub, axis=0)
        out[mask] = sub - mean
    return out


def subject_robust_scale(features, record_ids, eps=1e-9):
    X = np.asarray(features, dtype=float)
    r = np.asarray(record_ids)
    out = np.empty_like(X)
    unique, inverse = np.unique(r, return_inverse=True)
    for i, uid in enumerate(unique):
        mask = (inverse == i)
        sub = X[mask]
        med = np.median(sub, axis=0)
        q75 = np.percentile(sub, 75, axis=0)
        q25 = np.percentile(sub, 25, axis=0)
        iqr = q75 - q25
        iqr_safe = np.where(iqr < eps, eps, iqr)
        out[mask] = (sub - med) / iqr_safe
    return out


# -------------------------
# Simple selector stub for saving indices
# -------------------------
class SelectorStub:
    def __init__(self, support_idx):
        self._idx = np.asarray(support_idx, dtype=int)

    def get_support(self, indices=True):
        if indices:
            return self._idx
        else:
            mask = np.zeros(self._idx.max() + 1, dtype=bool)
            mask[self._idx] = True
            return mask


# -------------------------
# Stage 1: Variance thresholding
# -------------------------
def variance_thresholding(X, threshold):
    X = np.asarray(X, dtype=float)
    var = np.var(X, axis=0)
    keep_mask = var > threshold
    removed = np.where(~keep_mask)[0].tolist()
    return X[:, keep_mask], keep_mask, removed, var


# -------------------------
# Stage 2: Correlation filtering
# -------------------------
def correlation_filtering(X, corr_threshold=0.95):
    X = np.asarray(X, dtype=float)
    if X.shape[1] <= 1:
        return X, np.ones(X.shape[1], dtype=bool), []
    variances = np.var(X, axis=0)
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr)
    n_feats = corr.shape[0]
    keep = np.ones(n_feats, dtype=bool)
    removed = []
    for i in range(n_feats):
        if not keep[i]:
            continue
        for j in range(i + 1, n_feats):
            if not keep[j]:
                continue
            if abs(corr[i, j]) > corr_threshold:
                if variances[i] >= variances[j]:
                    keep[j] = False
                    removed.append(int(j))
                else:
                    keep[i] = False
                    removed.append(int(i))
                    break
    Xf = X[:, keep]
    return Xf, keep, removed


# -------------------------
# Stage 3: Mutual Information ranking (select top-k)
# -------------------------
def mutual_info_select(X, y, top_k=40, random_state=0):
    X = np.asarray(X, dtype=float)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    mi = mutual_info_classif(X, y_enc, discrete_features=False, random_state=random_state)
    n_feats = X.shape[1]
    if top_k >= n_feats:
        idx = np.argsort(-mi)
    else:
        idx = np.argsort(-mi)[:top_k]
    Xsel = X[:, idx]
    return Xsel, idx, mi


# -------------------------
# TeeOutput for logging capture
# -------------------------
class TeeOutput:
    def __init__(self, terminal, buffer):
        self.terminal = terminal
        self.buffer = buffer

    def write(self, message):
        self.terminal.write(message)
        try:
            self.terminal.flush()
        except Exception:
            pass
        self.buffer.write(message)

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.buffer.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.terminal, name)


# -------------------------
# Diagnostics saver
# -------------------------
def save_simple_diag(diag_obj, output_dir, prefix="diag"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    fn = os.path.join(output_dir, f"{prefix}_{ts}.json")
    try:
        with open(fn, "w") as f:
            json.dump(diag_obj, f, indent=2)
        print(f"[diagnostics] saved to {fn}")
    except Exception as e:
        print(f"[diagnostics] failed to save: {e}")
        fn = None
    return fn


# -------------------------
# Main pipeline
# -------------------------
def main():
    stdout_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(original_stdout, stdout_buffer)

    try:
        print("\n=== PROCESSING LOG ===")
        print(f"--- Sleep Scoring Pipeline - Iteration {config.CURRENT_ITERATION} ---")

        # STEP 1: DATA LOADING
        print("\n=== STEP 1: DATA LOADING ===")
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

        # Make record_ids available to other modules
        config.record_ids = record_ids

        # STEP 2: PREPROCESSING (with cache)
        print("\n=== STEP 2: PREPROCESSING ===")
        preprocessed_data = None
        cache_filename_preprocess = f"preprocessed_data_iter{config.CURRENT_ITERATION}.joblib"
        if getattr(config, "USE_CACHE", False):
            preprocessed_data = load_cache(cache_filename_preprocess, config.CACHE_DIR)
            if preprocessed_data is not None:
                print("Loaded preprocessed data from cache")
                if isinstance(preprocessed_data, dict) and 'eeg' in preprocessed_data:
                    cached_epochs = preprocessed_data['eeg'].shape[0]
                    if cached_epochs != len(labels):
                        print(f"WARNING: Cached preprocessed data ({cached_epochs} epochs) does not match current labels ({len(labels)} epochs). Clearing cache.")
                        cache_file = os.path.join(config.CACHE_DIR, cache_filename_preprocess)
                        if os.path.exists(cache_file):
                            os.remove(cache_file)
                        preprocessed_data = None

        if preprocessed_data is None:
            preprocessed_data = preprocess(multi_channel_data, config, channel_info=channel_info)
            if isinstance(preprocessed_data, dict) and 'eeg' in preprocessed_data:
                print(f"Preprocessed EEG shape: {preprocessed_data['eeg'].shape}")
                if preprocessed_data['eeg'].shape[0] != len(labels):
                    raise ValueError(f"Preprocessing mismatch: {preprocessed_data['eeg'].shape[0]} epochs vs {len(labels)} labels.")
            if getattr(config, "USE_CACHE", False):
                save_cache(preprocessed_data, cache_filename_preprocess, config.CACHE_DIR)
                print("Saved preprocessed data to cache")

        # STEP 3: FEATURE EXTRACTION (with cache)
        print("\n=== STEP 3: FEATURE EXTRACTION ===")
        features = None
        cache_filename_features = f"features_iter{config.CURRENT_ITERATION}.joblib"
        if getattr(config, "USE_CACHE", False):
            features = load_cache(cache_filename_features, config.CACHE_DIR)
            if features is not None:
                print("Loaded features from cache")

        if features is None:
            features = extract_features(preprocessed_data, config)
            print(f"Extracted features shape: {features.shape}")
            if features.shape[1] == 0:
                print("WARNING: No features extracted.")
            if features.shape[0] != len(labels):
                raise ValueError(f"Feature extraction mismatch: {features.shape[0]} samples vs {len(labels)} labels.")
            if getattr(config, "USE_CACHE", False):
                save_cache(features, cache_filename_features, config.CACHE_DIR)
                print("Saved features to cache")
        else:
            if features.shape[0] != len(labels):
                print("Cached features mismatch. Re-extracting.")
                cache_file = os.path.join(config.CACHE_DIR, cache_filename_features)
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                features = extract_features(preprocessed_data, config)
                if getattr(config, "USE_CACHE", False):
                    save_cache(features, cache_filename_features, config.CACHE_DIR)

        print("\n=== DEBUG: Raw features shape ===", features.shape)

        # Quick diagnostics (paste after extract_features)
        v = np.var(features, axis=0)
        print("Feature count:", features.shape[1])
        print("Variance - min:", v.min(), "median:", np.median(v), "max:", v.max())
        print("Counts: var < 1e-12:", np.sum(v < 1e-12), ", <1e-8:", np.sum(v < 1e-8), ", <1e-6:", np.sum(v < 1e-6))
        print("Any NaNs in features?", np.isnan(features).any())
        print("Feature means (first 10):", np.mean(features, axis=0)[:10])
        print("Feature stds (first 10):", np.std(features, axis=0)[:10])
        # Per-feature per-subject zero-std counts
        def per_feature_zero_counts(X, record_ids, eps=1e-12):
            X = np.asarray(X)
            r = np.asarray(record_ids)
            unique, inverse = np.unique(r, return_inverse=True)
            counts = np.zeros(X.shape[1], dtype=int)
            for i in range(len(unique)):
                mask = (inverse == i)
                counts += (np.std(X[mask], axis=0) < eps)
            return counts
        counts = per_feature_zero_counts(features, record_ids)
        print("Mean number of subjects where a feature is zero-std:", counts.mean())
        print("Number of features zero-std in >=50% subjects:", np.sum(counts >= (len(np.unique(record_ids)) / 2)))
##
        # alignment check
        if len(record_ids) != features.shape[0]:
            raise RuntimeError(f"Alignment error: record_ids length ({len(record_ids)}) != features rows ({features.shape[0]}).")

        # -------------------------
        # FEATURE SELECTION - STAGE 1 & 2 (on raw features)
        # -------------------------
        print("\n=== FEATURE SELECTION: stage 1 (variance) & stage 2 (correlation) ===")
        # thresholds from config (or defaults)
        var_ratio = getattr(config, "VARIANCE_THRESHOLD_RATIO", 1e-4)
        corr_threshold = getattr(config, "CORRELATION_THRESHOLD", 0.95)
        # compute threshold relative to max variance (consistent with original implementation)
        variances_all = np.var(features, axis=0)
        max_var = np.max(variances_all) if variances_all.size > 0 else 0.0
        threshold = max(max_var * var_ratio, getattr(config, "ABS_VARIANCE_THRESHOLD", 1e-12))

        X_stage1, keep_mask_stage1, removed_stage1, var = variance_thresholding(features, threshold=threshold)
        print(f"  Stage1 variance: removed {len(removed_stage1)} features, remaining {X_stage1.shape[1]}")

        X_stage2, keep_mask_stage2, removed_stage2 = correlation_filtering(X_stage1, corr_threshold=corr_threshold)
        print(f"  Stage2 correlation: removed {len(removed_stage2)} features, remaining {X_stage2.shape[1]}")

        # map kept indices back to original feature indices
        kept_idx_stage1 = np.where(keep_mask_stage1)[0]
        kept_idx_stage2 = kept_idx_stage1[np.where(keep_mask_stage2)[0]]
        print(f"  After stage2 final features count: {len(kept_idx_stage2)} (sample: {kept_idx_stage2[:10].tolist()})")

        if len(kept_idx_stage2) == 0:
            raise RuntimeError("No features remain after variance+correlation filtering. Check thresholds.")

        # -------------------------
        # Normalization toggle (apply on reduced features)
        # -------------------------
        print("\n=== Applying subject-level normalization on reduced feature set ===")
        X_reduced = features[:, kept_idx_stage2]  # (n_samples, n_kept)
        norm_method = getattr(config, "NORMALIZATION", "zscore")  # 'none','zscore','mean','robust'
        if norm_method == "none":
            X_reduced_norm = X_reduced.copy()
            print("Normalization: none (using raw reduced features)")
        elif norm_method == "zscore":
            X_reduced_norm = subject_zscore(X_reduced, record_ids, eps=getattr(config, "ZSCORE_EPS", 1e-9), verbose=True)
            print("Normalization: subject z-score")
        elif norm_method == "mean":
            X_reduced_norm = subject_mean_remove(X_reduced, record_ids)
            print("Normalization: subject mean removal")
        elif norm_method == "robust":
            X_reduced_norm = subject_robust_scale(X_reduced, record_ids, eps=getattr(config, "ZSCORE_EPS", 1e-9))
            print("Normalization: subject robust scaling (median/IQR)")
        else:
            raise ValueError("Unknown NORMALIZATION in config: " + str(norm_method))

        # -------------------------
        # STAGE 3: MI selection (on normalized reduced features)
        # -------------------------
        print("\n=== FEATURE SELECTION: stage 3 (mutual information ranking) ===")
        top_k = getattr(config, "TOP_K_FEATURES", 40)
        X_selected, mi_idx_rel, mi_scores = mutual_info_select(X_reduced_norm, labels, top_k=top_k, random_state=getattr(config, "RANDOM_STATE", 0))
        selected_original_indices = kept_idx_stage2[mi_idx_rel]
        print(f"  Selected top-{min(top_k, X_reduced.shape[1])} features -> final count {X_selected.shape[1]}")

        selector_obj = SelectorStub(selected_original_indices)
        selected_features = X_selected  # features to feed classifier

        # -------------------------
        # STEP 5: CLASSIFICATION
        # -------------------------
        print("\n=== STEP 5: CLASSIFICATION ===")
        if selected_features.shape[1] > 0:
            model, scaler, loso_aggregated = train_classifier(selected_features, labels, config)
            print(f"Trained {getattr(config, 'CLASSIFIER_TYPE', 'classifier')} classifier")
        else:
            print("WARNING: Cannot train classifier - no features available!")
            model = None
            scaler = None
            loso_aggregated = None

        if model is not None:
            sel_indices_list = selected_original_indices.tolist()
            model_bundle = {
                "model": model,
                "scaler": scaler,
                "feature_selector": selector_obj,
                "selected_indices": sel_indices_list,
                "normalization": norm_method,
                "kept_stage1_indices": kept_idx_stage1.tolist(),
                "kept_stage2_indices": kept_idx_stage2.tolist(),
            }
            model_cache_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
            save_cache(model_bundle, model_cache_filename, config.CACHE_DIR)
            print(f"Saved trained model, scaler and selector to cache: {model_cache_filename}")

        # -------------------------
        # STEP 6: VISUALIZATION
        # -------------------------
        print("\n=== STEP 6: VISUALIZATION ===")
        visuals_dir = os.path.join(config.OUTPUT_DIR if hasattr(config, 'OUTPUT_DIR') else '.', f"visuals_iter{config.CURRENT_ITERATION}")
        os.makedirs(visuals_dir, exist_ok=True)
        if model is not None:
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
            print("Skipping visualize_results - no trained model")

        # -------------------------
        # STEP 7: REPORT GENERATION
        # -------------------------
        print("\n=== STEP 7: PROCESSING LOG & REPORT GENERATION ===")
        sys.stdout = original_stdout
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

        # -------------------------
        # STEP 888: SAVE PER-ITERATION METRICS & PLOT OVERALL
        # -------------------------
        os.makedirs(visuals_dir, exist_ok=True)
        cluster_name = getattr(config, 'PICTURE_CLUSTER_NAME', None) or 'default'
        try:
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
            print(f"WARNING: Failed to save iteration performance: {e}")

        try:
            overall_path = plot_overall_performance(
                output_dir=visuals_dir,
                cluster_name=cluster_name,
                save_png=True,
                show_fig=False
            )
            print(f"Overall performance plot generated: {overall_path}")
        except FileNotFoundError:
            print("No overall performance files found to plot (this is normal on the first run).")
        except Exception as e:
            print(f"WARNING: Failed to generate overall performance plot: {e}")

    except Exception as e_main:
        try:
            sys.stdout = original_stdout
        except Exception:
            pass
        print(f"[ERROR] Pipeline failed: {e_main}")
        try:
            log_text = stdout_buffer.getvalue()
            ts = time.strftime("%Y%m%d_%H%M%S")
            ln = os.path.join(config.OUTPUT_DIR, f"pipeline_error_log_{ts}.txt")
            with open(ln, "w", encoding="utf-8") as f:
                f.write(log_text)
            print(f"[ERROR] Captured pipeline log saved to: {ln}")
        except Exception as e_log:
            print(f"[ERROR] Failed to save pipeline log: {e_log}")
        raise

    finally:
        try:
            sys.stdout = original_stdout
        except Exception:
            pass


if __name__ == "__main__":
    main()
