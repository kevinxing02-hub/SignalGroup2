#!/usr/bin/env python3
"""
compare_zscore_strategies.py

Compare multiple normalization strategies and print diagnostics:
 - no normalization
 - per-subject z-score (current)
 - per-subject mean removal only (subtract mean, keep global std)
 - per-subject robust scaling (median / IQR)

This script reuses your project's modules:
  - load_all_training_data
  - preprocess
  - extract_features
  - select_features
  - train_classifier
  - visualize_results (optional)
  - save_cache / load_cache
"""

import os
import json
import time
import numpy as np
import traceback
from collections import defaultdict

import config
from src.data_loader import load_all_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.classification import train_classifier
from src.utils import load_cache, save_cache
from visualization_helpers import plot_overall_performance

OUTDIR = os.path.join(getattr(config, "OUTPUT_DIR", "outputs"), "compare_zscore")
os.makedirs(OUTDIR, exist_ok=True)

def subject_zscore(X, record_ids, eps=1e-9):
    X = np.asarray(X, dtype=float)
    r = np.asarray(record_ids)
    out = np.empty_like(X)
    unique, inverse = np.unique(r, return_inverse=True)
    for i, uid in enumerate(unique):
        mask = (inverse == i)
        sub = X[mask]
        mean = np.mean(sub, axis=0)
        std = np.std(sub, axis=0)
        std_safe = np.where(std < eps, eps, std)
        out[mask] = (sub - mean) / std_safe
    return out

def subject_mean_remove(X, record_ids):
    X = np.asarray(X, dtype=float)
    r = np.asarray(record_ids)
    out = np.empty_like(X)
    unique, inverse = np.unique(r, return_inverse=True)
    for i, uid in enumerate(unique):
        mask = (inverse == i)
        sub = X[mask]
        mean = np.mean(sub, axis=0)
        out[mask] = (sub - mean)
    return out

def subject_robust_scale(X, record_ids, eps=1e-9):
    X = np.asarray(X, dtype=float)
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

def run_pipeline(features, labels, record_ids, apply_normalization=None):
    """
    apply_normalization: None | 'zscore' | 'mean_remove' | 'robust'
    returns: dict with selected_features, model, loso_aggregated, diagnostics
    """
    diag = {}
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)

    # copy features to avoid in-place changes
    X_work = X.copy()

    # Optionally apply normalization BEFORE select_features? we follow your current pipeline:
    # Use the corrected order: select stage1+2 globally, then apply normalization to reduced features,
    # then stage3 MI selection. But here we'll simply call your select_features which does stage1-3.
    # To mirror your existing run, we'll call select_features on raw data, THEN apply normalization to selected_features
    # (this is the simplest comparison directly matching your earlier "no zscore vs zscore after stage2" runs).
    # However to be consistent with our earlier fix, we'll apply normalization AFTER stage1+2 is performed inside select_features.
    # If select_features can't be controlled, we will run it on raw X_work and then normalize selected_features.

    selected_features, selector_obj = select_features(X_work, y, config)
    diag['selected_shape_pre_norm'] = selected_features.shape

    # Apply normalization on selected features if requested
    if apply_normalization is None:
        X_sel = selected_features
    elif apply_normalization == 'zscore':
        X_sel = subject_zscore(selected_features, record_ids)
    elif apply_normalization == 'mean_remove':
        X_sel = subject_mean_remove(selected_features, record_ids)
    elif apply_normalization == 'robust':
        X_sel = subject_robust_scale(selected_features, record_ids)
    else:
        raise ValueError("Unknown apply_normalization: " + str(apply_normalization))

    diag['applied_normalization'] = apply_normalization

    # Train classifier (this should run LOSO inside train_classifier)
    model, scaler, loso_agg = train_classifier(X_sel, y, config)
    diag['model_present'] = model is not None
    diag['loso_aggregated_available'] = loso_agg is not None

    return {
        "selected_features_raw": selected_features,
        "selected_features_after_norm": X_sel,
        "selector_obj": selector_obj,
        "model": model,
        "scaler": scaler,
        "loso_aggregated": loso_agg,
        "diagnostics": diag
    }

def summarize_loso(loso_agg):
    # loso_agg should contain per-sample test_labels and test_predictions arrays (as in your pipeline)
    # We'll compute per-subject F1s if loso_agg contains fold-wise info, else compute global metrics
    from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
    out = {}
    try:
        y_true = loso_agg['test_labels']
        y_pred = loso_agg['test_predictions']
        # global metrics
        out['accuracy'] = float(np.mean(y_true == y_pred))
        # macro F1
        out['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    except Exception as e:
        out['error'] = str(e)
    return out

def compare_runs(run_dicts, record_ids):
    """
    run_dicts: dict name -> result (from run_pipeline)
    Prints diagnostics, per-subject deltas, feature overlap and importance differences.
    """
    results = {}
    # collect selected original feature indices if selector_obj exposes get_support
    for name, r in run_dicts.items():
        sel = r['selector_obj']
        try:
            idx = sel.get_support(indices=True)
        except Exception:
            # fallback: try attribute selected_indices
            idx = getattr(sel, "selected_indices", None)
            if idx is None:
                idx = np.arange(r['selected_features_raw'].shape[1])
        results[name] = {
            "selected_indices": np.asarray(idx, dtype=int),
            "selected_shape": r['selected_features_raw'].shape,
            "loso_agg": r['loso_aggregated']
        }

    # compute pairwise overlaps
    names = list(results.keys())
    overlaps = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = set(results[names[i]]['selected_indices'].tolist())
            b = set(results[names[j]]['selected_indices'].tolist())
            inter = a.intersection(b)
            overlaps[f"{names[i]} vs {names[j]}"] = {
                "a_count": len(a),
                "b_count": len(b),
                "intersection_count": len(inter),
                "intersection_fraction_a": len(inter)/len(a) if len(a)>0 else 0,
                "intersection_fraction_b": len(inter)/len(b) if len(b)>0 else 0
            }

    # Summarize LOSO global metrics
    loso_summary = {}
    for name in names:
        loso_summary[name] = summarize_loso(results[name]['loso_agg'])

    # Per-subject F1 delta (requires loso_aggregated to have per-sample subject ids)
    per_subject = {}
    try:
        # loso_agg must include mapping of epoch->record_id or we have config.record_ids
        for name in names:
            la = results[name]['loso_agg']
            y_true = np.asarray(la['test_labels'])
            y_pred = np.asarray(la['test_predictions'])
            # We assume the order of y_true matches the original features order; use record_ids to split
            subj_ids = np.asarray(record_ids)
            unique = np.unique(subj_ids)
            subj_metrics = {}
            from sklearn.metrics import f1_score
            for uid in unique:
                mask = (subj_ids == uid)
                if np.sum(mask) == 0:
                    continue
                yt = y_true[mask]
                yp = y_pred[mask]
                f1 = float(f1_score(yt, yp, average='macro', zero_division=0))
                subj_metrics[str(uid)] = {"f1_macro": f1, "n_epochs": int(mask.sum())}
            per_subject[name] = subj_metrics
    except Exception:
        per_subject = None

    output = {
        "overlaps": overlaps,
        "global_loso": loso_summary,
        "per_subject": per_subject
    }
    # Save to disk
    outfn = os.path.join(OUTDIR, f"compare_results_{int(time.time())}.json")
    with open(outfn, "w") as f:
        json.dump(output, f, indent=2)
    print("Saved comparison summary to:", outfn)

    # pretty print summary
    print("\n=== Global LOSO summary ===")
    for name in names:
        print(f"{name}: {loso_summary[name]}")
    print("\n=== Feature overlaps ===")
    for k, v in overlaps.items():
        print(k, v)

    if per_subject is not None:
        print("\n=== Per-subject F1 (sample) ===")
        # show table of f1 by subject across runs
        subjects = sorted(list({s for name in per_subject for s in per_subject[name].keys()}))
        header = ["subject"] + names
        rows = []
        print("\t".join(header))
        for s in subjects:
            row = [s]
            for name in names:
                v = per_subject[name].get(s, {"f1_macro": None})
                row.append(f"{v['f1_macro']:.3f}" if v['f1_macro'] is not None else "NA")
            print("\t".join(row))

    return output

def main():
    # Load data
    use_single_recording = (config.CURRENT_ITERATION == 1)
    multi_channel_data, labels, record_ids, channel_info = load_all_training_data(config.TRAINING_DIR, use_single_recording=use_single_recording)
    print(f"Loaded {len(labels)} epochs, {len(np.unique(record_ids))} subjects")

    # Preprocess & extract features (respecting cache)
    cache_fn = f"features_iter{config.CURRENT_ITERATION}.joblib"
    features = None
    if getattr(config, "USE_CACHE", False):
        features = load_cache(cache_fn, config.CACHE_DIR)
    if features is None:
        pre = preprocess(multi_channel_data, config, channel_info=channel_info)
        features = extract_features(pre, config)
        if getattr(config, "USE_CACHE", False):
            save_cache(features, cache_fn, config.CACHE_DIR)
    print("Features shape:", features.shape)

    # Run different strategies
    runs = {}
    try:
        print("\n--- Running NO normalization ---")
        runs['no_norm'] = run_pipeline(features, labels, record_ids, apply_normalization=None)
    except Exception as e:
        print("Failed no_norm:", e, traceback.format_exc())

    try:
        print("\n--- Running subject Z-score normalization ---")
        runs['zscore'] = run_pipeline(features, labels, record_ids, apply_normalization='zscore')
    except Exception as e:
        print("Failed zscore:", e, traceback.format_exc())

    try:
        print("\n--- Running subject mean removal only ---")
        runs['mean_remove'] = run_pipeline(features, labels, record_ids, apply_normalization='mean_remove')
    except Exception as e:
        print("Failed mean_remove:", e, traceback.format_exc())

    try:
        print("\n--- Running subject robust scaling (median/IQR) ---")
        runs['robust'] = run_pipeline(features, labels, record_ids, apply_normalization='robust')
    except Exception as e:
        print("Failed robust:", e, traceback.format_exc())

    # Compare
    compare_runs(runs, record_ids)

if __name__ == "__main__":
    main()
