# src/feature_selection_gridsearch_runner.py
"""
Feature selection gridsearch runner (LOSO evaluation).

Saves partial results to OUTPUT_DIR/feature_selection_gridsearch/.
Resume capability: if grid_results_partial.csv exists, completed configs are skipped.
"""

import os
import sys
from pathlib import Path
import json
import hashlib
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

# project src path (assume this file is under project/src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler

# import your feature selection function (assumed to be in src/feature_selection.py)
# adjust if module name differs
try:
    from feature_selection import select_features
except Exception as e:
    raise ImportError(f"Cannot import select_features from feature_selection.py: {e}")

# ----------------------
# User / quick settings
# ----------------------
QUICK_MODE = True  # Set True to run a small quick grid for debugging
OUTDIR = Path(config.OUTPUT_DIR) / "feature_selection_gridsearch"
os.makedirs(OUTDIR, exist_ok=True)

# expected data file (created earlier by your pipeline)
DATA_FILE = Path(config.OUTPUT_DIR) / "features_dataset" / "features_dataset.npz"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_FILE}\n"
        "Please run your feature creation script to save features/labels/record_ids as a .npz "
        "with keys 'features','labels','record_ids' under outputs/features_dataset/"
    )

# ----------------------
# Helpers
# ----------------------
def params_hash(params: dict) -> str:
    """Stable short hash for a dict of parameters (sorted keys)."""
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def load_partial_results(path_csv: Path):
    if not path_csv.exists():
        return []
    try:
        df = pd.read_csv(path_csv)
        return df.to_dict(orient="records")
    except Exception:
        return []

def already_ran(params, partial_rows):
    h = params_hash(params)
    for r in partial_rows:
        # partial rows saved include original params as columns; we also saved 'params_hash'
        if "params_hash" in r and r["params_hash"] == h:
            return True
        # fallback: compare parameter key-values (best-effort)
        match = True
        for k,v in params.items():
            if k in r and str(r[k]) == str(v):
                continue
            else:
                match = False
                break
        if match:
            return True
    return False

def safe_select_features(X_train, y_train, cfg_obj, groups_for_selection=None):
    """
    Call select_features robustly and return (selected_indices, selector_obj).
    select_features may return (Xs, selector) or (selector,) or selector only; handle common cases.
    We want the selector (with .get_support or .indices_) and selected indices relative to original X_train.
    """
    res = select_features(X_train, y_train, cfg_obj, groups=groups_for_selection)
    # Normal case: (selected_features_matrix, selector_obj)
    if isinstance(res, tuple) and len(res) == 2:
        Xs_train, selector = res
        # try to get indices from selector
        try:
            idx = selector.get_support(indices=True)
            return np.asarray(idx, dtype=int), selector
        except Exception:
            # maybe selector is IndexSelector with .indices_
            idx = getattr(selector, "indices_", None)
            if idx is not None:
                return np.asarray(idx, dtype=int), selector
            # fallback: derive indices by comparing shapes (dangerous)
            if Xs_train.shape[1] <= X_train.shape[1]:
                # try to find columns by value matching (slow) - but better fallback to returning range
                # simplest safe fallback:
                return np.arange(Xs_train.shape[1], dtype=int), selector
            return np.arange(X_train.shape[1], dtype=int), selector
    else:
        # select_features returned selector-like or indices
        selector = res
        # if it's an array of indices
        if isinstance(selector, (list, np.ndarray)):
            return np.asarray(selector, dtype=int), selector
        # try to extract indices_
        idx = getattr(selector, "indices_", None)
        if idx is not None:
            return np.asarray(idx, dtype=int), selector
        # try get_support
        try:
            idx = selector.get_support(indices=True)
            return np.asarray(idx, dtype=int), selector
        except Exception:
            # give up: return all features
            n = X_train.shape[1]
            return np.arange(n, dtype=int), selector

# ----------------------
# Evaluation core
# ----------------------
def evaluate_config(X, y, groups, cfg_obj):
    """
    Run LOSO evaluation: for each held-out subject, perform feature selection on train only,
    train classifier on selected features, evaluate on test.
    Returns (mean_score, std_score, fold_scores_list)
    """
    logo = LeaveOneGroupOut()
    scores = []
    fold_scores = []
    fold_number = 0

    # iterate folds
    for train_idx, test_idx in logo.split(X, y, groups):
        fold_number += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]  # if you want selection to use groups internally (optional)

        # Optional scaling before selection & classifier
        if getattr(cfg_obj, "FEATURE_SELECTION_SCALE", False):
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Run feature selection on training subset only.
        # NOTE: we pass groups=None here because outer LOSO already isolates test subject.
        # If you want select_features to perform internal LOSO stability on train split, pass groups_train:
        #    sel_idx, selector = safe_select_features(X_train, y_train, cfg_obj, groups_for_selection=groups_train)
        sel_idx, selector = safe_select_features(X_train, y_train, cfg_obj, groups_for_selection=None)

        if sel_idx is None or len(sel_idx) == 0:
            # fallback to all features
            sel_idx = np.arange(X_train.shape[1], dtype=int)

        # transform train/test to selected features
        try:
            Xs_train = X_train[:, sel_idx]
            Xs_test = X_test[:, sel_idx]
        except Exception:
            # shapes mismatch: try safer selection via boolean mask
            mask = np.zeros(X_train.shape[1], dtype=bool)
            mask[sel_idx] = True
            Xs_train = X_train[:, mask]
            Xs_test = X_test[:, mask]

        # Train classifier (fast RF)
        clf = RandomForestClassifier(
            n_estimators=getattr(cfg_obj, "RF_N_ESTIMATORS", 100),
            max_depth=getattr(cfg_obj, "RF_MAX_DEPTH", 10),
            random_state=getattr(cfg_obj, "RANDOM_STATE", 42),
            n_jobs=1
        )
        clf.fit(Xs_train, y_train)
        y_pred = clf.predict(Xs_test)

        # macro-F1 (balanced across classes)
        score = f1_score(y_test, y_pred, average='macro', zero_division=0)
        scores.append(score)
        fold_scores.append({"fold": fold_number, "test_idx_count": len(test_idx), "score": float(score), "n_selected": int(len(sel_idx))})

    mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
    std_score = float(np.std(scores)) if len(scores) > 0 else 0.0
    return mean_score, std_score, fold_scores

# ----------------------
# Main runner
# ----------------------
def main():
    print("Loading features dataset ...")
    npz = np.load(DATA_FILE, allow_pickle=True)
    X = npz['features']
    y = npz['labels']
    groups = npz['record_ids']
    print("Loaded shapes:", X.shape, y.shape, groups.shape)

    # Build grid (can be adjusted)
    if QUICK_MODE:
        var_ratios = [1e-4, 1e-3]
        corr_threshs = [0.95]
        top_ks = [40]
        stability_thresholds = [0.6, 0.8]
        scale_options = [True, False]
    else:
        var_ratios = [1e-5, 1e-4, 1e-3]
        corr_threshs = [0.9, 0.95, 0.99]
        top_ks = [20, 40, 80]
        stability_thresholds = [0.6, 0.8, 1.0]
        scale_options = [True, False]

    grid = []
    for vr, ct, tk, st, sc in itertools.product(var_ratios, corr_threshs, top_ks, stability_thresholds, scale_options):
        grid.append({
            "VARIANCE_THRESHOLD_RATIO": vr,
            "CORRELATION_THRESHOLD": ct,
            "FEATURE_SELECTION_TOP_K": tk,
            "FEAT_STABILITY_THRESHOLD": st,
            "FEATURE_SELECTION_SCALE": sc
        })

    print(f"Total grid size: {len(grid)} (QUICK_MODE={QUICK_MODE})")

    # Load partial results to resume if present
    partial_csv = OUTDIR / "grid_results_partial.csv"
    partial_rows = load_partial_results(partial_csv)
    completed_hashes = {r.get("params_hash") for r in partial_rows if "params_hash" in r}

    results = partial_rows.copy()  # start from partial if any

    # iterate grid
    for idx, params in enumerate(grid, start=1):
        h = params_hash(params)
        if h in completed_hashes:
            print(f"[{idx}/{len(grid)}] SKIP (already done) {params}")
            continue

        print(f"\n[{idx}/{len(grid)}] Running config: {params}")
        # prepare cfg object with defaults copied from config module (only UPPER attrs)
        class CFG: pass
        cfg = CFG()
        for name in dir(config):
            if name.isupper():
                try:
                    setattr(cfg, name, getattr(config, name))
                except Exception:
                    pass
        # override with grid params
        for k, v in params.items():
            setattr(cfg, k, v)

        # Evaluate
        t0 = datetime.now()
        try:
            mean_score, std_score, fold_scores = evaluate_config(X, y, groups, cfg)
        except Exception as e:
            print(f"Config failed with exception: {type(e).__name__}: {e}")
            mean_score, std_score, fold_scores = float("nan"), float("nan"), []
        t1 = datetime.now()

        print(f"  => mean F1_macro: {mean_score:.4f} ± {std_score:.4f} (time: {(t1-t0).total_seconds():.1f}s)")

        # record row
        row = dict(params)
        row.update({
            "mean_f1": float(mean_score),
            "std_f1": float(std_score),
            "time_s": (t1-t0).total_seconds(),
            "timestamp": t1.isoformat(),
            "params_hash": h,
            "fold_details": json.dumps(fold_scores)
        })
        results.append(row)

        # save partial after each config
        df = pd.DataFrame(results)
        df.to_csv(partial_csv, index=False)
        np.savez_compressed(OUTDIR / "grid_results_partial.npz", results=results)

    # final save (rename partial -> final)
    df = pd.DataFrame(results)
    final_csv = OUTDIR / "grid_results.csv"
    df.to_csv(final_csv, index=False)
    np.savez_compressed(OUTDIR / "grid_results.npz", results=results)
    print("\nGrid search complete. Results saved to:", final_csv)
    print("You can inspect partial results:", partial_csv)

if __name__ == "__main__":
    main()
