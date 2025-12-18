# src/feature_selection.py
import numpy as np
from typing import Optional, Tuple, Dict, Any
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler

# -------------------------
# IndexSelector: minimal selector object (backwards-compatible)
# -------------------------
class IndexSelector:
    def __init__(self, indices):
        self.indices_ = np.asarray(indices, dtype=int)

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("IndexSelector.transform expects 2D array")
        return X[:, self.indices_]

    def get_support(self, indices=True):
        if indices:
            return self.indices_.copy()
        mask = np.zeros(getattr(self, "_original_n_features", self.indices_.max() + 1), dtype=bool)
        mask[self.indices_] = True
        return mask

    def _set_original_n_features(self, n):
        self._original_n_features = int(n)

# -------------------------
# Per-train selection: 3-stage pipeline used inside folds/global
# -------------------------
def _per_train_selection(X_train: np.ndarray, y_train: np.ndarray, config) -> np.ndarray:
    """
    Stage 1: Variance thresholding (relative threshold)
    Stage 2: Correlation pruning (keep highest-variance in correlated groups)
    Stage 3: Mutual information top-k selection
    Returns indices (relative to original feature numbering).
    """
    X_train = np.asarray(X_train)
    n_samples, n_features = X_train.shape

    top_k = getattr(config, "FEATURE_SELECTION_TOP_K", 40)
    var_ratio = getattr(config, "VARIANCE_THRESHOLD_RATIO", 1e-4)
    corr_thresh = getattr(config, "CORRELATION_THRESHOLD", 0.95)

    # Stage 1: variance threshold
    variances = np.var(X_train, axis=0)
    max_var = np.max(variances) if variances.size > 0 else 0.0
    threshold = max(max_var * var_ratio, 1e-12)
    try:
        vt = VarianceThreshold(threshold=threshold)
        X_s1 = vt.fit_transform(X_train)
        idx_s1 = vt.get_support(indices=True)
    except Exception:
        X_s1 = X_train.copy()
        idx_s1 = np.arange(n_features)

    if X_s1.shape[1] == 0:
        X_s1 = X_train.copy()
        idx_s1 = np.arange(n_features)

    # Stage 2: correlation pruning (keep highest-variance feature in correlated clusters)
    if X_s1.shape[1] == 1:
        X_s2 = X_s1
        idx_s2 = idx_s1
    else:
        try:
            corr = np.corrcoef(X_s1, rowvar=False)
            abs_corr = np.abs(corr)
            var_stage1 = variances[idx_s1]
            # preference order: higher variance earlier
            sorted_idx = np.argsort(var_stage1)[::-1]  # indices into idx_s1
            keep_mask = np.ones(abs_corr.shape[0], dtype=bool)
            for ii in sorted_idx:
                if not keep_mask[ii]:
                    continue
                correlated = (abs_corr[ii] > corr_thresh)
                correlated[ii] = False
                keep_mask[correlated] = False
            X_s2 = X_s1[:, keep_mask]
            idx_s2 = idx_s1[keep_mask]
            if X_s2.shape[1] == 0:
                X_s2 = X_s1.copy()
                idx_s2 = idx_s1.copy()
        except Exception:
            X_s2 = X_s1.copy()
            idx_s2 = idx_s1.copy()

    # Stage 3: mutual information top-k
    if X_s2.shape[1] <= top_k:
        final_indices = idx_s2
    else:
        try:
            mi_scores = mutual_info_classif(X_s2, y_train, discrete_features=False)
            top_local = np.argsort(mi_scores)[::-1][:top_k]
            final_indices = idx_s2[top_local]
        except Exception:
            final_indices = idx_s2

    final_indices = np.unique(np.asarray(final_indices, dtype=int))
    return final_indices

# -------------------------
# Main entrypoint
# -------------------------
def select_features(features: np.ndarray,
                    labels: np.ndarray,
                    config,
                    groups: Optional[np.ndarray] = None,
                    return_diagnostics: bool = False
                    ) -> Tuple[np.ndarray, IndexSelector, Optional[Dict[str, Any]]]:
    """
    Robust feature selection with LOSO aggregation and fallbacks.

    Returns:
      - By default (return_diagnostics==False): (selected_features, selector)
      - If return_diagnostics==True: (selected_features, selector, diagnostics_dict)

    diagnostics contains keys: selection_counts, freq, per_fold_selected, successful_folds, final_indices, mi_scores (optional), variances
    """
    print(f"\n=== Robust Feature Selection (Iteration {getattr(config, 'CURRENT_ITERATION', 'N/A')}) ===")
    X = np.asarray(features)
    y = np.asarray(labels)
    if X.ndim != 2:
        raise ValueError("features must be 2D array (n_samples x n_features)")
    n_samples, n_features = X.shape
    print(f"Initial feature count: {n_features}")

    if n_features == 0:
        sel = IndexSelector(np.arange(0))
        sel._set_original_n_features(n_features)
        diagnostics = {"selection_counts": np.zeros(0, dtype=int), "freq": np.zeros(0, dtype=float),
                       "per_fold_selected": [], "successful_folds": 0, "final_indices": np.array([], dtype=int)}
        if return_diagnostics:
            return X, sel, diagnostics
        else:
            return X, sel

    if y is None or len(y) != n_samples:
        raise ValueError("Labels must be provided and match number of feature rows")

    # Read config
    enabled = getattr(config, "FEATURE_SELECTION_ENABLED", True)
    do_scale = getattr(config, "FEATURE_SELECTION_SCALE", False)
    loso_enabled = getattr(config, "LOSO_ENABLED", True)
    stability_threshold = getattr(config, "FEAT_STABILITY_THRESHOLD", 1.0)
    min_features_iter2 = getattr(config, "FEATURE_SELECTION_MIN_FEATURES", 100)
    top_k = getattr(config, "FEATURE_SELECTION_TOP_K", 40)
    random_state = getattr(config, "RANDOM_STATE", None)

    # If disabled return identity selector
    if not enabled:
        sel = IndexSelector(np.arange(n_features))
        sel._set_original_n_features(n_features)
        diagnostics = {"selection_counts": np.zeros(n_features, dtype=int), "freq": np.ones(n_features, dtype=float),
                       "per_fold_selected": [], "successful_folds": 0, "final_indices": np.arange(n_features)}
        if return_diagnostics:
            return X, sel, diagnostics
        else:
            return X, sel

    # Iteration 2 special-case skip
    if getattr(config, "CURRENT_ITERATION", None) == 2 and n_features < min_features_iter2:
        sel = IndexSelector(np.arange(n_features))
        sel._set_original_n_features(n_features)
        diagnostics = {"selection_counts": np.zeros(n_features, dtype=int), "freq": np.ones(n_features, dtype=float),
                       "per_fold_selected": [], "successful_folds": 0, "final_indices": np.arange(n_features)}
        if return_diagnostics:
            return X, sel, diagnostics
        else:
            return X, sel

    # Optional scaling
    X_proc = X.copy()
    scaler = None
    if do_scale:
        try:
            scaler = RobustScaler()
            X_proc = scaler.fit_transform(X_proc)
        except Exception:
            X_proc = X.copy()
            scaler = None

    # Groups: prefer argument, else config.GROUPS
    if groups is None:
        groups = getattr(config, "GROUPS", None)

    use_loso = loso_enabled and (groups is not None)
    diagnostics: Dict[str, Any] = {}
    diagnostics['per_fold_selected'] = []
    diagnostics['selection_counts'] = np.zeros(n_features, dtype=int)
    diagnostics['successful_folds'] = 0
    diagnostics['mi_scores'] = None
    diagnostics['variances'] = np.var(X_proc, axis=0)

    final_indices = np.array([], dtype=int)

    if use_loso:
        groups = np.asarray(groups)
        if len(groups) != n_samples:
            print("Warning: provided groups length does not match samples. Falling back to non-LOSO selection.")
            use_loso = False

    if use_loso:
        print("\n[Stage LOSO] Running Leave-One-Group-Out per-training selection to find stable features...")
        logo = LeaveOneGroupOut()
        unique_groups = np.unique(groups)
        n_folds = len(unique_groups)
        print(f"  Unique groups (folds): {n_folds}")

        fold_idx = 0
        for train_idx, _ in logo.split(X_proc, y, groups):
            fold_idx += 1
            X_train = X_proc[train_idx, :]
            y_train = y[train_idx]
            try:
                selected_in_fold = _per_train_selection(X_train, y_train, config)
            except Exception as e:
                print(f"  Fold {fold_idx}: selection failed with {type(e).__name__}: {e}. Skipping fold.")
                continue

            if selected_in_fold.size == 0:
                print(f"  Fold {fold_idx}: no features selected on train (skipping).")
                continue

            diagnostics['selection_counts'][selected_in_fold] += 1
            diagnostics['per_fold_selected'].append(selected_in_fold)
            diagnostics['successful_folds'] += 1
            print(f"  Fold {fold_idx}: selected {selected_in_fold.size} features.")

        # frequency fraction based on total folds (unique_groups)
        freq = diagnostics['selection_counts'].astype(float) / float(len(unique_groups))
        diagnostics['freq'] = freq
        stable_mask = freq >= stability_threshold
        stable_indices = np.where(stable_mask)[0]
        print(f"  Features meeting stability threshold ({stability_threshold}): {stable_indices.size}")

        if stable_indices.size > 0:
            final_indices = np.asarray(stable_indices, dtype=int)
        else:
            print("  No features survived LOSO stability threshold. Falling back to global selection (no LOSO).")
            final_indices = _per_train_selection(X_proc, y, config)
            if final_indices.size == 0:
                print("  Global selection returned 0 features; falling back to top-k MI.")
                try:
                    mi_scores = mutual_info_classif(X_proc, y, discrete_features=False)
                    diagnostics['mi_scores'] = mi_scores
                    top_local = np.argsort(mi_scores)[::-1][:max(1, top_k)]
                    final_indices = np.unique(top_local)
                except Exception:
                    final_indices = np.arange(min(1, n_features))
    else:
        # Global selection on full dataset
        print("\n[Stage Global] LOSO not enabled or groups not provided — running global selection on whole dataset.")
        final_indices = _per_train_selection(X_proc, y, config)
        if final_indices.size == 0:
            print("  Global selection removed all features — falling back to top-k MI.")
            try:
                mi_scores = mutual_info_classif(X_proc, y, discrete_features=False)
                diagnostics['mi_scores'] = mi_scores
                top_local = np.argsort(mi_scores)[::-1][:max(1, top_k)]
                final_indices = np.unique(top_local)
            except Exception:
                final_indices = np.arange(min(1, n_features))

    # Safety: ensure valid indices
    final_indices = np.asarray(final_indices, dtype=int)
    final_indices = final_indices[(final_indices >= 0) & (final_indices < n_features)]
    if final_indices.size == 0:
        print("  Emergency fallback: selecting single highest-variance feature.")
        variances = diagnostics.get('variances', np.var(X_proc, axis=0))
        final_indices = np.array([int(np.argmax(variances))], dtype=int)

    # Build selected_features matrix and selector
    selected_features = X[:, final_indices]
    selector = IndexSelector(final_indices)
    selector._set_original_n_features(n_features)

    diagnostics['final_indices'] = final_indices
    diagnostics['selected_count'] = final_indices.size

    print(f"Final selected feature count: {final_indices.size}")

    if return_diagnostics:
        return selected_features, selector, diagnostics
    else:
        return selected_features, selector
