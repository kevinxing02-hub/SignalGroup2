import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import LeaveOneGroupOut

"""
Robust LOSO feature selection pipeline.

Public API:
    select_features(features: np.ndarray,
                    labels: np.ndarray,
                    config,
                    groups: Optional[np.ndarray] = None)
    -> (selected_features, selector_obj)

Behaviour highlights:
 - If groups is provided (or config.LOSO_ENABLED True and config provides groups),
   we perform LOSO across unique groups and aggregate per-fold selections.
 - We compute selection frequency and keep features with frequency >=
   config.FEAT_STABILITY_THRESHOLD (default 1.0 → chosen every fold).
 - If no features survive the stability threshold we fall back to a global selection
   (same 3 stages, applied on whole dataset) or pick top-k by MI to guarantee >0 output.
 - Always returns (selected_features, selector_obj) where selector implements:
     - transform(X)
     - get_support(indices=True/False)
     - _set_original_n_features(n)
"""

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

# ------------------ Helper: per-train selection (same logic as your original stages) ------------------
def _per_train_selection(X_train, y_train, config):
    """Run the 3-stage pipeline on training data and return selected original feature indices."""
    X_train = np.asarray(X_train)
    n_samples, n_features = X_train.shape

    # Configurable parameters (defaults kept from your original code)
    top_k = getattr(config, "FEATURE_SELECTION_TOP_K", 40)
    var_ratio = getattr(config, "VARIANCE_THRESHOLD_RATIO", 1e-4)
    corr_thresh = getattr(config, "CORRELATION_THRESHOLD", 0.95)

    # Stage 1: Variance thresholding (conservative)
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

    # Stage 2: Correlation pruning (keep highest-variance within correlated groups)
    if X_s1.shape[1] == 1:
        X_s2 = X_s1
        idx_s2 = idx_s1
    else:
        corr = np.corrcoef(X_s1, rowvar=False)
        abs_corr = np.abs(corr)
        var_stage1 = variances[idx_s1]
        sorted_idx = np.argsort(var_stage1)[::-1]  # order of preference (indices into idx_s1)
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

    # Stage 3: Mutual information top-k (applied on the stage2 set)
    if X_s2.shape[1] <= top_k:
        final_indices = idx_s2
    else:
        try:
            mi_scores = mutual_info_classif(X_s2, y_train, discrete_features=False)
            top_local = np.argsort(mi_scores)[::-1][:top_k]
            final_indices = idx_s2[top_local]
        except Exception:
            final_indices = idx_s2

    # Ensure unique, sorted indices (relative to original feature numbering)
    final_indices = np.unique(np.asarray(final_indices, dtype=int))
    return final_indices

# ------------------ Main entrypoint ------------------
def select_features(features: np.ndarray, labels: np.ndarray, config, groups: np.ndarray = None):
    print(f"\n=== Robust Feature Selection (Iteration {getattr(config, 'CURRENT_ITERATION', 'N/A')}) ===")
    features = np.asarray(features)
    n_samples, n_features = features.shape
    print(f"Initial feature count: {n_features}")

    # Basic input checks
    if n_features == 0:
        print("No features available — returning empty array and identity selector")
        sel = IndexSelector(np.arange(0))
        sel._set_original_n_features(n_features)
        return features, sel

    if labels is None or len(labels) != n_samples:
        raise ValueError("Labels must be provided and match number of feature rows")

    # Config switches & defaults
    enabled = getattr(config, "FEATURE_SELECTION_ENABLED", True)
    min_features_iter2 = getattr(config, "FEATURE_SELECTION_MIN_FEATURES", 100)
    loso_enabled = getattr(config, "LOSO_ENABLED", True)
    stability_threshold = getattr(config, "FEAT_STABILITY_THRESHOLD", 1.0)  # fraction of folds a feature must appear in
    top_k = getattr(config, "FEATURE_SELECTION_TOP_K", 40)

    if not enabled:
        print("Feature selection disabled by config. Returning original features and identity selector.")
        sel = IndexSelector(np.arange(n_features))
        sel._set_original_n_features(n_features)
        return features, sel

    # Iteration 2 skip (preserve behaviour)
    if getattr(config, "CURRENT_ITERATION", None) == 2 and n_features < min_features_iter2:
        print(f"Iteration 2 and features < {min_features_iter2} → skipping feature selection.")
        sel = IndexSelector(np.arange(n_features))
        sel._set_original_n_features(n_features)
        return features, sel

    # If LOSO desired, groups must be provided either via argument or config
    if loso_enabled and groups is None:
        groups = getattr(config, "GROUPS", None)

    # If LOSO is enabled and groups are available, run LOSO aggregation
    if loso_enabled and groups is not None:
        groups = np.asarray(groups)
        if len(groups) != n_samples:
            print("Warning: provided groups length does not match samples. Falling back to non-LOSO selection.")
            groups = None

    if loso_enabled and groups is not None:
        print("\n[Stage LOSO] Running Leave-One-Group-Out per-training selection to find stable features...")
        logo = LeaveOneGroupOut()
        unique_groups = np.unique(groups)
        n_folds = len(unique_groups)
        print(f"  Unique groups (folds): {n_folds}")

        # track selection counts for original feature indices
        selection_counts = np.zeros(n_features, dtype=int)
        fold_idx = 0
        for train_idx, _ in logo.split(features, labels, groups):
            fold_idx += 1
            X_train = features[train_idx, :]
            y_train = labels[train_idx]
            try:
                selected_in_fold = _per_train_selection(X_train, y_train, config)
            except Exception as e:
                print(f"  Fold {fold_idx}: selection failed with {type(e).__name__}: {e}. Skipping fold.")
                continue

            if selected_in_fold.size == 0:
                # nothing selected in this fold — skip counting
                print(f"  Fold {fold_idx}: no features selected on train (skipping).")
                continue

            selection_counts[selected_in_fold] += 1
            print(f"  Fold {fold_idx}: selected {selected_in_fold.size} features.")

        # Convert counts to frequency fraction
        # Note: use number of folds that actually ran (n_folds) — not number of successful folds
        freq = selection_counts.astype(float) / float(n_folds)
        # choose features with freq >= threshold (stability)
        stable_mask = freq >= stability_threshold
        stable_indices = np.where(stable_mask)[0]
        print(f"  Features meeting stability threshold ({stability_threshold}): {stable_indices.size}")

        if stable_indices.size > 0:
            final_indices = np.asarray(stable_indices, dtype=int)
        else:
            print("  No features survived LOSO stability threshold. Falling back to global selection (no LOSO).")
            # fallback: run the per-train selection on the whole dataset
            final_indices = _per_train_selection(features, labels, config)
            if final_indices.size == 0:
                # ultimate fallback: pick top_k by mutual information on whole data (ensure at least 1)
                print("  Global selection returned 0 features; falling back to top-k MI.")
                try:
                    mi_scores = mutual_info_classif(features, labels, discrete_features=False)
                    top_local = np.argsort(mi_scores)[::-1][:max(1, top_k)]
                    final_indices = np.unique(top_local)
                except Exception:
                    final_indices = np.arange(min(1, n_features))

    else:
        # LOSO not used — use single-shot selection on whole dataset (preserves original behaviour)
        print("\n[Stage Global] LOSO not enabled or groups not provided — running global selection on whole dataset.")
        final_indices = _per_train_selection(features, labels, config)
        if final_indices.size == 0:
            print("  Global selection removed all features — falling back to top-k MI.")
            try:
                mi_scores = mutual_info_classif(features, labels, discrete_features=False)
                top_local = np.argsort(mi_scores)[::-1][:max(1, top_k)]
                final_indices = np.unique(top_local)
            except Exception:
                final_indices = np.arange(min(1, n_features))

    # final safety: ensure at least one index and indices are within bounds
    final_indices = np.asarray(final_indices, dtype=int)
    final_indices = final_indices[(final_indices >= 0) & (final_indices < n_features)]
    if final_indices.size == 0:
        # emergency fallback to the single most variant feature
        print("  Emergency fallback: selecting single highest-variance feature.")
        variances = np.var(features, axis=0)
        final_indices = np.array([int(np.argmax(variances))], dtype=int)

    # Build output selected matrix and selector
    selected_features = features[:, final_indices]
    selector = IndexSelector(final_indices)
    selector._set_original_n_features(n_features)

    print(f"Final selected feature count: {final_indices.size}")
    return selected_features, selector
