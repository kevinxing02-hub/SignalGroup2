import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

"""
Robust feature selection pipeline that ALWAYS returns:
    (selected_features, selector_obj)

selector_obj implements:
    - transform(X) -> X[:, selected_indices]
    - get_support(indices=True/False)
    - _set_original_n_features(n)  (internal helper)
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


def select_features(features: np.ndarray, labels: np.ndarray, config):
    print(f"\n=== Feature Selection Pipeline (Iteration {config.CURRENT_ITERATION}) ===")
    n_samples, n_features = features.shape
    print(f"Initial feature count: {n_features}")

    # Basic input sanity
    if n_features == 0:
        print("No features available — returning empty array and identity selector")
        sel = IndexSelector(np.arange(0))
        sel._set_original_n_features(n_features)
        return features, sel

    if labels is None or len(labels) != n_samples:
        raise ValueError("Labels must be provided and match number of feature rows")

    # Configurable switches (with safe defaults)
    enabled = getattr(config, "FEATURE_SELECTION_ENABLED", True)
    min_features_iter2 = getattr(config, "FEATURE_SELECTION_MIN_FEATURES", 100)
    top_k = getattr(config, "FEATURE_SELECTION_TOP_K", 40)
    var_ratio = getattr(config, "VARIANCE_THRESHOLD_RATIO", 1e-4)
    corr_thresh = getattr(config, "CORRELATION_THRESHOLD", 0.95)

    if not enabled:
        print("Feature selection disabled by config. Returning original features and identity selector.")
        sel = IndexSelector(np.arange(n_features))
        sel._set_original_n_features(n_features)
        return features, sel

    # Iteration 2: conservative skip
    if config.CURRENT_ITERATION == 2 and n_features < min_features_iter2:
        print(f"Iteration 2 and features < {min_features_iter2} → skipping feature selection.")
        sel = IndexSelector(np.arange(n_features))
        sel._set_original_n_features(n_features)
        return features, sel

    # ---------- Stage 1: Variance Thresholding ----------
    print("\n[Stage 1] Variance Thresholding (conservative)...")
    variances = np.var(features, axis=0)
    max_var = np.max(variances)
    threshold = max(max_var * var_ratio, 1e-12)
    vt = VarianceThreshold(threshold=threshold)
    try:
        features_stage1 = vt.fit_transform(features)
        idx_stage1 = vt.get_support(indices=True)
    except Exception as e:
        print(f"  VarianceThreshold error {type(e).__name__}: {e}. Skipping stage1.")
        features_stage1 = features.copy()
        idx_stage1 = np.arange(n_features)

    removed_stage1 = n_features - features_stage1.shape[1]
    print(f"  Removed {removed_stage1} low-variance features")
    print(f"  Remaining after stage1: {features_stage1.shape[1]}")

    if features_stage1.shape[1] == 0:
        print("All features removed by variance thresholding — reverting to original features")
        features_stage1 = features.copy()
        idx_stage1 = np.arange(n_features)

    # ---------- Stage 2: Correlation pruning ----------
    print("\n[Stage 2] Correlation Filtering (group-wise, keep highest-variance)...")
    if features_stage1.shape[1] == 1:
        print("Only one feature left after stage1 → skipping correlation filtering.")
        features_stage2 = features_stage1
        idx_stage2 = idx_stage1
    else:
        corr = np.corrcoef(features_stage1, rowvar=False)
        abs_corr = np.abs(corr)
        n_f = abs_corr.shape[0]

        var_stage1 = variances[idx_stage1]
        sorted_idx = np.argsort(var_stage1)[::-1]  # indices into idx_stage1

        keep_mask = np.ones(n_f, dtype=bool)

        for ii in sorted_idx:
            if not keep_mask[ii]:
                continue
            correlated = (abs_corr[ii] > corr_thresh)
            correlated[ii] = False
            keep_mask[correlated] = False

        features_stage2 = features_stage1[:, keep_mask]
        idx_stage2 = idx_stage1[keep_mask]

        removed_corr = np.sum(~keep_mask)
        print(f"  Removed {removed_corr} correlated features (|r| > {corr_thresh})")
        print(f"  Remaining after stage2: {features_stage2.shape[1]}")

    # ---------- Stage 3: Mutual Information Top-K ----------
    print("\n[Stage 3] Mutual Information Ranking...")
    if features_stage2.shape[1] <= top_k:
        print(f"  Feature count ({features_stage2.shape[1]}) ≤ top_k ({top_k}) → skipping MI selection.")
        final_indices = idx_stage2
    else:
        try:
            mi_scores = mutual_info_classif(features_stage2, labels, discrete_features=False)
            top_indices_local = np.argsort(mi_scores)[::-1][:top_k]  # indices into features_stage2
            final_indices = idx_stage2[top_indices_local]  # map back to original indices
            print(f"  Selected top-{top_k} features via mutual information.")
            print(f"  Final feature count: {len(final_indices)}")
        except Exception as e:
            print(f"  mutual_info_classif failed with {type(e).__name__}: {e}. Using stage2 features.")
            final_indices = idx_stage2

    # Build final selected matrix and selector
    final_indices = np.asarray(final_indices, dtype=int)
    selected_features = features[:, final_indices]

    selector = IndexSelector(final_indices)
    selector._set_original_n_features(n_features)

    return selected_features, selector
