
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

"""
Robust feature selection pipeline.

Behaviour summary:
- If CURRENT_ITERATION == 2:
    - If number of features is small (< FEATURE_SELECTION_MIN_FEATURES) → SKIP selection
    - Else run the 3-stage pipeline (variance → correlation → MI) but with conservative thresholds.
- For later iterations (>=3) the full pipeline is applied.

Config accepted attributes (with defaults used if missing):
- FEATURE_SELECTION_ENABLED (bool)          : global switch (default True)
- FEATURE_SELECTION_MIN_FEATURES (int)      : min features to trigger selection in iter2 (default 100)
- FEATURE_SELECTION_TOP_K (int)             : final number of features to keep via MI (default 40)
- VARIANCE_THRESHOLD_RATIO (float)          : ratio * max_variance used for dynamic threshold (default 1e-4)
- CORRELATION_THRESHOLD (float)             : remove one of pair if |r| > threshold (default 0.95)
"""

def select_features(features: np.ndarray, labels: np.ndarray, config):
    print(f"\n=== Feature Selection Pipeline (Iteration {config.CURRENT_ITERATION}) ===")
    n_samples, n_features = features.shape
    print(f"Initial feature count: {n_features}")

    # Basic input sanity
    if n_features == 0:
        print("No features available — returning empty array")
        return features
    if labels is None or len(labels) != n_samples:
        raise ValueError("Labels must be provided and match number of feature rows")

    # Configurable switches (with safe defaults)
    enabled = getattr(config, "FEATURE_SELECTION_ENABLED", True)
    min_features_iter2 = getattr(config, "FEATURE_SELECTION_MIN_FEATURES", 100)
    top_k = getattr(config, "FEATURE_SELECTION_TOP_K", 40)
    var_ratio = getattr(config, "VARIANCE_THRESHOLD_RATIO", 1e-4)  # very conservative
    corr_thresh = getattr(config, "CORRELATION_THRESHOLD", 0.95)

    if not enabled:
        print("Feature selection disabled by config. Returning original features.")
        return features

    # Iteration 2: be conservative. Skip selection if feature count is small.
    if config.CURRENT_ITERATION == 2 and n_features < min_features_iter2:
        print(f"Iteration 2 and features < {min_features_iter2} → skipping feature selection.")
        return features

    # ---------- Stage 1: Variance Thresholding (conservative) ----------
    print("\n[Stage 1] Variance Thresholding (conservative)...")
    variances = np.var(features, axis=0)
    max_var = np.max(variances)
    # dynamic threshold: small fraction of max variance, but not zero
    threshold = max(max_var * var_ratio, 1e-12)
    vt = VarianceThreshold(threshold=threshold)
    try:
        features_stage1 = vt.fit_transform(features)
        idx_stage1 = vt.get_support(indices=True)
    except ValueError as e:
        # In case VarianceThreshold fails (e.g., numerical issues), skip this step
        print(f"  VarianceThreshold raised {type(e).__name__}: {e}. Skipping stage 1.")
        features_stage1 = features.copy()
        idx_stage1 = np.arange(n_features)

    removed_stage1 = n_features - features_stage1.shape[1]
    print(f"  Removed {removed_stage1} low-variance features")
    print(f"  Remaining after stage1: {features_stage1.shape[1]}")

    if features_stage1.shape[1] == 0:
        print("All features removed by variance thresholding — reverting to original features")
        features_stage1 = features.copy()
        idx_stage1 = np.arange(n_features)

    # ---------- Stage 2: Correlation-based pruning (keep highest-variance in groups) ----------
    print("\n[Stage 2] Correlation Filtering (group-wise, keep highest-variance)...")
    if features_stage1.shape[1] == 1:
        print("Only one feature left after stage1 → skipping correlation filtering.")
        features_stage2 = features_stage1
        idx_stage2 = idx_stage1
    else:
        # compute correlation matrix
        corr = np.corrcoef(features_stage1, rowvar=False)
        abs_corr = np.abs(corr)
        n_f = abs_corr.shape[0]

        # sort features by variance (descending) so we keep the most "informative" of correlated sets
        var_stage1 = variances[idx_stage1]
        sorted_idx = np.argsort(var_stage1)[::-1]  # indices into idx_stage1

        keep_mask = np.ones(n_f, dtype=bool)

        for ii in sorted_idx:
            if not keep_mask[ii]:
                continue
            # zero out features highly correlated with ii
            correlated = (abs_corr[ii] > corr_thresh)
            # don't remove itself
            correlated[ii] = False
            # remove those correlated features
            keep_mask[correlated] = False

        features_stage2 = features_stage1[:, keep_mask]
        idx_stage2 = idx_stage1[keep_mask]

        removed_corr = np.sum(~keep_mask)
        print(f"  Removed {removed_corr} correlated features (|r| > {corr_thresh})")
        print(f"  Remaining after stage2: {features_stage2.shape[1]}")

    # ---------- Stage 3: Mutual Information Top-K ----------
    print("\n[Stage 3] Mutual Information Ranking...")
    # If features are already <= k, skip MI selection
    if features_stage2.shape[1] <= top_k:
        print(f"  Feature count ({features_stage2.shape[1]}) ≤ top_k ({top_k}) → skipping MI selection.")
        return features_stage2

    # Compute mutual information scores robustly
    try:
        mi_scores = mutual_info_classif(features_stage2, labels, discrete_features=False)
        # sort descending and pick top_k
        top_indices = np.argsort(mi_scores)[::-1][:top_k]
        features_stage3 = features_stage2[:, top_indices]
        print(f"  Selected top-{top_k} features via mutual information.")
        print(f"  Final feature count: {features_stage3.shape[1]}")
        return features_stage3
    except Exception as e:
        print(f"  mutual_info_classif failed with {type(e).__name__}: {e}. Returning stage2 features.")
        return features_stage2
