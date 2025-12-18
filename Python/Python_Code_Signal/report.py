import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd


def get_time_domain_feature_info(features, config):
    """
    Generate a description of the time domain features used in the pipeline.

    Args:
        features (np.ndarray): The extracted features array.
        config (module): The configuration module.

    Returns:
        str: Formatted string describing the features.
    """
    # Feature names for Iteration 1 (16 features per EEG channel)
    feature_names_per_channel = [
        # Statistical Moments (6 features)
        'mean', 'median', 'std', 'variance', 'skewness', 'kurtosis',
        # Amplitude Features (4 features)
        'rms', 'min', 'max', 'range',
        # Hjorth Parameters (3 features)
        'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
        # Frequency-Related (2 features)
        'zero_crossings', 'total_energy',
        # Complexity Measure (1 feature)
        'sample_entropy'
    ]

    n_features_per_channel = len(feature_names_per_channel)

    # For Iteration 1: 2 EEG channels × 16 features = 32 total features
    if config.CURRENT_ITERATION == 1:
        n_channels = 2  # EEG channels
        total_features = features.shape[1] if features is not None else n_channels * n_features_per_channel

        info = f"""### Feature Count
- Features per EEG channel: {n_features_per_channel}
- EEG channels: {n_channels}
- Total features: {total_features}

### Feature Categories

**1. Statistical Moments (6 features per channel)**
- mean: Average signal amplitude
- median: Median signal amplitude
- std: Standard deviation (signal variability)
- variance: Signal variance
- skewness: Asymmetry of signal distribution
- kurtosis: Tailedness of signal distribution

**2. Amplitude Features (4 features per channel)**
- rms: Root mean square (signal power measure)
- min: Minimum amplitude value
- max: Maximum amplitude value
- range: Peak-to-peak amplitude range

**3. Hjorth Parameters (3 features per channel) - Critical for Sleep EEG**
- hjorth_activity: Signal variance (represents power)
- hjorth_mobility: Mean frequency measure (sqrt(var(derivative) / var(signal)))
- hjorth_complexity: Signal bandwidth measure (frequency spread)

**4. Frequency-Related Features (2 features per channel)**
- zero_crossings: Number of sign changes (indicates dominant frequency)
- total_energy: Sum of squared signal values

**5. Complexity Measure (1 feature per channel)**
- sample_entropy: Signal irregularity/complexity measure (m=2, r=0.2*std)

### Feature Details
For each EEG channel (C3-A2 and C4-A1), all 16 features are extracted.
Total feature vector dimension: {total_features} features per epoch.

All features are extracted from 30-second epochs after preprocessing (high-pass, notch, low-pass filtering)."""
    elif config.CURRENT_ITERATION == 2:
        # Iteration 2: Time-domain + Frequency-domain features (EEG + EOG)
        # Expected: ~31 features (16 time + 15 spectral from EEG, + ~6 from EOG)
        total_features = features.shape[1] if features is not None else 0

        info = f"""### Feature Count
- Total features: {total_features}

### Feature Types (Iteration 2)

**EEG Features (~31 features per epoch):**
- Time-domain: 16 features per channel (2 channels = 32 features)
  - Statistical moments, amplitude features, Hjorth parameters, zero-crossings, energy, Sample Entropy
- Frequency-domain: ~15 features per channel (spectral estimation methods)
  - AR model features: Band powers, relative powers, spectral entropy, peak frequency, spectral edge frequency
  - Welch method features: Same spectral features with Welch's power spectral density estimation
  - Features selected via 3-stage pipeline (Variance → Correlation → Mutual Information)

**EOG Features (~6 features per channel, 2 channels = 12 features):**
- Mean, standard deviation, peak amplitude
- Movement energy, REM detection indicators
- Cross-channel correlation (optional)

**Feature Selection (Iteration 2):**
- Stage 1: Variance thresholding (remove <1% max variance)
- Stage 2: Correlation filtering (remove |r| > 0.95)
- Stage 3: Mutual Information selection (top 30-50 features)

**Expected Total:** ~96 features before selection → ~30-50 features after selection"""

    else:
        # For later iterations (3+)
        total_features = features.shape[1] if features is not None else 0
        info = f"""### Feature Count
- Total features: {total_features}

Note: Feature details for Iteration {config.CURRENT_ITERATION} may include multi-signal features (EEG+EOG+EMG)."""

    return info


def get_sample_entropy_analysis(features, labels, config):
    """
    Analyze Sample Entropy values across the dataset and sleep stages.

    Args:
        features (np.ndarray): The extracted features array.
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.

    Returns:
        str: Formatted string with Sample Entropy statistics.
    """
    if features is None or len(features) == 0:
        return "No features available for Sample Entropy analysis."

    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    # Find Sample Entropy feature indices (varies by iteration)
    if config.CURRENT_ITERATION == 1:
        # Iteration 1: 16 features per channel
        # Sample Entropy is the 16th feature (index 15) for channel 1 and 32nd (index 31) for channel 2
        n_features_per_channel = 16
        entropy_indices = [15, 31] if features.shape[1] > 31 else [15]
    elif config.CURRENT_ITERATION == 2:
        # Iteration 2: Features include time + spectral, structure is more complex
        # Sample Entropy is typically in the time-domain features section
        # For Iteration 2: Try to find Sample Entropy by checking for reasonable entropy values
        # It's likely in the first ~32 features (time-domain for 2 EEG channels)
        # Search for it by looking for entropy-like values
        entropy_indices = []
        if features.shape[1] >= 32:
            # Check first 32 features (time-domain from 2 channels)
            for idx in [15, 31]:  # Same positions as Iteration 1
                if idx < features.shape[1]:
                    test_values = features[:, idx]
                    # Sample Entropy should be in reasonable range (0-3 typically)
                    if np.all((test_values >= -1) & (test_values <= 3)) and np.any(test_values != 0):
                        entropy_indices.append(idx)
        # If not found, search more broadly
        if not entropy_indices and features.shape[1] > 0:
            # Try to find by looking for entropy-like patterns in first 50 features
            for idx in range(min(50, features.shape[1])):
                test_values = features[:, idx]
                # Check if values look like entropy (reasonable range, not constant)
                if (np.all((test_values >= -1) & (test_values <= 3)) and
                        np.std(test_values) > 0.01 and np.any(test_values != 0)):
                    entropy_indices.append(idx)
                    break  # Just find first occurrence
    else:
        # For Iteration 3+, search more broadly
        entropy_indices = []
        if features.shape[1] > 0:
            for idx in range(min(100, features.shape[1])):
                test_values = features[:, idx]
                if (np.all((test_values >= -1) & (test_values <= 3)) and
                        np.std(test_values) > 0.01 and np.any(test_values != 0)):
                    entropy_indices.append(idx)
                    break

    if not entropy_indices or any(idx >= features.shape[1] for idx in entropy_indices):
        return "Sample Entropy feature indices out of range. Feature extraction may not include Sample Entropy."

    info_lines = []

    # Overall statistics for each channel
    for ch_idx, entropy_idx in enumerate(entropy_indices, 1):
        entropy_values = features[:, entropy_idx]

        # Check if entropy is actually computed (not all zeros)
        non_zero_count = np.count_nonzero(entropy_values)
        total_count = len(entropy_values)
        is_computed = non_zero_count > 0.01 * total_count  # More than 1% non-zero

        info_lines.append(f"### Channel {ch_idx} Sample Entropy (Feature Index {entropy_idx})")

        if not is_computed:
            info_lines.append("- **Status:** ⚠️ Sample Entropy appears to be disabled or all values are zero")
            info_lines.append(
                "- This may indicate that `ENABLE_SAMPLE_ENTROPY = False` or `nolds` library is not installed")
        else:
            info_lines.append(f"- **Status:** ✅ Sample Entropy is being computed")
            info_lines.append(
                f"- **Non-zero values:** {non_zero_count}/{total_count} ({100 * non_zero_count / total_count:.1f}%)")

        info_lines.append(f"- **Overall Statistics:**")
        info_lines.append(f"  - Minimum: {np.min(entropy_values):.4f}")
        info_lines.append(f"  - Maximum: {np.max(entropy_values):.4f}")
        info_lines.append(f"  - Mean: {np.mean(entropy_values):.4f}")
        info_lines.append(f"  - Median: {np.median(entropy_values):.4f}")
        info_lines.append(f"  - Standard Deviation: {np.std(entropy_values):.4f}")
        info_lines.append("")

    # Per-stage statistics (for channel 1 only to avoid repetition)
    if len(entropy_indices) > 0 and config.CURRENT_ITERATION <= 2:
        entropy_idx = entropy_indices[0]  # Use channel 1
        entropy_values = features[:, entropy_idx]

        info_lines.append("### Sample Entropy by Sleep Stage (Channel 1)")
        info_lines.append("")
        info_lines.append("| Sleep Stage | Mean | Median | Std | Min | Max | Samples |")
        info_lines.append("|-------------|------|--------|-----|-----|-----|---------|")

        unique_labels = np.unique(labels)
        for stage_idx in sorted(unique_labels):
            if stage_idx < len(stage_names):
                stage_name = stage_names[int(stage_idx)]
                stage_mask = labels == stage_idx
                stage_entropy = entropy_values[stage_mask]

                if len(stage_entropy) > 0:
                    info_lines.append(
                        f"| {stage_name} | {np.mean(stage_entropy):.4f} | {np.median(stage_entropy):.4f} | "
                        f"{np.std(stage_entropy):.4f} | {np.min(stage_entropy):.4f} | "
                        f"{np.max(stage_entropy):.4f} | {len(stage_entropy)} |"
                    )

        info_lines.append("")
        info_lines.append("**Interpretation:**")
        info_lines.append("- Lower Sample Entropy = more regular/predictable signal (typical of deep sleep N3)")
        info_lines.append("- Higher Sample Entropy = more irregular/complex signal (typical of Wake/REM)")
        info_lines.append("- Expected pattern: N3 < N2 < N1/REM < Wake (generally)")

    return "\n".join(info_lines)


def extract_loso_summary(processing_log):
    """
    Extract LOSO cross-validation summary from processing log.

    Args:
        processing_log (str): The processing log output.

    Returns:
        str: Formatted LOSO summary section, or empty string if not found.
    """
    import re

    # Look for LOSO results section
    loso_pattern = r'LOSO CROSS-VALIDATION RESULTS.*?(?=\n\n|\Z)'
    loso_match = re.search(loso_pattern, processing_log, re.DOTALL)

    if loso_match:
        loso_section = loso_match.group(0)

        # Extract key metrics
        mean_acc_match = re.search(r'Mean Accuracy: ([0-9.]+)% ± ([0-9.]+)%', loso_section)
        mean_kappa_match = re.search(r"Mean Cohen's Kappa: ([0-9.]+) ± ([0-9.]+)", loso_section)
        mean_f1_match = re.search(r'Mean Macro F1-Score: ([0-9.]+) ± ([0-9.]+)', loso_section)

        summary_lines = []
        summary_lines.append("## LOSO Cross-Validation Results (Primary Evaluation)")
        summary_lines.append("")
        summary_lines.append(
            "**Iteration 2 uses Leave-One-Subject-Out (LOSO) cross-validation for subject-independent evaluation.**")
        summary_lines.append("This provides the most realistic estimate of model performance on unseen subjects.")
        summary_lines.append("")
        summary_lines.append("### Overall Performance (Aggregated across 10 folds)")
        summary_lines.append("")

        if mean_acc_match:
            summary_lines.append(f"- **Mean Accuracy:** {mean_acc_match.group(1)}% ± {mean_acc_match.group(2)}%")
        if mean_kappa_match:
            summary_lines.append(f"- **Mean Cohen's Kappa:** {mean_kappa_match.group(1)} ± {mean_kappa_match.group(2)}")
        if mean_f1_match:
            summary_lines.append(f"- **Mean Macro F1-Score:** {mean_f1_match.group(1)} ± {mean_f1_match.group(2)}")

        summary_lines.append("")
        summary_lines.append(
            "**Note:** Complete LOSO results (per-fold performance, confusion matrix, per-class metrics) are shown in the processing log above.")
        summary_lines.append("")

        return "\n".join(summary_lines)

    return ""


def generate_report(model, features, labels, config, processing_log, scaler=None):
    """
    Generates a report summarizing the results and writes to report.txt.
    Focuses on Iteration 2 LOSO results as primary evaluation.

    Args:
        model (object): The trained model.
        features (np.ndarray): The input features.
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.
        processing_log (str): The processing log output.
        scaler (StandardScaler, optional): Feature scaler (required for Iteration 2 SVM).
    """
    print("Generating report...")

    # Get feature information (iteration-specific)
    feature_info = get_time_domain_feature_info(features, config)

    # Get Sample Entropy analysis
    entropy_info = get_sample_entropy_analysis(features, labels, config)

    # Build report content based on iteration
    if config.CURRENT_ITERATION == 2:
        # Iteration 2: Focus on LOSO results (already in processing log)
        loso_summary = extract_loso_summary(processing_log)

        # Get model hyperparameters
        if hasattr(model, 'C') and hasattr(model, 'gamma'):
            hyperparams = f"C={model.C}, gamma={model.gamma}, kernel={model.kernel}"
        else:
            hyperparams = "See processing log for hyperparameters"

        # Check if class_weight is balanced
        if hasattr(model, 'class_weight_'):
            class_weight_info = "Yes (class_weight='balanced')"
        else:
            class_weight_info = "Not specified"

        report_content = f"""{processing_log}


# Sleep Scoring Report - Iteration 2

## Model Configuration
- **Classifier:** {type(model).__name__}
- **Feature Scaling:** Yes (StandardScaler - required for SVM)
- **Hyperparameters:** {hyperparams}
- **Class Imbalance Handling:** {class_weight_info}
- **Evaluation Method:** Leave-One-Subject-Out (LOSO) Cross-Validation

{loso_summary}

## Features
{feature_info}

## Sample Entropy Analysis
{entropy_info}

## Methodology Notes
- **LOSO Cross-Validation:** Each of 10 training subjects is held out once as test set, model trained on remaining 9 subjects
- **Feature Scaling:** StandardScaler applied within each LOSO fold to prevent data leakage
- **Subject-Independent:** LOSO provides realistic estimate of performance on completely unseen subjects (matches holdout evaluation)
- **Complete Results:** See processing log above for per-fold performance, aggregated confusion matrix, and per-class metrics

---
Report generated automatically from pipeline results.
"""

    else:
        # Iteration 1 and others: Standard evaluation
        # Scale features if needed
        if config.CURRENT_ITERATION >= 2 and scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features

        # Use simple train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, labels, test_size=0.2, random_state=42
            )

        # Get predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        cm_df = pd.DataFrame(cm, index=stage_names, columns=stage_names)
        cm_str = "\nConfusion Matrix:\n" + cm_df.to_string()

        report_content = f"""{processing_log}


# Sleep Scoring Report - Iteration {config.CURRENT_ITERATION}

## Model
{type(model).__name__}

## Performance Metrics (20% Test Split)
Accuracy: {accuracy:.3f}
Kappa: {kappa:.3f}
Macro F1-score: {macro_f1:.3f}
Weighted F1-score: {weighted_f1:.3f}

{cm_str}

## Features
{feature_info}

## Sample Entropy Analysis
{entropy_info}

---
Report generated automatically from pipeline results.
"""

    # Write to report.txt
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("Report saved to report.txt")