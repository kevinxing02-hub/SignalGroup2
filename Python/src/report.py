import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


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
    else:
        # For later iterations
        total_features = features.shape[1] if features is not None else 0
        info = f"""### Feature Count
- Total features: {total_features}

Note: Feature details for Iteration {config.CURRENT_ITERATION} may include frequency-domain features."""

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

    # For Iteration 1: Sample Entropy is the 16th feature (index 15) for channel 1
    # and 32nd feature (index 31) for channel 2
    if config.CURRENT_ITERATION == 1:
        n_features_per_channel = 16
        # Sample Entropy indices: position 15 (channel 1) and 31 (channel 2)
        entropy_indices = [15, 31]
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    else:
        # For other iterations, assume similar structure
        entropy_indices = [features.shape[1] - 1] if features.shape[1] > 0 else []
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

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
    if len(entropy_indices) > 0 and config.CURRENT_ITERATION == 1:
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


def generate_report(model, features, labels, config, processing_log):
    """
    Generates a report summarizing the results and writes to report.txt.
    Fills in the missing performance metrics.

    Args:
        model (object): The trained model.
        features (np.ndarray): The input features.
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.
        processing_log (str): The processing log output.
    """
    print("Generating report...")

    # Calculate metrics using the same train/test split as in current classification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    # Get time domain feature information
    feature_info = get_time_domain_feature_info(features, config)

    # Get Sample Entropy analysis
    entropy_info = get_sample_entropy_analysis(features, labels, config)

    # Build report content
    report_content = f"""{processing_log}


# Sleep Scoring Report - Iteration {config.CURRENT_ITERATION}

## Model
{type(model).__name__}

## Performance
Accuracy: {accuracy:.3f}
Kappa: {kappa:.3f}
Macro F1-score: {macro_f1:.3f}
Weighted F1-score: {weighted_f1:.3f}

## Time Domain Features
{feature_info}

## Sample Entropy Analysis
{entropy_info}

## Notes
Report generated automatically from pipeline results.
"""

    # Write to report.txt
    '''
    with open("report.txt", "w") as f:
        f.write(report_content)
    print("Report saved to report.txt")
    '''

    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("Report saved to report.txt")
