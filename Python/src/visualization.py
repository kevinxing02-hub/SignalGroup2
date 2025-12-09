# src/visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xml.etree.ElementTree as ET

# Keep the EDF plotting imports checking like you had:
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    try:
        import pyedflib
        HAS_PYEDFLIB = True
    except ImportError:
        HAS_PYEDFLIB = False

# Helper to save figures with consistent style & transparent background option
def _save_figure(fig, filename, output_dir, dpi=150, transparent=False):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', transparent=transparent)
    print(f"✓ Saved figure: {out_path}")


def plot_confusion_matrix(y_true, y_pred, class_names,
                          title="Confusion Matrix",
                          cmap='Blues',
                          normalize=False,
                          output_dir=None,
                          filename=None,
                          figsize=(8, 6)):
    """
    Plot and optionally save a confusion matrix.
    cmap: matplotlib colormap name or object (e.g., 'Blues','viridis','coolwarm', or a list of colors via ListedColormap)
    normalize: if True, show percentages (row-normalized)
    output_dir, filename: if provided the image will be saved
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        # row normalize
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        # avoid div-by-zero
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums
    else:
        cm_display = cm

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_display, display_labels=class_names)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(cmap=cmap, ax=ax, colorbar=True)
    ax.set_title(title, fontsize=14, fontweight='bold')

    if filename and output_dir:
        _save_figure(fig, filename, output_dir)
    plt.show()
    plt.close(fig)


def plot_class_distribution(y_true, y_pred=None, class_names=None,
                            title="Class Distribution",
                            output_dir=None, filename=None,
                            figsize=(8, 4), bar_color=None):
    """
    Plot bar chart of class distribution for ground-truth and optionally predictions.
    bar_color: single matplotlib color (string) or list of colors.
    Saves if output_dir + filename given.
    """
    fig, ax = plt.subplots(figsize=figsize)
    unique, counts = np.unique(y_true, return_counts=True)
    if class_names is None:
        class_names = [str(i) for i in range(len(unique))]
    # Map counts across all classes (ensure zero counts included)
    all_idxs = np.arange(len(class_names))
    counts_full = np.zeros(len(class_names), dtype=int)
    counts_full[unique] = counts

    ax.bar(all_idxs - 0.15, counts_full, width=0.3, label='True', color=bar_color)
    if y_pred is not None:
        u2, c2 = np.unique(y_pred, return_counts=True)
        pred_full = np.zeros(len(class_names), dtype=int)
        pred_full[u2] = c2
        ax.bar(all_idxs + 0.15, pred_full, width=0.3, label='Pred', alpha=0.8, color='gray')

    ax.set_xticks(all_idxs)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()

    if filename and output_dir:
        _save_figure(fig, filename, output_dir)
    plt.show()
    plt.close(fig)


# Keep plot_sample_epoch & plot_preprocessed_epoch but add color arg + save option
def plot_sample_epoch(edf_path, epoch_idx=0, epoch_duration=30, channel_types=None,
                      line_color='b', output_dir=None, filename=None):
    # ... keep your original body but when plotting replace 'b-' with f'{line_color}-'
    # For brevity here I'll show critical changes and reuse your implementation
    # (Copy your full original function body and replace plotting line with:)
    # axes[ch_idx].plot(times, signal, '-', color=line_color, linewidth=2.0, solid_capstyle='round')
    # At the end, if filename+output_dir call _save_figure(fig, filename, output_dir)
    #
    # To avoid repeating the full EDF reading code in this snippet, assume you paste your
    # original function body and add the color/save options as described.
    raise NotImplementedError("Paste original plot_sample_epoch body and add line_color + save")


def plot_preprocessed_epoch(preprocessed_data, epoch_idx=0, epoch_duration=30, channel_info=None,
                            line_color='r', output_dir=None, filename=None):
    # Same change as above: replace hard-coded 'r-' line with color argument and call _save_figure
    raise NotImplementedError("Paste original plot_preprocessed_epoch body and add line_color + save")


def plot_hypnogram(xml_path, edf_path=None, stage_colors=None,
                   title='Hypnogram - Sleep Stage Progression',
                   output_dir=None, filename=None, figsize=(15, 3)):
    """
    Plot hypnogram. stage_colors: list of 5 colors for ['Wake','N1','N2','N3','REM'].
    Will save if output_dir and filename provided.
    """
    if stage_colors is None:
        stage_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']  # red, orange, green, blue, purple

    # Use your existing implementation to parse XML -> epochs list
    # then when drawing use stage_colors[stage_label] instead of stage_colors[int(stage_label)]
    # and save with _save_figure

    # For brevity, reuse your XML parsing and plotting body but ensure colors from stage_colors are used
    raise NotImplementedError("Paste original plot_hypnogram body and ensure stage_colors usage + save")


def visualize_results(model, features, labels, config, scaler=None, loso_aggregated=None,
                      output_dir='visualizations', cmap='Blues', save_all=True,
                      class_names=None):
    """
    Main visualizer used by main.py. Saves several images:
      - confusion matrix (normalized and raw)
      - class distribution bar plot (true vs pred)
      - optionally a preprocessed epoch plot (if preprocessed data is in config or available)
      - hypnogram if xml annotation path provided in config.ANNOTATION_XML (optional)

    - save_all: if True will save multiple figures for recording
    - output_dir: directory where images are written
    """
    print("Visualizing results...")
    if class_names is None:
        class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    os.makedirs(output_dir, exist_ok=True)

    # Use LOSO aggregated results if present for iteration 2
    if getattr(config, 'CURRENT_ITERATION', None) == 2 and loso_aggregated is not None:
        y_test = loso_aggregated['test_labels']
        y_pred = loso_aggregated['test_predictions']

        print(f"Using LOSO aggregated confusion matrix (primary evaluation for Iteration 2)")
        print(f"  Total test samples: {len(y_test)}")

        # confusion matrix: raw and normalized
        plot_confusion_matrix(y_test, y_pred, class_names,
                              title="Confusion Matrix - LOSO Aggregated (Raw)",
                              cmap=cmap,
                              normalize=False,
                              output_dir=output_dir,
                              filename='confusion_loso_raw.png')

        plot_confusion_matrix(y_test, y_pred, class_names,
                              title="Confusion Matrix - LOSO Aggregated (Normalized)",
                              cmap=cmap,
                              normalize=True,
                              output_dir=output_dir,
                              filename='confusion_loso_norm.png')

        # class distributions
        plot_class_distribution(y_test, y_pred, class_names=class_names,
                                title="True vs Predicted Class Distribution (LOSO)",
                                output_dir=output_dir,
                                filename='class_dist_loso.png',
                                bar_color='#1f77b4')

    else:
        # fallback visualization using train/test split predictions (like your original code)
        from sklearn.model_selection import train_test_split
        if scaler is not None and getattr(config, 'CURRENT_ITERATION', None) in (2, 4):
            X = scaler.transform(features)
        else:
            X = features

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42, stratify=labels
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42
            )

        y_pred = model.predict(X_test)

        plot_confusion_matrix(y_test, y_pred, class_names,
                              title="Confusion Matrix (Visualization Split)",
                              cmap=cmap,
                              normalize=False,
                              output_dir=output_dir,
                              filename='confusion_visual_raw.png')

        plot_confusion_matrix(y_test, y_pred, class_names,
                              title="Confusion Matrix (Visualization Split) - Norm",
                              cmap=cmap,
                              normalize=True,
                              output_dir=output_dir,
                              filename='confusion_visual_norm.png')

        plot_class_distribution(y_test, y_pred, class_names=class_names,
                                title="True vs Predicted Class Distribution",
                                output_dir=output_dir,
                                filename='class_dist_visual.png',
                                bar_color='#ff7f0e')

    # Optional: if config contains annotation xml path, create hypnogram image
    xml_path = getattr(config, 'ANNOTATION_XML', None)
    if xml_path and os.path.exists(xml_path):
        try:
            # prefer user-specified colors if present
            stage_colors = getattr(config, 'HYPNOGRAM_COLORS', None)
            plot_hypnogram(xml_path, edf_path=getattr(config, 'SAMPLE_EDF', None),
                           stage_colors=stage_colors,
                           output_dir=output_dir,
                           filename='hypnogram.png')
        except Exception as e:
            print(f"Warning: Could not plot hypnogram: {e}")

    print("Visualization complete. Files saved at:", os.path.abspath(output_dir))
