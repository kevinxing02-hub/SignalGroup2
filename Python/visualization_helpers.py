"""
visualization_helpers.py

Utilities to save per-iteration performance metrics and to generate an overall
training/testing performance figure that aggregates results from multiple
training iterations (missions). Filenames include a cluster name so you can
change the cluster name in config and all saved pictures will change accordingly.

This upgraded version produces a more informative figure:
- If multiple iterations are present: plots line charts across iterations for
  Train accuracy, LOSO/Test accuracy, Train F1 (macro) and Kappa (if available).
- If only a single iteration is present (your case: iteration 4 final):
  produces a composite figure with:
    * left: annotated bar chart of Train acc / LOSO acc / Train F1 / Kappa
    * right: confusion matrix heatmap (if available)
    * numeric values printed on bars for clarity

Also saves PNG and SVG copies (same filename base) so you get a crisp vector
output for reports.

Usage (in main.py, right after training finishes and `model` is available):

    from visualization_helpers import save_iteration_performance, plot_overall_performance

    save_iteration_performance(...)
    plot_overall_performance(output_dir=config.OUTPUT_DIR, cluster_name=getattr(config,'PICTURE_CLUSTER_NAME',None))

"""

import os
import json
from datetime import datetime
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score


def _safe_predict(model, X):
    """Try to predict with model; handle pipelines/scalers applied outside."""
    try:
        y_pred = model.predict(X)
    except Exception:
        # try if model has attribute 'estimator_' or fitted inside a pipeline
        if hasattr(model, 'estimator_'):
            try:
                y_pred = model.estimator_.predict(X)
            except Exception:
                y_pred = None
        else:
            y_pred = None
    return y_pred


def save_iteration_performance(config, iteration, model, scaler, features, labels,
                                loso_aggregated=None, output_dir='outputs', cluster_name=None,
                                run_loso_if_missing=True):
    """Compute and save performance metrics for a single iteration.

    - Saves JSON file `performance_iter{iteration}_{cluster_name}.json` inside output_dir
    - Returns the metrics dict
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = {}
    try:
        X = np.asarray(features)
        y = np.asarray(labels)
    except Exception:
        X = features
        y = labels

    # apply scaler if provided
    X_for_pred = X
    if scaler is not None:
        try:
            X_for_pred = scaler.transform(X)
        except Exception:
            X_for_pred = X

    # compute training predictions and metrics
    train_pred = _safe_predict(model, X_for_pred)
    if train_pred is not None:
        try:
            metrics['train_accuracy'] = float(accuracy_score(y, train_pred))
            metrics['train_f1_macro'] = float(f1_score(y, train_pred, average='macro'))
            metrics['train_precision_macro'] = float(precision_score(y, train_pred, average='macro', zero_division=0))
            metrics['train_recall_macro'] = float(recall_score(y, train_pred, average='macro', zero_division=0))
            metrics['confusion_matrix'] = np.asarray(confusion_matrix(y, train_pred)).tolist()
            try:
                metrics['kappa'] = float(cohen_kappa_score(y, train_pred))
            except Exception:
                metrics['kappa'] = None
        except Exception:
            metrics['train_accuracy'] = None
            metrics['kappa'] = None
    else:
        metrics['train_accuracy'] = None
        metrics['kappa'] = None

    # include loso_aggregated if available (library-specific)
    if loso_aggregated is not None:
        if isinstance(loso_aggregated, dict):
            for key in ('test_accuracy', 'loso_accuracy', 'accuracy'):
                if key in loso_aggregated:
                    metrics['loso_accuracy'] = float(loso_aggregated[key])
                    break
            for key in ('per_fold_accuracies', 'fold_accuracies', 'loso_fold_accuracies'):
                if key in loso_aggregated:
                    try:
                        metrics['loso_fold_accuracies'] = list(map(float, loso_aggregated[key]))
                        metrics['loso_std'] = float(np.std(metrics['loso_fold_accuracies']))
                    except Exception:
                        pass
        else:
            try:
                metrics['loso_accuracy'] = float(loso_aggregated)
            except Exception:
                pass

    # if LOSO missing, optionally compute using config.record_ids
    if 'loso_accuracy' not in metrics and run_loso_if_missing:
        record_ids = getattr(config, 'record_ids', None)
        if record_ids is not None:
            try:
                logo = LeaveOneGroupOut()
                scores = cross_val_score(model, X_for_pred, y, groups=np.asarray(record_ids), cv=logo, scoring='accuracy', n_jobs=1)
                metrics['loso_accuracy'] = float(np.mean(scores))
                metrics['loso_std'] = float(np.std(scores))
                metrics['loso_fold_accuracies'] = list(map(float, scores))
            except Exception:
                metrics['loso_accuracy'] = None
        else:
            metrics['loso_accuracy'] = None

    # timestamp and meta
    metrics['iteration'] = int(iteration)
    metrics['timestamp'] = datetime.utcnow().isoformat() + 'Z'
    metrics['cluster_name'] = cluster_name or getattr(config, 'PICTURE_CLUSTER_NAME', None) or 'default'

    # Save to file
    fname = f"performance_iter{iteration}_{metrics['cluster_name']}.json"
    file_path = os.path.join(output_dir, fname)
    try:
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        fallback = os.path.join(output_dir, f"performance_iter{iteration}.json")
        with open(fallback, 'w') as f:
            json.dump(metrics, f, indent=2)
        file_path = fallback

    print(f"Saved iteration performance to: {file_path}")
    return metrics


def _annotate_bars(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(fmt.format(h), xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', fontsize=9)


def _plot_confusion_matrix(ax, cm, classes=None, fmt='d'):
    cm_arr = np.asarray(cm)
    im = ax.imshow(cm_arr, interpolation='nearest', aspect='auto')
    ax.figure.colorbar(im, ax=ax)
    if classes is not None:
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # annotate cells
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, format(int(cm_arr[i, j]), fmt), ha='center', va='center', color='white' if cm_arr[i, j] > cm_arr.max()/2 else 'black')


def plot_overall_performance(output_dir='outputs', cluster_name=None, save_png=True, save_svg=True, show_fig=False):
    """Scan a single folder (output_dir) for performance_iter*.json files, aggregate and plot.

    - If multiple iteration files exist in the folder, plots trends across iterations.
    - If a single file exists, produces a detailed composite figure.
    """
    pattern = os.path.join(output_dir, f"performance_iter*_{cluster_name or '*'}*.json")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        files = sorted(glob.glob(os.path.join(output_dir, 'performance_iter*.json')))
    if len(files) == 0:
        raise FileNotFoundError(f"No performance json files found in {output_dir}")

    records = []
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                r = json.load(f)
                # normalize some keys
                if 'iteration' in r:
                    try:
                        r['iteration'] = int(r['iteration'])
                    except Exception:
                        pass
                records.append(r)
        except Exception:
            continue

    # sort records by iteration
    records = sorted(records, key=lambda x: int(x.get('iteration', 0)))

    # if only one record -> create detailed composite
    if len(records) == 1:
        r = records[0]
        it = int(r.get('iteration', 0))
        train_acc = r.get('train_accuracy')
        loso_acc = r.get('loso_accuracy')
        train_f1 = r.get('train_f1_macro')
        kappa = r.get('kappa')
        cm = r.get('confusion_matrix')

        metrics = [
            ('Train accuracy', train_acc),
            ('LOSO / test acc', loso_acc),
            ('Train F1 (macro)', train_f1),
            ('Kappa', kappa)
        ]

        labels = [m[0] for m in metrics]
        vals = [0.0 if m[1] is None else float(m[1]) for m in metrics]

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={'width_ratios': [1.2, 1]})
        ax_bar = axes[0]
        ax_cm = axes[1]

        bars = ax_bar.bar(range(len(vals)), vals)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_xticks(range(len(vals)))
        ax_bar.set_xticklabels(labels, rotation=25, ha='right')
        ax_bar.set_title(f'Iteration {it} — key metrics (cluster: {cluster_name or "default"})')
        _annotate_bars(ax_bar, bars, fmt="{:.3f}")

        if cm is not None:
            try:
                _plot_confusion_matrix(ax_cm, np.asarray(cm))
                ax_cm.set_title('Confusion matrix (train)')
            except Exception:
                ax_cm.text(0.5, 0.5, 'Confusion matrix not available', ha='center')
        else:
            ax_cm.text(0.5, 0.5, 'Confusion matrix not available', ha='center', va='center')

        plt.tight_layout()

        base_name = f"overall_performance_{cluster_name or 'default'}_iter{it}"
        out_png = os.path.join(output_dir, base_name + '.png')
        out_svg = os.path.join(output_dir, base_name + '.svg')
        if save_png:
            plt.savefig(out_png, dpi=200)
            print(f"Saved overall performance figure to: {out_png}")
        if save_svg:
            plt.savefig(out_svg)
            print(f"Saved overall performance (svg) to: {out_svg}")
        if show_fig:
            plt.show()
        else:
            plt.close()
        return out_png

    # Otherwise multiple records -> time series plot
    iters = [int(r.get('iteration', i)) for i, r in enumerate(records)]

    def metric_list(k):
        return [None if r.get(k) is None else float(r.get(k)) for r in records]

    train_acc = metric_list('train_accuracy')
    loso_acc = metric_list('loso_accuracy')
    train_f1 = metric_list('train_f1_macro')
    kappa = metric_list('kappa')

    plt.figure(figsize=(10, 5))
    plotted_any = False
    if any(v is not None for v in train_acc):
        plt.plot(iters, train_acc, marker='o', label='Train accuracy')
        plotted_any = True
    if any(v is not None for v in loso_acc):
        plt.plot(iters, loso_acc, marker='o', label='LOSO/Test accuracy')
        plotted_any = True
    if any(v is not None for v in train_f1):
        plt.plot(iters, train_f1, marker='o', label='Train F1 (macro)')
        plotted_any = True
    if any(v is not None for v in kappa):
        plt.plot(iters, kappa, marker='o', label='Kappa')
        plotted_any = True

    if not plotted_any:
        raise RuntimeError('No usable performance metrics to plot (train_accuracy / loso_accuracy / train_f1_macro / kappa not found).')

    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.ylim(-0.05, 1.05)
    plt.xticks(iters)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.title(f'Overall training & testing performance — cluster: {cluster_name or "default"}')
    plt.legend()
    plt.tight_layout()

    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_name = f"overall_performance_{cluster_name or 'default'}_{timestamp}.png"
    out_path = os.path.join(output_dir, out_name)
    if save_png:
        plt.savefig(out_path, dpi=200)
        print(f"Saved overall performance figure to: {out_path}")
    if save_svg:
        out_svg = out_path.rsplit('.', 1)[0] + '.svg'
        plt.savefig(out_svg)
        print(f"Saved overall performance (svg) to: {out_svg}")

    if show_fig:
        plt.show()
    else:
        plt.close()

    return out_path


if __name__ == '__main__':
    # convenience CLI
    import argparse
    parser = argparse.ArgumentParser(description='Plot overall training/testing performance across iterations')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--cluster_name', default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    plot_overall_performance(output_dir=args.output_dir, cluster_name=args.cluster_name, show_fig=args.show)
    # convenience CLI
    import argparse
    parser = argparse.ArgumentParser(description='Plot overall training/testing performance across iterations')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--cluster_name', default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    plot_overall_performance(output_dir=args.output_dir, cluster_name=args.cluster_name, show_fig=args.show)
