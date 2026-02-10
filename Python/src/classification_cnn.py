# classification.py  (修改版)
import os
import numpy as np
import pandas as pd
from collections import Counter

# SVM + Scaling + LOSO
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

# 新的统一训练器（包含 RF / CNN / HYBRID 实现）
# 请确保 src/classification_cnn_rf.py 中实现了 train_classifier 函数并返回 (model, scaler, loso_aggregated)
from src.classification_cnn_rf import train_classifier as train_classifier_new

# Iteration 1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE

# Iteration 3+ (保留导入以兼容旧代码)
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, confusion_matrix, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score
)

# Visualization helpers (assume these are in src.visualization)
try:
    from src.visualization import plot_confusion_matrix, plot_class_distribution, record_training_checkpoint
except Exception:
    # If not available, define simple placeholders to avoid crashes (won't save images)
    def plot_confusion_matrix(*args, **kwargs):
        pass
    def plot_class_distribution(*args, **kwargs):
        pass
    def record_training_checkpoint(*args, **kwargs):
        pass

def print_performance_metrics(y_true, y_pred):
    """Helper to print a small set of metrics (kept for compatibility with original code)."""
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1-macro:", f1_score(y_true, y_pred, average="macro"))
    print("Cohen Kappa:", cohen_kappa_score(y_true, y_pred))
    try:
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix:\n", cm)
    except Exception:
        pass

def train_classifier(features, labels, config, val_data=None):
    """
    Universal training function for all iterations.

    val_data: optional tuple (X_val, y_val) used for checkpointing when enabled via config.
    Returns: model, scaler, loso_aggregated
      - loso_aggregated is None for iterations that don't produce LOSO sample-level outputs.
      - For Iteration 4 it will be a dict with keys:
          'test_labels' : np.ndarray (concatenated y_test from each fold)
          'test_predictions' : np.ndarray (concatenated y_pred from each fold)
          'per_fold' : list of dicts with per-fold metrics
    """
    print(f"\n====================================================")
    print(f" Training Classifier (Iteration {config.CURRENT_ITERATION})")
    print(f"====================================================")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape:   {labels.shape}\n")

    if features.shape[1] == 0:
        raise ValueError("ERROR: No features provided!")

    # Default when no scaler is used
    scaler = None
    loso_aggregated = None  # default: only filled for iterations that collect LOSO

    # -------------------------
    # ITERATION 1 — k-NN + SMOTE
    # -------------------------
    if config.CURRENT_ITERATION == 1:
        print("\n=== Iteration 1: k-NN Classification ===\n")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            print("Using stratified train/test split")
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            print("Stratified split failed — falling back to non-stratified split")

        print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

        # SMOTE oversampling
        print("\nApplying SMOTE oversampling...")
        print("Original class distribution:", Counter(y_train))
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("Balanced class distribution:", Counter(y_train))

        model = KNeighborsClassifier(n_neighbors=getattr(config, 'KNN_N_NEIGHBORS', 5))

        # 5-fold Stratified CV on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
        print("CV F1 scores:", np.round(scores, 3))
        print("Mean F1:", np.mean(scores))

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print_performance_metrics(y_test, y_pred)

        print("\nCohen Kappa:", cohen_kappa_score(y_test, y_pred))
        try:
            # k-NN may support predict_proba, check
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                print("Weighted ROC-AUC:", roc_auc)
        except Exception:
            print("ROC-AUC not available for k-NN or failed to compute.")

        return model, scaler, loso_aggregated

    # -------------------------
    # ITERATION 2 — SVM + LOSO
    # -------------------------
    elif config.CURRENT_ITERATION == 2:
        print("\n=== Iteration 2: SVM + LOSO Cross-Validation ===\n")

        if not hasattr(config, "record_ids"):
            raise ValueError("Iteration 2 requires config.record_ids (subject ID per epoch).")

        groups = np.array(config.record_ids)
        logo = LeaveOneGroupOut()
        loso_results = []
        fold_idx = 1
        best_overall_params = None

        # Hyperparameter grid used inside each fold (small grid)
        param_grid = {
            "C": [0.1, 1, 10],
            "gamma": ["scale", 0.01, 0.001],
            "kernel": ["rbf"],
        }

        for train_idx, test_idx in logo.split(features, labels, groups):
            test_subject = groups[test_idx][0]
            print(f"\n---- LOSO Fold {fold_idx} — Test Subject {test_subject} ----")
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            inner_scaler = StandardScaler()
            X_train_scaled = inner_scaler.fit_transform(X_train)
            X_test_scaled = inner_scaler.transform(X_test)

            grid = GridSearchCV(SVC(), param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
            grid.fit(X_train_scaled, y_train)
            best_svm = grid.best_estimator_
            print(f"  Best params (fold): {grid.best_params_}")

            best_overall_params = grid.best_params_  # will end up with last fold's best; acceptable fallback

            y_pred = best_svm.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            print(f"Fold Accuracy: {acc:.3f}, Kappa: {kappa:.3f}")

            loso_results.append({"accuracy": acc, "kappa": kappa})
            fold_idx += 1

        # Train final SVM on all data using a scaler
        scaler = StandardScaler()
        X_scaled_all = scaler.fit_transform(features)

        if best_overall_params is None:
            # fallback to reasonable default
            best_overall_params = {"C": 1, "gamma": "scale"}

        model = SVC(
            C=best_overall_params.get("C", 1),
            gamma=best_overall_params.get("gamma", "scale"),
            kernel="rbf",
            probability=False  # set True if you need predict_proba (slower)
        )
        model.fit(X_scaled_all, labels)
        print("\nFinal SVM trained on all data.")
        return model, scaler, loso_aggregated

    # -------------------------
    # ITERATION 3 — DELEGATE TO classification_cnn_rf.train_classifier
    # -------------------------
    elif config.CURRENT_ITERATION == 3:
        print("\n=== Iteration 3: delegated training (RF/CNN/HYBRID via classification_cnn_rf) ===\n")
        # 直接调用新的统一训练函数（在 src/classification_cnn_rf.py 中）
        # 该函数会根据 config.MODEL_TYPE 执行 RF / CNN / HYBRID（并返回 model, scaler, loso_aggregated）
        model, scaler, loso_aggregated = train_classifier_new(features, labels, config, val_data=val_data)
        print("Iteration 3: delegated training complete.")
        return model, scaler, loso_aggregated

    # -------------------------
    # ITERATION 4 — DELEGATE TO classification_cnn_rf.train_classifier
    # -------------------------
    elif config.CURRENT_ITERATION == 4:
        print("\n=== Iteration 4: delegated training (RF tuning + LOSO or CNN/HYBRID via classification_cnn_rf) ===\n")
        # 同上：让新的模块负责 GroupKFold 网格搜索、LOSO 评估与最终模型训练
        model, scaler, loso_aggregated = train_classifier_new(features, labels, config, val_data=val_data)
        print("Iteration 4: delegated training complete.")
        return model, scaler, loso_aggregated

    else:
        raise ValueError("Invalid CURRENT_ITERATION in config (must be 1–4).")
