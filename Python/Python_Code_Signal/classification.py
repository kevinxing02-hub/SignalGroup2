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

# Iteration 1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE

# Iteration 3+
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
    # ITERATION 3 — RF basic
    # -------------------------
    elif config.CURRENT_ITERATION == 3:
        print("\n=== Iteration 3: Random Forest ===\n")

        model = RandomForestClassifier(
            n_estimators=getattr(config, 'RF_N_ESTIMATORS', 100),
            max_depth=getattr(config, 'RF_MAX_DEPTH', None),
            min_samples_split=getattr(config, 'RF_MIN_SAMPLES_SPLIT', 2),
            min_samples_leaf=getattr(config, "RF_MIN_SAMPLES_LEAF", 1),
            class_weight=getattr(config, "RF_CLASS_WEIGHT", None),
            random_state=42,
            n_jobs=-1,
        )

        model.fit(features, labels)
        print("Random Forest trained (Iteration 3).")
        return model, scaler, loso_aggregated

    # -------------------------
    # ITERATION 4 — RF + Group-tuning + LOSO
    # -------------------------
    elif config.CURRENT_ITERATION == 4:
        print("\n=== Iteration 4: Random Forest + Hyperparameter Tuning + LOSO ===\n")

        if not hasattr(config, "record_ids"):
            raise ValueError("Iteration 4 requires config.record_ids (one subject ID per epoch).")

        groups = np.array(config.record_ids)

        # Pipeline with scaler and RF for grid search
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(class_weight=getattr(config, "RF_CLASS_WEIGHT", None), random_state=42, n_jobs=-1))
        ])

        param_grid = {
            "rf__n_estimators": [100, 200],
            "rf__max_depth": [10, 20, None],
            "rf__min_samples_split": [2, 5],
            "rf__min_samples_leaf": [1, 2],
        }

        gkf = GroupKFold(n_splits=5)
        scorer = make_scorer(f1_score, average="macro")

        grid = GridSearchCV(pipe, param_grid=param_grid, cv=gkf.split(features, labels, groups),
                            scoring=scorer, n_jobs=-1, verbose=1)
        grid.fit(features, labels)

        print("\nBest parameters from GroupKFold grid search:")
        print(grid.best_params_)
        print(f"Best macro-F1: {grid.best_score_:.3f}")

        # Extract RF parameters
        best_rf_params = {k.replace("rf__", ""): v for k, v in grid.best_params_.items() if k.startswith("rf__")}

        # Final LOSO evaluation: collect per-fold y_test and y_pred for aggregation
        logo = LeaveOneGroupOut()
        final_results = []
        fold_idx = 1

        # containers to aggregate fold-level arrays
        loso_test_labels = []
        loso_test_predictions = []

        for train_idx, test_idx in logo.split(features, labels, groups):
            test_subject = groups[test_idx][0]
            print(f"\nLOSO fold {fold_idx}: test subject = {test_subject}")
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_test_scaled = fold_scaler.transform(X_test)

            rf = RandomForestClassifier(**best_rf_params,
                                        class_weight=getattr(config, "RF_CLASS_WEIGHT", None),
                                        random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_test_scaled)

            # collect per-fold arrays
            loso_test_labels.append(np.asarray(y_test))
            loso_test_predictions.append(np.asarray(y_pred))

            acc = accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")

            final_results.append({"accuracy": acc, "kappa": kappa, "f1_macro": f1m})

            print(f"  Accuracy:  {acc:.3f}")
            print(f"  Kappa:     {kappa:.3f}")
            print(f"  F1-macro:  {f1m:.3f}")

            fold_idx += 1

        # Concatenate LOSO fold-level arrays into flat arrays
        try:
            loso_test_labels = np.concatenate(loso_test_labels) if len(loso_test_labels) > 0 else np.array([])
            loso_test_predictions = np.concatenate(loso_test_predictions) if len(loso_test_predictions) > 0 else np.array([])
        except Exception as e:
            print("ERROR: failed to concatenate LOSO fold-level arrays:", e)
            raise

        # package aggregated LOSO outputs
        loso_aggregated = {
            "test_labels": loso_test_labels,
            "test_predictions": loso_test_predictions,
            "per_fold": final_results
        }

        accs = [r["accuracy"] for r in final_results]
        kappas = [r["kappa"] for r in final_results]
        f1s = [r["f1_macro"] for r in final_results]

        print("\nFinal LOSO Performance (Iteration 4):")
        if len(accs) > 0:
            print(f"  Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
            print(f"  Kappa:    {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")
            print(f"  F1-macro: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        else:
            print("  No LOSO folds collected (unexpected).")

        # Train final model on ALL data
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(features)

        model = RandomForestClassifier(**best_rf_params,
                                       class_weight=getattr(config, "RF_CLASS_WEIGHT", None),
                                       random_state=42, n_jobs=-1)
        model.fit(X_all_scaled, labels)
        print("\nFinal Random Forest trained on all data (Iteration 4).")

        # Optional: checkpointing during training (if user supplies a validation set & config flags)
        if getattr(config, "SAVE_TRAINING_CHECKPOINTS", False) and val_data is not None:
            X_val, y_val = val_data
            visuals_dir = os.path.join(getattr(config, "OUTPUT_DIR", "."), f"visuals_iter{config.CURRENT_ITERATION}")
            os.makedirs(visuals_dir, exist_ok=True)
            record_training_checkpoint(model, X_val, y_val, epoch=0,
                                       output_dir=visuals_dir, prefix='final', class_names=getattr(config, 'CLASS_NAMES', None))

        return model, scaler, loso_aggregated

    else:
        raise ValueError("Invalid CURRENT_ITERATION in config (must be 1–4).")
