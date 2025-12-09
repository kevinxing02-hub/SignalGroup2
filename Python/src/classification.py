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
    Returns: model, scaler
    """
    print(f"\n====================================================")
    print(f" Training Classifier (Iteration {config.CURRENT_ITERATION})")
    print(f"====================================================")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape:   {labels.shape}\n")

    if features.shape[1] == 0:
        raise ValueError("ERROR: No features provided!")

    # Default when no scaler is used (e.g. iterations 1 and 3)
    scaler = None

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

        return model, scaler

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
        return model, scaler

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
        return model, scaler

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

        # Final LOSO evaluation
        logo = LeaveOneGroupOut()
        final_results = []
        fold_idx = 1
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

            acc = accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")

            final_results.append({"accuracy": acc, "kappa": kappa, "f1_macro": f1m})

            print(f"  Accuracy:  {acc:.3f}")
            print(f"  Kappa:     {kappa:.3f}")
            print(f"  F1-macro:  {f1m:.3f}")

            fold_idx += 1

        accs = [r["accuracy"] for r in final_results]
        kappas = [r["kappa"] for r in final_results]
        f1s = [r["f1_macro"] for r in final_results]

        print("\nFinal LOSO Performance (Iteration 4):")
        print(f"  Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"  Kappa:    {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")
        print(f"  F1-macro: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

        # Train final model on ALL data
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(features)

        model = RandomForestClassifier(**best_rf_params,
                                       class_weight=getattr(config, "RF_CLASS_WEIGHT", None),
                                       random_state=42, n_jobs=-1)
        model.fit(X_all_scaled, labels)
        print("\nFinal Random Forest trained on all data (Iteration 4).")

        # Optional: checkpointing during training (if user supplies a validation set & config flags)
        # NOTE: RF doesn't support partial_fit; this will only run if you call record_training_checkpoint manually.
        if getattr(config, "SAVE_TRAINING_CHECKPOINTS", False) and val_data is not None:
            X_val, y_val = val_data
            visuals_dir = os.path.join(getattr(config, "OUTPUT_DIR", "."), f"visuals_iter{config.CURRENT_ITERATION}")
            os.makedirs(visuals_dir, exist_ok=True)
            # Example: do a single snapshot after final fit
            record_training_checkpoint(model, X_val, y_val, epoch=0,
                                       output_dir=visuals_dir, prefix='final', class_names=getattr(config, 'CLASS_NAMES', None))

        return model, scaler

    else:
        raise ValueError("Invalid CURRENT_ITERATION in config (must be 1–4).")


def print_performance_metrics(y_true, y_pred):
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stage_labels = list(range(5))

    print("\n" + "=" * 70)
    print("SLEEP STAGE CLASSIFICATION METRICS")
    print("=" * 70)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy:        {acc:.3f}")
    print(f"Macro F1:        {macro_f1:.3f}")
    print(f"Weighted F1:     {weighted_f1:.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=stage_labels)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=stage_names, columns=stage_names).to_string())

    print("\nPer-Class Metrics:")
    print("-" * 70)
    print(f"{'Stage':<8}{'Accuracy':<10}{'Sensitivity':<12}{'Specificity':<12}{'F1-Score':<10}")

    for i, name in enumerate(stage_names):
        mask = (y_true == i)
        class_acc = np.mean(y_pred[mask] == i) if np.sum(mask) > 0 else 0
        sens = recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]

        tn = np.sum((y_true != i) & (y_pred != i))
        fp = np.sum((y_true != i) & (y_pred == i))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        f1 = f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]

        print(f"{name:<8}{class_acc:<10.3f}{sens:<12.3f}{spec:<12.3f}{f1:<10.3f}")

    print("-" * 70)
