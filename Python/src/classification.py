import numpy as np
import pandas as pd
from collections import Counter

# SVM + Scaling + LOSO
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV

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

# ============================================================
#   MAIN CLASSIFIER FUNCTION
# ============================================================

def train_classifier(features, labels, config):
    """
    Universal training function for all iterations.

    Iteration 1 → k-NN + SMOTE + 5-fold CV
    Iteration 2 → SVM + StandardScaler + LOSO CV
    Iteration 3 → RandomForest

    Return ALWAYS:
        model, scaler
    where scaler can be None (e.g. for kNN / RF).
    """

    print(f"\n====================================================")
    print(f" Training Classifier (Iteration {config.CURRENT_ITERATION})")
    print(f"====================================================")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape:   {labels.shape}\n")

    if features.shape[1] == 0:
        raise ValueError("❌ ERROR: No features provided!")

    # default om ingen scaler används (iter 1 & 3)
    scaler = None

    # ============================================================
    # ITERATION 1 — k-NN + SMOTE + Stratified CV
    # ============================================================
    if config.CURRENT_ITERATION == 1:

        print("\n=== Iteration 1: k-NN Classification ===\n")

        # -------------------------
        # Train/test split
        # -------------------------
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            print("Using stratified train/test split")
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            print("⚠ Stratified split failed — falling back to non-stratified")

        print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

        # -------------------------
        # Handle class imbalance
        # -------------------------
        print("\nApplying SMOTE oversampling...")
        print("Original distribution:", Counter(y_train))

        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        print("Balanced distribution:", Counter(y_train))

        # -------------------------
        # Build and evaluate model
        # -------------------------
        model = KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS)

        print("\n5-fold Stratified CV on training set...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X_train, y_train, cv=cv,
            scoring="f1_weighted", n_jobs=-1
        )
        print("CV F1 scores:", np.round(scores, 3))
        print("Mean F1:", np.mean(scores))

        # Train final model
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print_performance_metrics(y_test, y_pred)

        print("\nCohen Kappa:", cohen_kappa_score(y_test, y_pred))

        try:
            y_prob = model.predict_proba(X_test)
            roc_auc = roc_auc_score(
                y_test, y_prob, multi_class="ovr", average="weighted"
            )
            print("Weighted ROC-AUC:", roc_auc)
        except Exception:
            print("ROC-AUC not available for k-NN.")

        # 🔁 Viktigt: returnera (model, scaler)
        return model, scaler

    # ============================================================
    # ITERATION 2 — SVM + StandardScaler + LOSO CV
    # ============================================================
    elif config.CURRENT_ITERATION == 2:

        print("\n=== Iteration 2: SVM + LOSO Cross-Validation ===\n")

        # ---------------------------------------
        # Require subject IDs for LOSO
        # ---------------------------------------
        if not hasattr(config, "record_ids"):
            raise ValueError(
                "❌ ERROR: Iteration 2 requires config.record_ids (subject ID per epoch)"
            )

        groups = np.array(config.record_ids)
        logo = LeaveOneGroupOut()
        loso_results = []

        fold_idx = 1
        best_overall_params = None

        # ====================================================
        # LOSO LOOP  — one subject held out per fold
        # ====================================================
        for train_idx, test_idx in logo.split(features, labels, groups):

            test_subject = groups[test_idx][0]
            print(f"\n---- LOSO Fold {fold_idx} — Test Subject {test_subject} ----")

            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # -----------------------------
            # Standardization (critical!)
            # -----------------------------
            inner_scaler = StandardScaler()
            X_train_scaled = inner_scaler.fit_transform(X_train)
            X_test_scaled = inner_scaler.transform(X_test)

            # -----------------------------
            # Small hyperparameter grid
            # -----------------------------
            param_grid = {
                "C": [0.1, 1, 10],
                "gamma": ["scale", 0.01, 0.001],
                "kernel": ["rbf"],
            }

            grid = GridSearchCV(
                SVC(),
                param_grid,
                cv=3,
                scoring="f1_weighted",
                n_jobs=-1,
            )

            grid.fit(X_train_scaled, y_train)
            best_svm = grid.best_estimator_

            print(f"  Best params: {grid.best_params_}")

            # Save last best params for final model
            best_overall_params = grid.best_params_

            # -----------------------------
            # Evaluate fold
            # -----------------------------
            y_pred = best_svm.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)

            print(f"Fold Accuracy: {acc:.3f}, Kappa: {kappa:.3f}")

            loso_results.append({"accuracy": acc, "kappa": kappa})
            fold_idx += 1

        # ------------------------------------------
        # Final Model (trained on ALL data)
        # ------------------------------------------

        scaler = StandardScaler()
        X_scaled_all = scaler.fit_transform(features)

        # Use the best hyperparameters found in LOSO
        model = SVC(
            C=best_overall_params["C"],
            gamma=best_overall_params["gamma"],
            kernel="rbf",
            random_state=42,
        )

        model.fit(X_scaled_all, labels)
        print("\nFinal SVM trained on all data.")

        # Return BOTH model and scaler
        return model, scaler

    # ============================================================
    # ITERATION 3 — Random Forest
    # ============================================================
    elif config.CURRENT_ITERATION >= 3:

        print("\n=== Iteration 3+: Random Forest ===\n")

        model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(features, labels)
        print("Random Forest trained.")

        # Ingen scaler används här → scaler = None
        return model, scaler

    else:
        raise ValueError("Invalid CURRENT_ITERATION in config.")


# ============================================================
#   PERFORMANCE METRICS
# ============================================================

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
        sens = recall_score(
            y_true, y_pred, labels=[i],
            average=None, zero_division=0
        )[0]

        tn = np.sum((y_true != i) & (y_pred != i))
        fp = np.sum((y_true != i) & (y_pred == i))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        f1 = f1_score(
            y_true, y_pred, labels=[i],
            average=None, zero_division=0
        )[0]

        print(f"{name:<8}{class_acc:<10.3f}{sens:<12.3f}{spec:<12.3f}{f1:<10.3f}")

    print("-" * 70)
