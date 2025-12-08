import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, recall_score,
    f1_score, roc_auc_score, cohen_kappa_score
)
from imblearn.over_sampling import SMOTE
from collections import Counter


def train_classifier(features, labels, config):
    """
    TRUE 10-Fold Cross Validation
    - No train_test_split
    - All data used for training & testing
    - 10-fold with rotation (each sample tested once)
    """

    print(f"Training {config.CLASSIFIER_TYPE} classifier...")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    if features.shape[0] == 0:
        raise ValueError("No features to train!")

    # ===========================
    # 1️⃣ Build the model template
    # ===========================
    if config.CURRENT_ITERATION == 1:
        model = KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS)
        print(f"Using k-NN with k={config.KNN_N_NEIGHBORS}")

    elif config.CURRENT_ITERATION == 2:
        model = SVC(
            C=getattr(config, 'SVM_C', 1.0),
            kernel=getattr(config, 'SVM_KERNEL', 'rbf'),
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        print(f"Using SVM with C={model.C}, kernel={model.kernel}")

    elif config.CURRENT_ITERATION == 3:
        model = RandomForestClassifier(
            n_estimators=getattr(config, 'RF_N_ESTIMATORS', 100),
            max_depth=getattr(config, 'RF_MAX_DEPTH', None),
            min_samples_split=getattr(config, 'RF_MIN_SAMPLES_SPLIT', 2),
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        print(f"Using Random Forest with {model.n_estimators} trees")

    else:
        rf = RandomForestClassifier(
            n_estimators=getattr(config, 'RF_N_ESTIMATORS', 200),
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        svm = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced')
        model = VotingClassifier([('rf', rf), ('svm', svm)], voting='soft')
        print("Using Ensemble (RF + SVM)")

    # ===========================
    # 2️⃣ 10-Fold CV Initialization
    # ===========================
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    acc_list = []
    f1_list = []
    kappa_list = []

    fold_id = 1

    print("\n================= START 10-FOLD CROSS VALIDATION =================")

    for train_idx, test_idx in skf.split(features, labels):

        print(f"\n========== Fold {fold_id} ==========")

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        print(f"Original class distribution (train): {Counter(y_train)}")

        # ===========================
        # 3️⃣ SMOTE / Class balancing
        # ===========================
        if config.CLASSIFIER_TYPE in ['knn', 'svm']:
            print("Applying SMOTE oversampling...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {Counter(y_train)}")

        # ===========================
        # 4️⃣ Train model on this fold
        # ===========================
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ===========================
        # 5️⃣ Metrics
        # ===========================
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_test, y_pred)

        acc_list.append(acc)
        f1_list.append(f1)
        kappa_list.append(kappa)

        print(f"Fold {fold_id} Accuracy: {acc:.3f}")
        print(f"Fold {fold_id} Weighted F1: {f1:.3f}")
        print(f"Fold {fold_id} Kappa: {kappa:.3f}")

        fold_id += 1

    # ===========================
    # 6️⃣ Final Summary
    # ===========================
    print("\n================ FINAL 10-FOLD RESULTS ================")
    print(f"Mean Accuracy: {np.mean(acc_list):.3f} ± {np.std(acc_list):.3f}")
    print(f"Mean Weighted F1: {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")
    print(f"Mean Kappa: {np.mean(kappa_list):.3f} ± {np.std(kappa_list):.3f}")
    print("\nPipeline Complete ✔")

    return model   # Optional: return the last trained model