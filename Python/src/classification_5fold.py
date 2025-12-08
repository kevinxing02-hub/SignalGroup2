import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, recall_score,
    f1_score, roc_auc_score, cohen_kappa_score
)
from imblearn.over_sampling import SMOTE
from collections import Counter


def train_classifier(features, labels, config):
    """
    Enhanced training function implementing:
    1. Proper stratified k-fold cross-validation
    2. Handling of class imbalance (SMOTE / class weights)
    3. Basic hyperparameter tuning examples
    4. Advanced evaluation metrics (Cohen’s Kappa, ROC-AUC)
    5. Ensemble method support for iteration >= 4
    """

    print(f"Training {config.CLASSIFIER_TYPE} classifier...")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Basic validation
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("No features available for training!")

    # Stratified train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print("✅ Using stratified train/test split to maintain class balance")
    except ValueError as e:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        print(f"⚠️ Fallback to non-stratified split: {e}")

    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # ===============================
    # 1️⃣ Handle Class Imbalance
    # ===============================
    print(f"\nOriginal class distribution (train): {Counter(y_train)}")

    if config.CLASSIFIER_TYPE in ['knn', 'svm']:
        # SMOTE works well for continuous features
        print("Applying SMOTE oversampling for minority classes...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"New balanced class distribution: {Counter(y_train)}")

    elif config.CLASSIFIER_TYPE in ['random_forest']:
        # Use built-in class weights for tree-based models
        print("Applying class_weight='balanced' for Random Forest")

    # ===============================
    # 2️⃣ Select & Configure Classifier
    # ===============================
    if config.CURRENT_ITERATION == 1:
        model = KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS)
        print(f"Using k-NN with k={config.KNN_N_NEIGHBORS}")

    elif config.CURRENT_ITERATION == 2:
        # SVM with tuned hyperparameters
        model = SVC(
            C=getattr(config, 'SVM_C', 1.0),
            kernel=getattr(config, 'SVM_KERNEL', 'rbf'),
            gamma='scale',
            class_weight='balanced',  # handle imbalance
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

    elif config.CURRENT_ITERATION >= 4:
        # Ensemble example (Random Forest + SVM)
        rf = RandomForestClassifier(
            n_estimators=getattr(config, 'RF_N_ESTIMATORS', 200),
            max_depth=None,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        svm = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42)
        model = VotingClassifier(estimators=[('rf', rf), ('svm', svm)], voting='soft')
        print("Using Ensemble (Random Forest + SVM)")

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    # ===============================
    # 3️⃣ Cross-Validation
    # ===============================
    print("\nPerforming 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
    print(f"Cross-validation F1-scores: {np.round(cv_scores, 3)}")
    print(f"Mean F1-score (CV): {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    # ===============================
    # 4️⃣ Train Final Model
    # ===============================
    print("\nTraining final model on full training set...")
    model.fit(X_train, y_train)

    # ===============================
    # 5️⃣ Evaluation
    # ===============================
    y_pred = model.predict(X_test)
    print_performance_metrics(y_test, y_pred)

    # --- Advanced Metrics ---
    print("\n" + "="*70)
    print("ADVANCED EVALUATION METRICS")
    print("="*70)

    # Cohen’s Kappa
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"Cohen's Kappa: {kappa:.3f}")

    # ROC-AUC per class (only if model supports probability output)
    try:
        y_prob = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        print(f"Weighted ROC-AUC: {roc_auc:.3f}")
    except Exception:
        print("⚠️ ROC-AUC skipped (model does not support probability outputs)")

    print("="*70)
    print("PIPELINE COMPLETE ✅")

    return model


def print_performance_metrics(y_true, y_pred):
    """Detailed sleep stage classification metrics."""
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stage_labels = list(range(5))

    print("\n" + "="*70)
    print("SLEEP STAGE CLASSIFICATION PERFORMANCE METRICS")
    print("="*70)

    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"Macro F1-Score: {macro_f1:.3f}")
    print(f"Weighted F1-Score: {weighted_f1:.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=stage_labels)
    cm_df = pd.DataFrame(cm, index=stage_names, columns=stage_names)
    print("\nConfusion Matrix:\n", cm_df.to_string())

    print("\nPer-Class Metrics:")
    print(f"{'Stage':<8} {'Recall':<10} {'F1-Score':<10}")
    for i, stage in enumerate(stage_names):
        recall = recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]
        f1 = f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]
        print(f"{stage:<8} {recall:<10.3f} {f1:<10.3f}")
    print("="*70)