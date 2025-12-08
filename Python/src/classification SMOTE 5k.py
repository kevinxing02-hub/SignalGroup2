import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

def train_classifier(features, labels, config):
    """
    Enhanced training function with:
      1. Proper cross-validation (Stratified K-Fold)
      2. Class imbalance handling (SMOTE or class weights)
    """
    print(f"Training {config.CLASSIFIER_TYPE} classifier...")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Basic validation
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("No features available for training!")

    # ----- 1️⃣ HANDLE CLASS IMBALANCE -----
    # Many sleep datasets have far more N2 than N1/REM
    print("Applying SMOTE to balance classes in training data...")
    smote = SMOTE(random_state=42)
    features_balanced, labels_balanced = smote.fit_resample(features, labels)
    print("Class distribution before SMOTE:", np.bincount(labels))
    print("Class distribution after SMOTE :", np.bincount(labels_balanced))

    # ----- 2️⃣ CHOOSE CLASSIFIER -----
    if config.CURRENT_ITERATION == 1:
        model = KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS)
        print(f"Using k-NN with k={config.KNN_N_NEIGHBORS}")

    elif config.CURRENT_ITERATION == 2:
        # For SVM we can use class weights to rebalance
        class_weights = compute_class_weight('balanced', classes=np.unique(labels_balanced), y=labels_balanced)
        weight_dict = {cls: w for cls, w in zip(np.unique(labels_balanced), class_weights)}

        model = SVC(
            C=getattr(config, 'SVM_C', 1.0),
            kernel=getattr(config, 'SVM_KERNEL', 'rbf'),
            class_weight=weight_dict,
            random_state=42
        )
        print(f"Using SVM with C={model.C}, kernel={model.kernel}")

    elif config.CURRENT_ITERATION >= 3:
        model = RandomForestClassifier(
            n_estimators=getattr(config, 'RF_N_ESTIMATORS', 100),
            max_depth=getattr(config, 'RF_MAX_DEPTH', None),
            min_samples_split=getattr(config, 'RF_MIN_SAMPLES_SPLIT', 2),
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'  # handles imbalance internally
        )
        print(f"Using Random Forest with {model.n_estimators} trees")

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    # ----- 3️⃣ STRATIFIED K-FOLD CROSS-VALIDATION -----
    print("\nPerforming 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(model, features_balanced, labels_balanced,
                                cv=skf, scoring='f1_weighted', n_jobs=-1)
    acc_scores = cross_val_score(model, features_balanced, labels_balanced,
                                 cv=skf, scoring='accuracy', n_jobs=-1)

    print(f"Mean CV Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
    print(f"Mean CV F1-Weighted: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")

    # ----- 4️⃣ FINAL TRAIN/TEST SPLIT FOR REPORTING -----
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features_balanced, labels_balanced, test_size=0.2,
        stratify=labels_balanced, random_state=42
    )

    print(f"Training model on {X_train.shape[0]} samples and evaluating on {X_test.shape[0]} samples...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal Holdout Accuracy: {overall_accuracy:.3f}")
    print_performance_metrics(y_test, y_pred)

    return model



def print_performance_metrics(y_true, y_pred):
    """
    Print comprehensive performance metrics for sleep stage classification.

    Includes accuracy, sensitivity (recall), specificity, and F1-score for each sleep stage.
    """

    # Sleep stage labels and names (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stage_labels = list(range(5))

    print("\n" + "="*70)
    print("SLEEP STAGE CLASSIFICATION PERFORMANCE METRICS")
    print("="*70)

    # Overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"Macro F1-Score: {macro_f1:.3f}")
    print(f"Weighted F1-Score: {weighted_f1:.3f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=stage_labels)

    # Create a formatted confusion matrix
    cm_df = pd.DataFrame(cm, index=stage_names, columns=stage_names)
    print(cm_df.to_string())

    # Per-class metrics
    print("\nPer-Class Performance Metrics:")
    print("-" * 70)
    print(f"{'Stage':<8} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1-Score':<10}")
    print("-" * 70)

    # Calculate metrics for each sleep stage
    for i, stage_name in enumerate(stage_names):
        if i in y_true:  # Only calculate if stage is present in test set
            # Per-class accuracy (percentage of this class correctly classified)
            class_mask = (y_true == i)
            if np.sum(class_mask) > 0:
                class_accuracy = np.sum((y_pred == i) & (y_true == i)) / np.sum(class_mask)
            else:
                class_accuracy = 0.0

            # Sensitivity (Recall) - True Positive Rate
            sensitivity = recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]

            # Specificity - True Negative Rate
            tn = np.sum((y_true != i) & (y_pred != i))
            fp = np.sum((y_true != i) & (y_pred == i))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # F1-Score
            f1 = f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0]

            print(f"{stage_name:<8} {class_accuracy:<10.3f} {sensitivity:<12.3f} {specificity:<12.3f} {f1:<10.3f}")
        else:
            print(f"{stage_name:<8} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<10}")

    print("-" * 70)

    # Class distribution in test set
    print("\nClass Distribution in Test Set:")
    unique, counts = np.unique(y_true, return_counts=True)
    total_samples = len(y_true)

    for stage_idx, count in zip(unique, counts):
        stage_name = stage_names[stage_idx]
        percentage = count / total_samples * 100
        print(f"{stage_name}: {count} samples ({percentage:.1f}%)")

    # Sleep scoring specific notes
    print("\nNotes for Sleep Scoring:")
    print("- Sensitivity = Recall = True Positive Rate (correctly identified stages)")
    print("- Specificity = True Negative Rate (correctly rejected stages)")
    print("- Sleep stage imbalance is natural (more N2, less N1/REM)")
    print("- Consider Cohen's kappa for chance-corrected agreement")
    print("- Clinical focus: High sensitivity for REM and N3 stages")