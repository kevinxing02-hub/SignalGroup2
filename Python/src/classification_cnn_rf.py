# classification_cnn_rf.py
"""
Unified classification script supporting RandomForest (RF), 1D-CNN (CNN), and HYBRID (CNN embeddings -> RF).
Designed to be LOSO/Group-aware and drop-in compatible with prior train_classifier signature.
"""

import os
import numpy as np
from collections import Counter
import warnings

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, backend as K

# Optional utility for saving / loading sklearn models
import joblib

# -------------------------
# Small default config class
# -------------------------
class DefaultConfig:
    CURRENT_ITERATION = 4  # not strictly used, but kept for compatibility
    MODEL_TYPE = "RF"      # "RF", "CNN", or "HYBRID"
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = None
    RF_MIN_SAMPLES_SPLIT = 2
    RF_MIN_SAMPLES_LEAF = 1
    RF_CLASS_WEIGHT = "balanced"  # or None
    CNN_EPOCHS = 50
    CNN_BATCH_SIZE = 64
    CNN_LR = 1e-3
    CNN_DROPOUT = 0.4
    SAVE_TRAINING_CHECKPOINTS = False
    OUTPUT_DIR = "."
    CLASS_NAMES = None
    # for hybrid pipeline
    EMBEDDING_LAYER_NAME = "embedding"  # layer name to extract embeddings from
    VERBOSE = 2

# -------------------------
# Model building helpers
# -------------------------
def build_1d_cnn(input_shape, n_classes, dropout=0.4, name_prefix="cnn"):
    """
    Small 1D CNN. Returns compiled model (loss still to be compiled outside optionally).
    input_shape: (timesteps, channels)
    """
    inp = layers.Input(shape=input_shape, name=f"{name_prefix}_input")
    x = inp
    x = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=2)(x)

    x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # embedding layer (named, used for HYBRID)
    emb = layers.Dense(128, activation='relu', name='embedding')(x)
    x = layers.Dropout(dropout)(emb)
    out = layers.Dense(n_classes, activation='softmax', name='softmax')(x)

    model = models.Model(inputs=inp, outputs=out, name=f"{name_prefix}_model")
    return model

def extract_embeddings(model, X, batch_size=64):
    """Given a compiled/loaded model that has a layer named 'embedding', return embeddings for X."""
    try:
        emb_layer = model.get_layer('embedding')
    except Exception as e:
        raise ValueError("Model does not contain a layer named 'embedding'.") from e
    emb_model = models.Model(inputs=model.input, outputs=emb_layer.output)
    embeddings = emb_model.predict(X, batch_size=batch_size, verbose=0)
    return embeddings

# -------------------------
# Main training function
# -------------------------
def train_classifier(features, labels, config=None, val_data=None):
    """
    features: np.ndarray
        - If shape == (n_samples, timesteps, channels) => CNN-capable raw input
        - If shape == (n_samples, n_features) => flat features (RF-friendly)
    labels: np.ndarray (n_samples,)
    config: object with attributes controlling behaviour (see DefaultConfig)
    val_data: optional (X_val, y_val) used for checkpointing when enabled
    Returns: model, scaler, loso_aggregated
    """
    if config is None:
        config = DefaultConfig()

    print("\n=== train_classifier (CNN/RF/HYBRID) ===")
    print("Model type:", config.MODEL_TYPE)
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    # Basic checks
    if len(features.shape) not in (2, 3):
        raise ValueError("features must be 2D (n_samples, n_features) or 3D (n_samples, timesteps, channels)")

    # default outputs
    scaler = None
    loso_aggregated = None
    model = None

    # If features are 2D but user asked for CNN, attempt to auto-reshape
    is_time_series = (features.ndim == 3)
    if (config.MODEL_TYPE in ("CNN", "HYBRID")) and not is_time_series:
        # try to interpret last axis as channels if perfect square/time length
        n_samples, n_features = features.shape
        # heuristic: if n_features divisible by a plausible channel count (1-8), try reshape.
        possible = False
        for ch in (1, 2, 3, 4, 6, 8, 16):
            if n_features % ch == 0 and (n_features // ch) >= 4:
                timesteps = n_features // ch
                features = features.reshape(n_samples, timesteps, ch)
                is_time_series = True
                possible = True
                print(f"Auto-reshaped features -> (n_samples, timesteps, channels) = {features.shape}")
                break
        if not possible:
            warnings.warn("Requested CNN/HYBRID but input is flat. Please provide raw time-series shaped (n, timesteps, channels). Falling back to RF.")
            config.MODEL_TYPE = "RF"

    # If RF-only path
    if config.MODEL_TYPE == "RF":
        print("Training Random Forest on provided features (flat expected).")
        rf = RandomForestClassifier(
            n_estimators=getattr(config, 'RF_N_ESTIMATORS', 200),
            max_depth=getattr(config, 'RF_MAX_DEPTH', None),
            min_samples_split=getattr(config, 'RF_MIN_SAMPLES_SPLIT', 2),
            min_samples_leaf=getattr(config, 'RF_MIN_SAMPLES_LEAF', 1),
            class_weight=getattr(config, 'RF_CLASS_WEIGHT', None),
            random_state=42,
            n_jobs=-1,
        )
        # If features are 3D, flatten per sample
        if features.ndim == 3:
            n_samples = features.shape[0]
            X_flat = features.reshape(n_samples, -1)
        else:
            X_flat = features

        rf.fit(X_flat, labels)
        model = rf
        scaler = None
        print("Random Forest trained.")
        return model, scaler, loso_aggregated

    # For CNN or HYBRID, we require groups (subject ids) for LOSO in config.record_ids
    if not hasattr(config, "record_ids"):
        raise ValueError("CNN/HYBRID training requires config.record_ids (one subject ID per epoch).")

    groups = np.array(config.record_ids)
    logo = LeaveOneGroupOut()

    # prepare containers for LOSO aggregation
    loso_test_labels = []
    loso_test_predictions = []
    per_fold = []

    n_classes = len(np.unique(labels))

    # iterate LOSO folds
    fold_idx = 1
    for train_idx, test_idx in logo.split(features, labels, groups):
        subj = np.unique(groups[test_idx])
        print(f"\nLOSO fold {fold_idx} — test subject(s): {subj}")

        X_train = features[train_idx]
        X_test = features[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # Per-fold scaler: we scale features across samples, flattening time & channels for fit_transform,
        # then reshape back. This avoids information leakage across folds.
        fold_scaler = StandardScaler()
        n_train = X_train.shape[0]
        X_train_flat = X_train.reshape(n_train, -1)
        fold_scaler.fit(X_train_flat)
        X_train = fold_scaler.transform(X_train_flat).reshape(n_train, features.shape[1], features.shape[2])

        n_test = X_test.shape[0]
        X_test = fold_scaler.transform(X_test.reshape(n_test, -1)).reshape(n_test, features.shape[1], features.shape[2])

        # compute class weights from training labels
        classes = np.unique(y_train)
        class_weights_all = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {int(c): w for c, w in zip(classes, class_weights_all)}

        # -------------------------
        # CNN training (end-to-end)
        # -------------------------
        if config.MODEL_TYPE == "CNN":
            K.clear_session()
            cnn = build_1d_cnn(input_shape=(features.shape[1], features.shape[2]),
                               n_classes=n_classes, dropout=getattr(config, "CNN_DROPOUT", 0.4))
            cnn.compile(optimizer=optimizers.Adam(getattr(config, 'CNN_LR', 1e-3)),
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
            cb = [es]
            if getattr(config, "SAVE_TRAINING_CHECKPOINTS", False) and val_data is not None:
                ckpt_dir = os.path.join(getattr(config, "OUTPUT_DIR", "."), f"checkpoints_fold{fold_idx}")
                os.makedirs(ckpt_dir, exist_ok=True)
                cp = callbacks.ModelCheckpoint(os.path.join(ckpt_dir, "best.h5"), save_best_only=True, monitor='val_loss')
                cb.append(cp)

            hist = cnn.fit(
                X_train, y_train,
                validation_split=0.15,
                batch_size=getattr(config, "CNN_BATCH_SIZE", 64),
                epochs=getattr(config, "CNN_EPOCHS", 50),
                class_weight=class_weight_dict,
                callbacks=cb,
                verbose=getattr(config, "VERBOSE", 2)
            )

            y_prob = cnn.predict(X_test, batch_size=getattr(config, "CNN_BATCH_SIZE", 64), verbose=0)
            y_pred = np.argmax(y_prob, axis=1)

            # collect
            loso_test_labels.append(y_test)
            loso_test_predictions.append(y_pred)

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average='macro')
            kappa = cohen_kappa_score(y_test, y_pred)
            per_fold.append({'accuracy': acc, 'f1_macro': f1m, 'kappa': kappa})
            print(f"Fold {fold_idx} CNN: acc={acc:.3f}, f1_macro={f1m:.3f}, kappa={kappa:.3f}")

            # keep last fold's model as "model" to return (you can change to save best across folds)
            model = cnn

        # -------------------------
        # HYBRID: CNN embeddings -> RF
        # -------------------------
        elif config.MODEL_TYPE == "HYBRID":
            # Step 1: train CNN embedding on train set
            K.clear_session()
            cnn = build_1d_cnn(input_shape=(features.shape[1], features.shape[2]),
                               n_classes=n_classes, dropout=getattr(config, "CNN_DROPOUT", 0.4))
            cnn.compile(optimizer=optimizers.Adam(getattr(config, 'CNN_LR', 1e-3)),
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
            cnn.fit(
                X_train, y_train,
                validation_split=0.15,
                batch_size=getattr(config, "CNN_BATCH_SIZE", 64),
                epochs=getattr(config, "CNN_EPOCHS", 50),
                class_weight=class_weight_dict,
                callbacks=[es],
                verbose=getattr(config, "VERBOSE", 2)
            )

            # Step 2: extract embeddings
            X_train_emb = extract_embeddings(cnn, X_train, batch_size=getattr(config, "CNN_BATCH_SIZE", 64))
            X_test_emb = extract_embeddings(cnn, X_test, batch_size=getattr(config, "CNN_BATCH_SIZE", 64))

            # Step 3: fit RF on train embeddings and predict on test embeddings
            rf = RandomForestClassifier(
                n_estimators=getattr(config, 'RF_N_ESTIMATORS', 200),
                max_depth=getattr(config, 'RF_MAX_DEPTH', None),
                min_samples_split=getattr(config, 'RF_MIN_SAMPLES_SPLIT', 2),
                min_samples_leaf=getattr(config, 'RF_MIN_SAMPLES_LEAF', 1),
                class_weight=getattr(config, 'RF_CLASS_WEIGHT', None),
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train_emb, y_train)
            y_pred = rf.predict(X_test_emb)

            loso_test_labels.append(y_test)
            loso_test_predictions.append(y_pred)

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average='macro')
            kappa = cohen_kappa_score(y_test, y_pred)
            per_fold.append({'accuracy': acc, 'f1_macro': f1m, 'kappa': kappa})
            print(f"Fold {fold_idx} HYBRID: acc={acc:.3f}, f1_macro={f1m:.3f}, kappa={kappa:.3f}")

            # For return, we will package the final RF trained on all embeddings later.
            # Keep CNN around to extract embeddings for full dataset later
            model = {'cnn': cnn, 'rf_fold': rf}

        else:
            raise ValueError("Unknown MODEL_TYPE in config (valid: RF, CNN, HYBRID).")

        fold_idx += 1

    # after LOSO loop: aggregate
    try:
        loso_test_labels = np.concatenate(loso_test_labels) if len(loso_test_labels) > 0 else np.array([])
        loso_test_predictions = np.concatenate(loso_test_predictions) if len(loso_test_predictions) > 0 else np.array([])
    except Exception as e:
        print("ERROR concatenating LOSO outputs:", e)
        raise

    loso_aggregated = {
        "test_labels": loso_test_labels,
        "test_predictions": loso_test_predictions,
        "per_fold": per_fold
    }

    # Print summary metrics
    if loso_test_labels.size > 0:
        print("\nLOSO aggregated results:")
        print("Accuracy:", accuracy_score(loso_test_labels, loso_test_predictions))
        print("F1-macro:", f1_score(loso_test_labels, loso_test_predictions, average='macro'))
        print("Kappa:", cohen_kappa_score(loso_test_labels, loso_test_predictions))

    # Train final model on ALL data for deployment (following the same approach)
    print("\nTraining final model on ALL data (for deployment)...")
    # Re-fit scaler on all data (flatten then reshape)
    global_scaler = StandardScaler()
    if features.ndim == 3:
        n_all = features.shape[0]
        X_all_flat = features.reshape(n_all, -1)
        global_scaler.fit(X_all_flat)
        X_all_scaled = global_scaler.transform(X_all_flat).reshape(n_all, features.shape[1], features.shape[2])
    else:
        global_scaler.fit(features)
        X_all_scaled = global_scaler.transform(features)
    scaler = global_scaler

    if config.MODEL_TYPE == "CNN":
        K.clear_session()
        final_cnn = build_1d_cnn(input_shape=(features.shape[1], features.shape[2]), n_classes=n_classes,
                                 dropout=getattr(config, "CNN_DROPOUT", 0.4))
        final_cnn.compile(optimizer=optimizers.Adam(getattr(config, 'CNN_LR', 1e-3)),
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # fit on full dataset
        final_cnn.fit(X_all_scaled, labels,
                      batch_size=getattr(config, "CNN_BATCH_SIZE", 64),
                      epochs=getattr(config, "CNN_EPOCHS", 50),
                      verbose=getattr(config, "VERBOSE", 2))
        model = final_cnn
        print("Final CNN trained on all data.")

    elif config.MODEL_TYPE == "HYBRID":
        # Train CNN on all data, extract embeddings, then train RF on embeddings
        K.clear_session()
        final_cnn = build_1d_cnn(input_shape=(features.shape[1], features.shape[2]), n_classes=n_classes,
                                 dropout=getattr(config, "CNN_DROPOUT", 0.4))
        final_cnn.compile(optimizer=optimizers.Adam(getattr(config, 'CNN_LR', 1e-3)),
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        final_cnn.fit(X_all_scaled, labels,
                      batch_size=getattr(config, "CNN_BATCH_SIZE", 64),
                      epochs=getattr(config, "CNN_EPOCHS", 50),
                      verbose=getattr(config, "VERBOSE", 2))
        X_all_emb = extract_embeddings(final_cnn, X_all_scaled, batch_size=getattr(config, "CNN_BATCH_SIZE", 64))

        final_rf = RandomForestClassifier(
            n_estimators=getattr(config, 'RF_N_ESTIMATORS', 200),
            max_depth=getattr(config, 'RF_MAX_DEPTH', None),
            min_samples_split=getattr(config, 'RF_MIN_SAMPLES_SPLIT', 2),
            min_samples_leaf=getattr(config, 'RF_MIN_SAMPLES_LEAF', 1),
            class_weight=getattr(config, 'RF_CLASS_WEIGHT', None),
            random_state=42, n_jobs=-1
        )
        final_rf.fit(X_all_emb, labels)
        model = {'cnn': final_cnn, 'rf': final_rf}
        print("Final HYBRID (CNN embeddings -> RF) trained on all data.")

    else:
        # should not reach here
        pass

    # optional checkpoint on validation data
    if getattr(config, "SAVE_TRAINING_CHECKPOINTS", False) and val_data is not None:
        try:
            X_val, y_val = val_data
            os.makedirs(getattr(config, "OUTPUT_DIR", "."), exist_ok=True)
            if config.MODEL_TYPE == "CNN":
                model.save(os.path.join(config.OUTPUT_DIR, "final_cnn_model.h5"))
            elif config.MODEL_TYPE == "HYBRID":
                model['cnn'].save(os.path.join(config.OUTPUT_DIR, "final_hybrid_cnn.h5"))
                joblib.dump(model['rf'], os.path.join(config.OUTPUT_DIR, "final_hybrid_rf.joblib"))
            elif config.MODEL_TYPE == "RF":
                joblib.dump(model, os.path.join(config.OUTPUT_DIR, "final_rf.joblib"))
        except Exception as e:
            print("Warning: failed to save final checkpoints:", e)

    return model, scaler, loso_aggregated

# -------------------------
# Example quick test (only runs when script executed directly)
# -------------------------
if __name__ == "__main__":
    # small synthetic test to check shapes & flow
    from sklearn.datasets import make_classification
    n = 500
    timesteps = 128
    channels = 1
    X_flat, y = make_classification(n_samples=n, n_features=timesteps*channels, n_informative=20, n_classes=3, random_state=42)
    X = X_flat.reshape(n, timesteps, channels)

    cfg = DefaultConfig()
    cfg.MODEL_TYPE = "HYBRID"   # try "RF", "CNN", "HYBRID"
    # fake record ids: 10 subjects repeated
    cfg.record_ids = np.repeat(np.arange(10), n//10 + 1)[:n]

    model, scaler, loso = train_classifier(X, y, cfg)
    print("Done. LOSO per-fold:", loso['per_fold'])
