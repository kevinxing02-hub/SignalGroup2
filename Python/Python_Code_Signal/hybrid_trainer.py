# src/hybrid_trainer.py
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
import joblib
import tensorflow as tf
from hybrid_models import build_cnn_embedding_model
from hybrid_utils import get_embedding_model, extract_embeddings
import time

def train_hybrid(X_epochs, y, groups=None, cfg=None, save_dir="./outputs/hybrid", verbose=True):
    """
    X_epochs: (n_samples, n_channels, n_timesteps) OR (n_samples, n_timesteps, n_channels)
    We'll convert to (n_samples, n_timesteps, n_channels) for Keras.
    y: (n_samples,)
    groups: per-sample group ids for LOSO. If None, train single-shot split.
    cfg: config object with CNN + RF hyperparams
    """
    os.makedirs(save_dir, exist_ok=True)
    # Normalize axes => keras expects (samples, timesteps, channels)
    if X_epochs.ndim != 3:
        raise ValueError("X_epochs must be 3D: (n_samples, channels, samples) or (n_samples, samples, channels)")
    # if channels first, convert
    n0, d1, d2 = X_epochs.shape
    # heuristic: if d1 small (<=16) and d2 large, assume (n, ch, samples)
    if d1 <= 32 and d2 > d1:
        X = X_epochs.transpose(0, 2, 1)  # (n, samples, channels)
    else:
        X = X_epochs  # assume already (n, samples, channels)

    n_samples, timesteps, channels = X.shape
    if verbose:
        print(f"Hybrid training on {n_samples} epochs, timesteps={timesteps}, channels={channels}")

    # Prepare config defaults
    cnn_epochs = getattr(cfg, "CNN_EPOCHS", 30)
    batch_size = getattr(cfg, "CNN_BATCH_SIZE", 64)
    embed_dim = getattr(cfg, "CNN_EMBED_DIM", 128)
    cnn_lr = getattr(cfg, "CNN_LR", 1e-3)
    cnn_patience = getattr(cfg, "CNN_PATIENCE", 6)
    rf_n = getattr(cfg, "RF_N_ESTIMATORS", 200)
    rf_depth = getattr(cfg, "RF_MAX_DEPTH", 20)
    random_state = getattr(cfg, "RF_RANDOM_STATE", 42)
    save_embeddings = getattr(cfg, "HYBRID_SAVE_EMBEDDINGS", True)
    emb_dir = getattr(cfg, "HYBRID_EMBED_DIR", save_dir)

    os.makedirs(emb_dir, exist_ok=True)

    # LOSO or single-split
    if groups is not None and getattr(cfg, "HYBRID_USE_LOSO", True):
        logo = LeaveOneGroupOut()
        splits = list(logo.split(X, y, groups))
        use_loso = True
    else:
        # use single train/test split (stratified)
        tr_idx, te_idx = train_test_split(np.arange(n_samples), test_size=0.2, stratify=y, random_state=random_state)
        splits = [(tr_idx, te_idx)]
        use_loso = False

    fold_results = []
    total_start = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n=== Fold {fold_idx}/{len(splits)}: Train {len(train_idx)} / Test {len(test_idx)} ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scaling per-channel: fit on train
        scaler = RobustScaler()
        # scaler expects 2D: flatten time+channels per epoch, but we want per-sample per-channel scale:
        # We'll scale across time for each channel separately: reshape (n_samples*timesteps, channels)
        X_train_2d = X_train.reshape(-1, channels)
        scaler.fit(X_train_2d)
        # transform
        X_train_scaled = scaler.transform(X_train_2d).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, channels)).reshape(X_test.shape)

        # Build CNN model (rebuild per-fold to avoid state carryover)
        tf.keras.backend.clear_session()
        model = build_cnn_embedding_model(input_samples=timesteps, input_channels=channels, embed_dim=embed_dim, dropout=getattr(cfg,"CNN_DROPOUT",0.4))
        # set optimizer LR
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cnn_lr),
                      loss={"softmax":"sparse_categorical_crossentropy"},
                      metrics={"softmax":"accuracy"})

        # callbacks
        ckpt = os.path.join(save_dir, f"cnn_fold{fold_idx}.h5")
        cb = [
            tf.keras.callbacks.EarlyStopping(monitor="val_softmax_accuracy", patience=cnn_patience, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_softmax_accuracy", save_best_only=True, save_weights_only=False, verbose=0)
        ]

        # Train CNN (supervised) on epochs (train set)
        history = model.fit(
            X_train_scaled, {"softmax": y_train, "embedding": np.zeros((len(train_idx), embed_dim))}, # embedding unused in loss
            validation_split=0.1,
            epochs=cnn_epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose=2
        )

        # Extract embedding model and compute embeddings
        emb_model = get_embedding_model(model)
        emb_train = extract_embeddings(emb_model, X_train_scaled, batch_size=batch_size)
        emb_test  = extract_embeddings(emb_model, X_test_scaled, batch_size=batch_size)

        if save_embeddings:
            np.savez_compressed(os.path.join(emb_dir, f"emb_fold{fold_idx}.npz"),
                                emb_train=emb_train, emb_test=emb_test, y_train=y_train, y_test=y_test, train_idx=train_idx, test_idx=test_idx)

        # Train RF on embeddings
        rf = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_depth, random_state=random_state, n_jobs=-1)
        rf.fit(emb_train, y_train)
        y_pred = rf.predict(emb_test)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        print(f"Fold {fold_idx} F1_macro: {f1:.4f}  (test samples: {len(test_idx)})")

        # optionally save RF
        joblib.dump(rf, os.path.join(save_dir, f"rf_fold{fold_idx}.joblib"))

        fold_results.append({"fold": fold_idx, "f1_macro": float(f1), "n_train": len(train_idx), "n_test": len(test_idx)})

    total_time = time.time() - total_start
    mean_f1 = float(np.mean([r["f1_macro"] for r in fold_results]))
    std_f1 = float(np.std([r["f1_macro"] for r in fold_results]))
    print(f"\nHybrid training complete. mean F1_macro: {mean_f1:.4f} ± {std_f1:.4f}  (time {total_time:.1f}s)")

    return {"folds": fold_results, "mean_f1": mean_f1, "std_f1": std_f1}
