# run_inference.py (修改版)
import os
import traceback
import numpy as np
import config

from src.data_loader import load_holdout_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
# 保留原来的 inference helper（备用），但我们在此处做更具兼容性的推理逻辑
from src.inference import generate_submission_file
from src.utils import load_cache, save_cache

# 尝试导入 tensorflow，仅用于检测 Keras 模型与提取 embedding；若不可用，会降级处理
try:
    import tensorflow as tf
    from tensorflow.keras import Model as KerasModel
except Exception:
    tf = None
    KerasModel = None


def normalize_preprocessed_output(obj):
    """
    Ensure preprocess() output is compatible with extract_features().
    - Prefer dict with 'eeg' key
    - Accept ndarray (single-channel)
    - If tuple/list: try to extract dict or ndarray
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "shape"):
        return obj

    if isinstance(obj, (tuple, list)):
        # Prefer dict with 'eeg'
        for item in obj:
            if isinstance(item, dict) and 'eeg' in item:
                return item
        # Prefer ndarray
        for item in obj:
            if hasattr(item, "shape"):
                return item
        # fallback: first element
        return obj[0]

    return obj


def _ensure_channel_info(record_info):
    """
    Ensure record_info contains minimal channel_info keys expected by preprocess.
    Returns a dict with keys like 'eeg_fs', 'eog_fs', 'emg_fs', 'epoch_length', and channel names.
    """
    if record_info is None:
        return {
            "epoch_length": 30,
            "eeg_fs": 125,
            "eog_fs": 125,
            "emg_fs": 125,
            "eeg_names": [], 
            "eog_names": [],
            "emg_names": []
        }

    # copy and fill defaults
    info = dict(record_info)
    info.setdefault("epoch_length", record_info.get("epoch_length", 30))
    # sampling_rates in your loader are under 'sampling_rates'
    sr = record_info.get("sampling_rates", {})
    info.setdefault("eeg_fs", sr.get("eeg", record_info.get("eeg_fs", 125)))
    info.setdefault("eog_fs", sr.get("eog", record_info.get("eog_fs", 125)))
    info.setdefault("emg_fs", sr.get("emg", record_info.get("emg_fs", 125)))

    # channel lists
    info.setdefault("eeg_names", record_info.get("channels", []) if record_info.get("channels") else record_info.get("eeg_names", []))
    info.setdefault("eog_names", record_info.get("eog_names", []))
    info.setdefault("emg_names", record_info.get("emg_names", []))

    return info


# --------------- 辅助检测函数 ---------------
def is_sklearn_estimator(obj):
    """Rudimentary check whether obj is an sklearn estimator (has get_params or predict and not Keras)."""
    if obj is None:
        return False
    if KerasModel is not None and isinstance(obj, KerasModel):
        return False
    return hasattr(obj, "predict") and hasattr(obj, "get_params")


def is_keras_model(obj):
    if KerasModel is None:
        return False
    return isinstance(obj, KerasModel)


def try_auto_reshape_flat_to_3d(X_flat):
    """
    Heuristic to reshape (n_samples, n_features) -> (n_samples, timesteps, channels).
    Tries divisors in set [1,2,3,4,6,8,16], returns reshaped array or None on failure.
    """
    if X_flat is None:
        return None
    if not hasattr(X_flat, "shape") or X_flat.ndim != 2:
        return None
    n_samples, n_features = X_flat.shape
    for ch in (1, 2, 3, 4, 6, 8, 16):
        if n_features % ch == 0 and (n_features // ch) >= 4:
            timesteps = n_features // ch
            try:
                reshaped = X_flat.reshape(n_samples, timesteps, ch)
                print(f"[INFO] Auto-reshaped flat features {X_flat.shape} -> {reshaped.shape} (channels={ch})")
                return reshaped
            except Exception:
                continue
    return None


def extract_embedding_from_cnn(cnn_model, X, batch_size=64):
    """
    从 Keras cnn_model 中找到名为 'embedding' 的层并返回该层输出（embeddings）。
    如果没有找到 embedding 层，则返回 None 并打印提示。
    """
    if KerasModel is None:
        raise RuntimeError("TensorFlow/Keras not available in this environment.")
    try:
        emb_layer = cnn_model.get_layer('embedding')
    except Exception:
        print("❌ CNN 模型中没有名为 'embedding' 的层，无法提取中间表示。")
        return None
    emb_model = tf.keras.Model(inputs=cnn_model.input, outputs=emb_layer.output)
    emb = emb_model.predict(X, batch_size=batch_size, verbose=0)
    return emb


# --------------- 主推理流程 ---------------
def run_inference():
    print(f"\n--- Sleep Scoring Inference - Iteration {config.CURRENT_ITERATION} ---")

    # === Load trained model bundle ===
    model_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
    model_bundle = load_cache(model_filename, config.CACHE_DIR)

    if model_bundle is None:
        print("❌ Error: No trained model found. Run main.py first.")
        return

    # model_bundle 预期包含至少 'model' 和 'scaler'（scaler 可为 None）
    print(f"[DEBUG] model_bundle keys: {list(model_bundle.keys())}")
    model = model_bundle.get("model")
    scaler = model_bundle.get("scaler", None)
    saved_model_type = model_bundle.get("model_type", None)  # 可选：在训练时若保存了 model_type 优先使用

    print(f"Loaded trained model & scaler from: {model_filename}\n")
    print(f"[DEBUG] Inferred saved_model_type: {saved_model_type}; model type: {type(model)}")

    # === Prepare final submission lists ===
    all_predictions = []
    all_record_ids = []
    all_epoch_ids = []

    # === Scan holdout directory ===
    print(f"Scanning holdout directory: {config.HOLDOUT_DIR}\n")
    edf_files = sorted([f for f in os.listdir(config.HOLDOUT_DIR) if f.lower().endswith(".edf")])

    if not edf_files:
        print("❌ No EDF files found in holdout directory.")
        return

    print(f"Found EDF files: {edf_files}\n")

    # ============================================================
    #   MAIN LOOP — PROCESS EACH HOLDOUT RECORD
    # ============================================================
    for edf_name in edf_files:
        record_id = os.path.splitext(edf_name)[0]
        edf_path = os.path.join(config.HOLDOUT_DIR, edf_name)

        print(f"\n=== Processing {record_id} ===")

        try:
            # ---------------------------------------------------------
            # 1. Load holdout EDF (returns tuple: (multi_channel_data, record_info))
            # ---------------------------------------------------------
            multi_channel_data, record_info = load_holdout_data(edf_path)

            if not isinstance(multi_channel_data, dict):
                raise ValueError(
                    f"load_holdout_data() returned unexpected type for data: {type(multi_channel_data)}"
                )

            # Ensure we have a usable channel_info for preprocess
            channel_info = _ensure_channel_info(record_info)

            # ---------------------------------------------------------
            # 2. Preprocessing
            # ---------------------------------------------------------
            cache_pre = f"preprocessed_{record_id}_iter{config.CURRENT_ITERATION}.joblib"
            preprocessed = load_cache(cache_pre, config.CACHE_DIR) if config.USE_CACHE else None

            if preprocessed is None:
                # pass channel_info into preprocess
                preprocessed = preprocess(multi_channel_data, config, channel_info=channel_info)
                if config.USE_CACHE:
                    save_cache(preprocessed, cache_pre, config.CACHE_DIR)

            # Normalize for feature extractor
            preprocessed = normalize_preprocessed_output(preprocessed)

            # ---------------------------------------------------------
            # 3. Feature extraction
            # ---------------------------------------------------------
            cache_feat = f"features_{record_id}_iter{config.CURRENT_ITERATION}.joblib"
            features = load_cache(cache_feat, config.CACHE_DIR) if config.USE_CACHE else None

            if features is None:
                features = extract_features(preprocessed, config)
                if config.USE_CACHE:
                    save_cache(features, cache_feat, config.CACHE_DIR)

            if not hasattr(features, "shape") or len(features.shape) != 2:
                raise RuntimeError(
                    f"Feature extraction failed for {record_id}. Got shape: {getattr(features, 'shape', None)}"
                )

            # ---------------------------------------------------------
            # 4. Apply saved selector (if any) and scaling
            # ---------------------------------------------------------
            # attempt to apply saved selector or selected_indices from model_bundle
            selector_obj = model_bundle.get("feature_selector", None)
            selected_indices = model_bundle.get("selected_indices", None)

            if selector_obj is not None:
                try:
                    features_for_scaling = selector_obj.transform(features)
                    print("✔ Applied saved feature_selector.transform() to features before scaling.")
                except Exception as ex:
                    print(f"❌ Saved feature_selector.transform failed: {ex}")
                    features_for_scaling = features
            elif selected_indices is not None:
                try:
                    sel = np.asarray(selected_indices, dtype=int)
                    features_for_scaling = features[:, sel]
                    print("✔ Applied saved selected_indices to features before scaling.")
                except Exception as ex:
                    print(f"❌ Applying selected_indices failed: {ex}")
                    features_for_scaling = features
            else:
                features_for_scaling = features

            # scaling if scaler present
            if scaler is not None:
                try:
                    if getattr(scaler, "n_features_in_", None) == getattr(features_for_scaling, "shape", (None, None))[1]:
                        features_scaled = scaler.transform(features_for_scaling)
                    else:
                        print("⚠️ scaler input features do not match expected shape after selector. Using unscaled features_for_scaling.")
                        features_scaled = features_for_scaling
                except Exception as ex:
                    print(f"❌ scaler.transform() raised: {ex}. Using unscaled features_for_scaling.")
                    features_scaled = features_for_scaling
            else:
                print("⚠️  Warning: No scaler found — using raw features.")
                features_scaled = features_for_scaling

            # ---------------------------------------------------------
            # 5. Predict — 支持 sklearn / keras / hybrid dict 三种情况
            # ---------------------------------------------------------
            predictions = None

            # If model_bundle explicitly saved model_type，优先使用它
            model_type = saved_model_type or None

            # try to infer model_type if not provided
            if model_type is None:
                if isinstance(model, dict) and ('rf' in model or 'cnn' in model):
                    model_type = "HYBRID"
                elif is_keras_model(model):
                    model_type = "CNN"
                elif is_sklearn_estimator(model):
                    model_type = "RF"
                else:
                    # fallback: attempt to use sklearn-like predict
                    model_type = "RF"

            print(f"[DEBUG] Using inferred model_type = {model_type}")

            # ------------------
            # Case: sklearn-like model (RF/SVM/etc.)
            # ------------------
            if model_type.upper() in ("RF", "SKLEARN", "SKLEARN_ESTIMATOR"):
                # standard sklearn pipeline: predict on features_scaled
                try:
                    predictions = model.predict(features_scaled)
                    print(f"✔ Predicted with sklearn model: {type(model)}")
                except Exception as ex:
                    raise RuntimeError(f"Sklearn model.predict failed: {ex}")

            # ------------------
            # Case: Keras end-to-end CNN
            # ------------------
            elif model_type.upper() == "CNN":
                if not is_keras_model(model):
                    # maybe stored as {'cnn': <model>} in this case extract
                    if isinstance(model, dict) and 'cnn' in model:
                        cnn = model['cnn']
                    else:
                        raise RuntimeError("配置表明为 CNN，但加载到的 model 不是 Keras 模型。请确认 model_bundle 内容（是否保存了 'cnn'）。")
                else:
                    cnn = model

                if not is_keras_model(cnn):
                    raise RuntimeError("CNN 推理需要 TensorFlow/Keras 模型，但找不到有效的模型对象。")

                # CNN 期待 3D 输入 (n, timesteps, channels)
                if features_scaled.ndim == 2:
                    X3 = try_auto_reshape_flat_to_3d(features_scaled)
                    if X3 is None:
                        raise RuntimeError(
                            "当前 CNN 模型需要 3D 时间序列输入，但你提供的是扁平特征 (n_samples, n_features)。\n"
                            "解决方案：\n"
                            "  1) 在训练时使用 CNN（需提供原始时序信号），并在 model_bundle 中保存适用的预处理/selector；\n"
                            "  2) 或者在此处提供原始时序数据并把它传入 CNN 进行推理；\n"
                            "  3) 若你意在使用 RF，请把 model_bundle 中的 model_type 设为 'RF'，或重新生成 RF 模型并保存到 model_bundle。"
                        )
                    else:
                        X_input = X3
                else:
                    X_input = features_scaled  # already 3D

                # CNN 直接输出 class probabilities or logits depending on model; take argmax
                try:
                    y_prob = cnn.predict(X_input, batch_size=getattr(config, "CNN_BATCH_SIZE", 64), verbose=0)
                    # if shape is (n, ) it's already classes; if 2D take argmax
                    if y_prob.ndim == 1:
                        predictions = y_prob.astype(int).tolist()
                    else:
                        predictions = np.argmax(y_prob, axis=1)
                    print("✔ Predicted with CNN model.")
                except Exception as ex:
                    raise RuntimeError(f"CNN predict failed: {ex}")

            # ------------------
            # Case: HYBRID (cnn + rf)
            # ------------------
            elif model_type.upper() == "HYBRID":
                # model can be either dict{'cnn':..., 'rf':...} or similar
                if not isinstance(model, dict):
                    raise RuntimeError("HYBRID 模型应以 dict 存储，包含 'cnn' 与 'rf' 键。")

                cnn = model.get('cnn', None)
                rf = model.get('rf', None)

                if rf is None:
                    raise RuntimeError("HYBRID 模型缺少 'rf' 部分，无法继续。")
                # If RF can accept features directly, try that first (fast path)
                try:
                    if hasattr(rf, "predict"):
                        # If RF expects same feature count as features_scaled, this will work.
                        rf_expected = getattr(rf, "n_features_in_", None)
                        if rf_expected is not None and rf_expected == features_scaled.shape[1]:
                            preds = rf.predict(features_scaled)
                            predictions = preds
                            print("✔ HYBRID: direct RF-on-features path succeeded.")
                        else:
                            # otherwise attempt embedding path if CNN present
                            if cnn is None:
                                # fallback: try to predict anyway and catch errors
                                preds = rf.predict(features_scaled)
                                predictions = preds
                                print("⚠️ HYBRID: RF.predict used despite feature-shape mismatch (may be incorrect).")
                            else:
                                # embedding path: need to feed CNN a 3D X and extract embedding
                                if not is_keras_model(cnn):
                                    raise RuntimeError("HYBRID 中的 cnn 不是 Keras 模型，无法提取 embedding。")
                                # prepare X for cnn: try reshape
                                if features_scaled.ndim == 2:
                                    X3 = try_auto_reshape_flat_to_3d(features_scaled)
                                    if X3 is None:
                                        raise RuntimeError(
                                            "HYBRID 模型需要通过 cnn 提取 embedding，但当前 features 是扁平向量，无法自动 reshape。\n"
                                            "请在训练时保存原始时序或在 model_bundle 中保存 'selected_indices' / 'feature_selector' 的信息，"
                                            "或在此处提供原始时序以便提取 embedding。"
                                        )
                                else:
                                    X3 = features_scaled
                                # extract embedding and feed rf
                                emb = extract_embedding_from_cnn(cnn, X3, batch_size=getattr(config, "CNN_BATCH_SIZE", 64))
                                if emb is None:
                                    raise RuntimeError("无法从 cnn 提取 embedding（embedding 层不存在或提取失败）。")
                                predictions = rf.predict(emb)
                                print("✔ HYBRID: embedding -> RF 路径成功。")
                    else:
                        raise RuntimeError("HYBRID.rf 对象没有 predict 方法。")
                except Exception as ex:
                    raise RuntimeError(f"HYBRID inference failed: {ex}")

            else:
                raise RuntimeError(f"不支持的 model_type: {model_type}")

            # ensure predictions is a 1D iterable of length n_epochs
            if predictions is None:
                raise RuntimeError("模型未返回预测值 (predictions is None).")
            predictions = list(np.asarray(predictions).reshape(-1).tolist())

            # ---------------------------------------------------------
            # 6. Store results
            # ---------------------------------------------------------
            n_epochs = len(predictions)
            all_predictions.extend(predictions)
            all_record_ids.extend([record_id] * n_epochs)
            all_epoch_ids.extend(list(range(n_epochs)))

            print(f"✔ Finished {record_id}: {n_epochs} epochs")

        except Exception as e:
            print(f"\n❌ ERROR processing {record_id}: {e}")
            traceback.print_exc()
            print("Skipping this record and continuing...\n")
            continue

    # ============================================================
    #   STEP 7 — Generate Submission
    # ============================================================
    if not all_predictions:
        print("❌ No predictions generated. Submission aborted.")
        return

    print("\nGenerating submission file...\n")

    generate_submission_file(
        all_predictions,
        all_record_ids,
        all_epoch_ids,
        config
    )

    print("\n--- Inference Completed Successfully ---\n")


if __name__ == "__main__":
    run_inference()
