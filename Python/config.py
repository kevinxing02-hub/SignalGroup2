# config.py  (修改版)
"""
Project configuration for Signal process.
已做的改动与说明：
- 保持原有参数与注释（向后兼容）。
- 新增 MODEL_TYPE、CNN/HYBRID 超参、CHECKPOINT/OUTPUT 配置、RECORD_IDS 占位（LOSO 必需）。
- 默认 MODEL_TYPE = "RF"（避免没有 TensorFlow 的环境报错）。如果要用 CNN/HYBRID，请把 MODEL_TYPE 改为 "CNN" 或 "HYBRID"
  并设置 RECORD_IDS（长度 == 样本数，每个 epoch 对应的 subject id）。
"""

# -- Project Configuration --
# Set the current iteration of the project (1-4).
# This controls which parts of the pipeline are active.
CURRENT_ITERATION = 4

# Normalization scheme used upstream (e.g. 'mean', 'zscore', 'none')
NORMALIZATION = 'mean'

# Use cached data for preprocessing and feature extraction.
USE_CACHE = False  # Temporarily disabled for testing with real data

# -- File Paths --
import os
DATA_DIR = 'S:/SignalGoupWork/'  # Kevin path: 'S:/SignalGoupWork'
TRAINING_DIR = f'{DATA_DIR}training'
HOLDOUT_DIR = f'{DATA_DIR}holdout'
SAMPLE_DIR = f'{DATA_DIR}sample/'
CACHE_DIR = 'cache/'
OUTPUT_DIR = './outputs/'  # 新增：默认输出路径（模型/可视化/检查点）

# Validate and create directories if needed
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}\nPlease ensure you are running from the correct directory.")
if not os.path.exists(CACHE_DIR):
    print(f"Creating cache directory: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)
if not os.path.exists(OUTPUT_DIR):
    print(f"Creating output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Preprocessing --
# ------------------------------
# Preprocessing parameters (tweak these for experiments)
# ------------------------------
# Recommended defaults chosen to:
#  - remove baseline wander without strongly attenuating delta (0.5-4 Hz)
#  - notch powerline at 50 Hz (Europe) and optionally harmonics
#  - use zero-phase filtering (filtfilt) with safe padding choices
PREPROCESS = {
    # Baseline (high-pass) cutoff in Hz. Keep <0.5 to preserve delta band.
    # Hardware HP at 0.15 Hz exists; software HP typically 0.2-0.5 Hz. 0.3 is a sensible default.
    "highpass": 0.30,

    # Final lowpass cutoff (or bandpass high edge) in Hz. 35-45 Hz typical for sleep EEG.
    "lowpass": 40.0,

    # Filter orders (for butterworth / sos). Higher order = steeper roll-off but more transients.
    "hp_order": 2,      # high-pass order (small to avoid strong transients)
    "bp_order": 4,      # bandpass / lowpass order

    # Notch filter settings
    "notch_freq": 50.0,       # fundamental (set 60.0 if US)
    "notch_Q": 30.0,          # quality factor for iirnotch (narrow but effective)
    "notch_harmonics": True,  # also apply at 2x, 3x if below Nyquist

    # filtfilt padding options (see scipy.signal.filtfilt)
    # padtype: 'odd' or 'even' recommended for biomedical signals; None disables padding.
    # padlen: None -> let scipy compute default padlen (3*(len(a)-1)) OR set explicit int
    # Setting an explicit padlen can help if you know signal length, but default is usually fine.
    "padtype": "odd",   # 'odd'/'even'/'constant'/None
    "padlen": None,     # None -> scipy default; or set e.g. int(3 * max_filter_len)

    # If you want to preprocess the continuous recording (recommended) rather than per-epoch
    "process_continuous": True,

    # If debug/diagnostic plots should be created in OUTPUT_DIR
    "debug_plots": False,

    # Directory to save debug/diagnostic plots (default uses project OUTPUT_DIR)
    "debug_outputs_dir": OUTPUT_DIR if 'OUTPUT_DIR' in globals() else "./outputs/",

    # Whether to apply simple per-epoch normalization (zero-mean, unit-variance)
    "apply_epoch_normalization": False,

    # Safety limits (to avoid accidentally aggressive settings)
    "min_highpass": 0.01,  # don't allow >0.5 recommended without conscious choice
    "max_lowpass": 0.5 * 125.0  # depends on expected global fs (e.g. 125Hz)
}

# Whether to apply per-channel normalization in preprocessing.py
# (e.g. zero-mean, unit-variance per epoch)
#APPLY_NORMALIZATION = False   # set True if you want to enable it

# -- Feature Extraction --
# (Add feature-specific parameters here)
# -- Feature Extraction (Iteration 2+) --

# ===============================
#   AR Model Feature Settings
# ===============================
AR_MODEL_ORDERS = [8, 10, 12, 14, 16]
AR_DEFAULT_ORDER = 16  # default for production
AR_METHOD = 'burg'  # always use Burg for EEG

# ===============================
#   Welch PSD Settings
# ===============================
WELCH_WINDOW = 'hann'
WELCH_SEGMENT_SEC = 4  # 4-second windows for 30s epoch (literature standard)
WELCH_OVERLAP_SEC = 2  # 50% overlap
WELCH_NFFT = None  # use auto FFT size unless overridden

# ===============================
#   Wavelet Transform Settings
# ===============================
WAVELET_FAMILY = 'db4'  # Daubechies-4: gold standard for sleep EEG
WAVELET_LEVELS = 5  # suitable for fs=125Hz (covers delta–beta range)

# ===============================
#   Feature Selection Settings
# ===============================
FEATURE_SELECTION_ENABLED = True
FEATURE_SELECTION_MIN_FEATURES = 100     # Only for Iteration 2
FEATURE_SELECTION_TOP_K = 40             # Final number of features kept
VARIANCE_THRESHOLD_RATIO = 1e-4          # Conservative variance threshold
CORRELATION_THRESHOLD = 0.95             # Remove features with |r| > threshold

# --------------------------
#   LOSO Feature Selection
# --------------------------
LOSO_ENABLED = True
LOSO_MAX_FOLDS = None
#FEAT_STABILITY_THRESHOLD = 1.0
GROUPS = None  # 可选：如果你希望在 config 中直接放 groups，否则在函数调用时传入
#
# feature selection tuning
FEATURE_SELECTION_ENABLED = True
FEATURE_SELECTION_SCALE = True  # whether to RobustScale before selection
FEATURE_SELECTION_TOP_K = 40
VARIANCE_THRESHOLD_RATIO = 1e-4
CORRELATION_THRESHOLD = 0.95
FEAT_STABILITY_THRESHOLD = 1.0  # prefer 0.8 over 1.0 for robustness
LOSO_ENABLED = True
RANDOM_STATE = 42


# Nonlinear features (Iteration 4 final)
ENABLE_NONLINEAR_FEATURES = True
NONLINEAR_PER_CHANNEL = False
NONLINEAR_MIN_EPOCH_LEN = 30
NONLINEAR_FEATURES = [
    "sampen", "perm_entropy", "app_entropy", "higuchi_fd",
    "katz_fd", "dfa", "lziv", "hurst"
]

# -- Classification --
# Iteration-specific parameters - students should modify these based on current iteration
if CURRENT_ITERATION == 1:
    CLASSIFIER_TYPE = 'knn'
    KNN_N_NEIGHBORS = 5
elif CURRENT_ITERATION == 2:
    CLASSIFIER_TYPE = 'svm'
    SVM_C = 1.0
    SVM_KERNEL = 'rbf'
elif CURRENT_ITERATION == 3:
    # Iteration 3: Multi-signal processing with Random Forest
    CLASSIFIER_TYPE = 'random_forest'
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
    RF_MIN_SAMPLES_SPLIT = 2

elif CURRENT_ITERATION == 4:
    CLASSIFIER_TYPE = 'random_forest'
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = 20
    RF_MIN_SAMPLES_SPLIT = 5
    RF_MIN_SAMPLES_LEAF = 2
    RF_CLASS_WEIGHT = 'balanced'
else:
    raise ValueError(f"Invalid CURRENT_ITERATION: {CURRENT_ITERATION}. Must be 1-4.")

# ------------------------------
# New settings for CNN / HYBRID
# ------------------------------
# Use MODEL_TYPE to select between RF, CNN, or HYBRID pipelines for Iteration 3/4 delegation.
# Default kept as 'RF' to be safe. Set to 'CNN' or 'HYBRID' to enable deep learning flows.
MODEL_TYPE = "RF"  # valid options: "RF", "CNN", "HYBRID"

# IMPORTANT: 如果你选 'CNN' 或 'HYBRID'，**必须**设置 RECORD_IDS（长度 = n_samples，每个 epoch 对应 subject id）。
# 示例： RECORD_IDS = np.repeat(np.arange(n_subjects), epochs_per_subject)[:n_samples]
RECORD_IDS = None  # 占位：在运行前请替换为实际的数组或在调用 train 函数时通过 config.record_ids 赋值

# CNN 超参
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 32
CNN_LR = 1e-4
CNN_DROPOUT = 0.4
CNN_PATIENCE = 8  # early stopping patience

# HYBRID（CNN embedding -> RF）相关（RF 超参可在上面覆盖）
EMBEDDING_LAYER_NAME = "embedding"  # 保持与 classification_cnn_rf 中一致

# 保存与日志
SAVE_TRAINING_CHECKPOINTS = False
VERBOSE = 2  # 0/1/2 for Keras verbosity

# 类名（可选，用于可视化）
CLASS_NAMES = None  # e.g. ['Wake','N1','N2','N3','REM']

# 其它（保持兼容）
SUBMISSION_FILE = 'submission.csv'

# ------------------------------
# Quick sanity checks & helper functions
# ------------------------------
def validate_for_deep_learning():
    """
    检查是否满足 CNN/HYBRID 运行的最低要求：
    - MODEL_TYPE 在 (CNN, HYBRID)
    - RECORD_IDS 已设置且长度正确（需要在调用 train 时和 features 长度匹配）
    - TensorFlow 可用（不会在这里 import TF，但在运行时 classification 模块会检查）
    """
    if MODEL_TYPE.upper() in ("CNN", "HYBRID"):
        if RECORD_IDS is None:
            raise ValueError("MODEL_TYPE is CNN/HYBRID but RECORD_IDS is None. 请在 config 中设置 RECORD_IDS（每个 epoch 的 subject id）。")
        # 不能在这里检查长度，因为 features 在 config 中不可见；调用训练时再做精确检查。

# (可选) 在模块导入时不强制执行 validate_for_deep_learning，以免造成导入错误；
# 在实际训练前，train 函数会再次检查 config.record_ids 的存在性。

