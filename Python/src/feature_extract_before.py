# config.py  (建议替换你现有的 config.py)
"""
Project configuration for Sleep Scoring pipeline.
This file contains PREPROCESS / FEATURE_SELECTION tuning and general project settings.
Adjust the boxed sections below for experiments.
"""

import os
from pathlib import Path
import numpy as np

# ---------------------------
# Basic project paths & I/O
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = r"S:/SignalGoupWork/"           # <- 修改为你本地数据根路径（示例）
TRAINING_DIR = str(Path(DATA_DIR) / "training")
HOLDOUT_DIR = str(Path(DATA_DIR) / "holdout")
SAMPLE_DIR = str(Path(DATA_DIR) / "sample")
CACHE_DIR = str(PROJECT_ROOT / "cache")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs")

# ensure directories exist (create if missing)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# High-level run options
# ---------------------------
CURRENT_ITERATION = 4         # 1..4
MODEL_TYPE = "RF"            # "RF" / "CNN" / "HYBRID"

VERBOSE = 2
RANDOM_STATE = 42

# ---------------------------
# Preprocessing tunables (used by preprocessing.PreprocConfig and legacy code)
# ---------------------------
# These values are conservative defaults that preserve delta band and notch power.
PREPROCESS = {
    # Filtering
    "HP_CUTOFF": 0.1,         # high-pass cutoff (Hz) - keep low to preserve delta
    "LP_CUTOFF": 40.0,        # low-pass cutoff (Hz)
    "HP_ORDER": 2,
    "LP_ORDER": 4,
    # Notch
    "NOTCH_FREQ": 50.0,
    "NOTCH_Q": 30.0,
    "NOTCH_HARMONICS": 2,     # 50 and 100 (if below Nyquist)
    # filtfilt padding
    "PADTYPE": "odd",         # 'odd'|'even'|'constant'|None
    "PADLEN": None,           # None -> allow scipy default (safe)
    # EOG / EMG handling
    "DO_EOG_REGRESSION": True,
    "EOG_REGRESSION_PER_EPOCH": True,
    "DETECT_EMG_EPOCHS": True,
    "EMG_POWER_BAND": (20.0, 40.0),
    "EMG_POWER_PCT_THRESHOLD": 75.0,
    "APPLY_EMG_LOWPASS": False,    # default OFF to avoid delta attenuation
    "EMG_LOWPASS_CUT": 30.0,
    # debug / output
    "DEBUG_PLOTS": True,            # set True to save PSD/edge plots (preproc test)
    "DEBUG_OUTPUTS_DIR": str(Path(OUTPUT_DIR) / "preproc_validation"),
}

# Note: preprocessing.PreprocConfig is independent; the PreprocConfig default will be used
# if you don't explicitly build it from PREPROCESS dict. But keeping keys here helps CLI/grid.

# ---------------------------
# Feature-extraction settings
# ---------------------------
# Defaults used by feature_extraction module
WELCH_WINDOW = "hann"
WELCH_SEGMENT_SEC = 4
WELCH_OVERLAP_SEC = 2
WELCH_NFFT = None

WAVELET_FAMILY = "db4"
WAVELET_LEVELS = 5

ENABLE_SAMPLE_ENTROPY = False
SIGMA_PER_CHANNEL = False    # sigma features aggregated by default

# ---------------------------
# Feature selection tuning (you will run gridsearch or use these defaults)
# ---------------------------
# Recommended start (based on your grid results)
FEATURE_SELECTION_ENABLED = True
FEATURE_SELECTION_SCALE = True         # RobustScaler pre-selection (recommended)
FEATURE_SELECTION_TOP_K = 40           # top-K MI after variance/corr pruning
VARIANCE_THRESHOLD_RATIO = 1e-5        # fraction of max variance to threshold (small keeps more features)
CORRELATION_THRESHOLD = 0.90           # drop features with |corr| > this (keep highest-variance)
FEAT_STABILITY_THRESHOLD = 0.80       # fraction of folds the feature must appear in to be considered "stable"
LOSO_ENABLED = True                    # whether to use LOSO aggregation inside select_features
FEATURE_SELECTION_MIN_FEATURES = 100   # legacy: skip selection for tiny feature sets (iter 2 behavior)
RANDOM_STATE = RANDOM_STATE

# ---------------------------
# Classifier / training defaults (safe defaults for RF)
# ---------------------------
if CURRENT_ITERATION == 1:
    CLASSIFIER_TYPE = "knn"
elif CURRENT_ITERATION == 2:
    CLASSIFIER_TYPE = "svm"
elif CURRENT_ITERATION in (3,4):
    CLASSIFIER_TYPE = "random_forest"
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = 20
    RF_MIN_SAMPLES_SPLIT = 2
    RF_MIN_SAMPLES_LEAF = 2
    RF_CLASS_WEIGHT = "balanced"
else:
    CLASSIFIER_TYPE = "random_forest"

# ---------------------------
# CNN / hybrid placeholders (only used if MODEL_TYPE set to CNN or HYBRID)
# ---------------------------
MODEL_TYPE = MODEL_TYPE
RECORD_IDS = None  # set before training if using CNN/HYBRID (length == n_epochs)

# ---------------------------
# Logging / debugging toggles (fixes the earlier AttributeError)
# ---------------------------
DEBUG = True
DEBUG_PLOTS = PREPROCESS["DEBUG_PLOTS"]     # some modules expect config.DEBUG_PLOTS
PREPROCESS_OUTPUT_DIR = PREPROCESS["DEBUG_OUTPUTS_DIR"]

# ---------------------------
# Save / checkpointing
# ---------------------------
SAVE_TRAINING_CHECKPOINTS = False
CHECKPOINT_DIR = str(Path(OUTPUT_DIR) / "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------
# Some convenience functions for your scripts (optional)
# ---------------------------
def build_preproc_config():
    """
    Helper: build a preprocessing.PreprocConfig-like object (for code that expects dataclass).
    Usage: from config import build_preproc_config; cfg = build_preproc_config()
    """
    class _C:
        pass
    c = _C()
    c.highpass = PREPROCESS["HP_CUTOFF"]
    c.lowpass = PREPROCESS["LP_CUTOFF"]
    c.hp_order = PREPROCESS["HP_ORDER"]
    c.lp_order = PREPROCESS["LP_ORDER"]
    c.notch_freq = PREPROCESS["NOTCH_FREQ"]
    c.notch_Q = PREPROCESS["NOTCH_Q"]
    c.notch_harmonics = PREPROCESS["NOTCH_HARMONICS"]
    c.padtype = PREPROCESS["PADTYPE"]
    c.padlen = PREPROCESS["PADLEN"]
    c.do_eog_regression = PREPROCESS["DO_EOG_REGRESSION"]
    c.eog_regression_per_epoch = PREPROCESS["EOG_REGRESSION_PER_EPOCH"]
    c.detect_emg_epochs = PREPROCESS["DETECT_EMG_EPOCHS"]
    c.emg_power_band = PREPROCESS["EMG_POWER_BAND"]
    c.emg_power_pct_threshold = PREPROCESS["EMG_POWER_PCT_THRESHOLD"]
    c.apply_emg_lowpass = PREPROCESS["APPLY_EMG_LOWPASS"]
    c.emg_lowpass_cut = PREPROCESS["EMG_LOWPASS_CUT"]
    c.debug_plots = PREPROCESS["DEBUG_PLOTS"]
    c.outputs_dir = PREPROCESS["DEBUG_OUTPUTS_DIR"]
    return c

# ---------------------------
# Quick note:
#   - If you want to try the strict "best" config from the grid (stability=1.0),
#     change FEAT_STABILITY_THRESHOLD = 1.0 below before running pipeline.
# ---------------------------

