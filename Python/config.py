# -- Project Configuration --

# Set the current iteration of the project (1-4). 
# This controls which parts of the pipeline are active.
CURRENT_ITERATION = 3

# Set to True to use cached data for preprocessing and feature extraction.
USE_CACHE = False  # Temporarily disabled for testing with real data

# -- File Paths --
import os
DATA_DIR = './'
TRAINING_DIR = f'{DATA_DIR}TrainData'
HOLDOUT_DIR = f'{DATA_DIR}TestFeature'
SAMPLE_DIR = f'{DATA_DIR}sample/'
CACHE_DIR = 'cache/'

# Validate and create directories if needed
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}\nPlease ensure you are running from the correct directory.")
if not os.path.exists(CACHE_DIR):
    print(f"Creating cache directory: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

# -- Preprocessing --
LOW_PASS_FILTER_FREQ = 40  # Hz


# Whether to apply per-channel normalization in preprocessing.py
# (e.g. zero-mean, unit-variance per epoch)
APPLY_NORMALIZATION = True   # sätt True om du vill aktivera det

# -- Feature Extraction --
# (Add feature-specific parameters here)
# -- Feature Extraction (Iteration 2+) --


# ===============================
#   AR Model Feature Settings
# ===============================
# Candidate model orders to test; actual order chosen inside feature extraction
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

# Note: fs is passed dynamically from preprocessing


# ===============================
#   Wavelet Transform Settings
# ===============================
WAVELET_FAMILY = 'db4'  # Daubechies-4: gold standard for sleep EEG
WAVELET_LEVELS = 5  # suitable for fs=125Hz (covers delta–beta range)

# Feature_selection
FEATURE_SELECTION_TOP_K = 40

# -- Classification --
# Iteration-specific parameters - students should modify these based on current iteration
if CURRENT_ITERATION == 1:
    # Iteration 1: Basic pipeline with k-NN
    CLASSIFIER_TYPE = 'knn'
    KNN_N_NEIGHBORS = 5
elif CURRENT_ITERATION == 2:
    # Iteration 2: Enhanced EEG processing with SVM

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
    # Iteration 4: Full system optimization
    CLASSIFIER_TYPE = 'random_forest'
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = None
    RF_MIN_SAMPLES_SPLIT = 5
else:
    raise ValueError(f"Invalid CURRENT_ITERATION: {CURRENT_ITERATION}. Must be 1-4.")

# -- Submission --
SUBMISSION_FILE = 'submission.csv'