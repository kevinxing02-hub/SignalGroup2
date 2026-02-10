"""
compute_nonlinear_features.py

Exports:
    compute_nonlinear_features(epoch, fs, recalc_params=None) -> dict
    vectorized_nonlinear_features(epochs, fs, recalc_params=None) -> np.ndarray

Notes:
 - epoch: 1D numpy array (single epoch)
 - epochs: 2D numpy array, shape (n_epochs, n_samples)
 - fs: sampling frequency in Hz (used for methods that need scale)
 - recalc_params: optional dict to tune methods (m, r for sampen, emb for perm, k_max for higuchi etc.)
"""

import numpy as np

# Try imports; prefer antropy and nolds
try:
    import antropy as ant
except Exception:
    ant = None

try:
    import nolds
except Exception:
    nolds = None

from scipy.signal import detrend

# ---------- Small helpers ----------
def _safe_std(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.std(x))

def _check_length_min(x, min_len=50):
    if len(x) < min_len:
        # Many of these methods need some minimum length; return False so caller can skip or pad.
        return False
    return True

# ---------- Feature functions (use antropy/nolds where available) ----------
def sample_entropy(x, m=2, r=None):
    """Sample entropy (SampEn). r defaults to 0.2 * std(x)."""
    x = np.asarray(x, dtype=float)
    if r is None:
        r = 0.2 * _safe_std(x)
    if ant is not None:
        try:
            return float(ant.sample_entropy(x, order=m, r=r))
        except Exception:
            pass
    if nolds is not None:
        try:
            # nolds.sampen returns sample entropy
            return float(nolds.sampen(x, emb_dim=m, metric='chebyshev', r=r))
        except Exception:
            pass
    # Fallback: simple implementation for m=2 (approx). Not as robust as libraries.
    # Uses the classic SampEn algorithm (O(N^2)).
    N = len(x)
    if N < m + 1:
        return np.nan
    if r <= 0:
        r = 0.2 * _safe_std(x) + 1e-12

    def _phi(m_):
        C = 0
        for i in range(N - m_):
            xi = x[i : i + m_]
            for j in range(i + 1, N - m_ + 1):
                xj = x[j : j + m_]
                if np.max(np.abs(xi - xj)) <= r:
                    C += 1
        return C

    try:
        B = _phi(m)
        A = _phi(m + 1)
        if B == 0:
            return np.inf
        return float(-np.log(A / B)) if A > 0 else np.inf
    except Exception:
        return np.nan

def permutation_entropy(x, order=3, delay=1, normalize=True):
    """Permutation entropy (Bandt & Pompe)."""
    x = np.asarray(x, dtype=float)
    if ant is not None:
        try:
            return float(ant.perm_entropy(x, order=order, delay=delay, normalize=normalize))
        except Exception:
            pass
    # Fallback implementation (small, straightforward)
    from math import factorial
    N = len(x)
    if N < order * delay:
        return np.nan
    # build ordinal patterns
    patterns = {}
    for i in range(N - (order - 1) * delay):
        window = x[i : i + order * delay : delay]
        ranks = tuple(np.argsort(window).tolist())
        patterns[ranks] = patterns.get(ranks, 0) + 1
    ps = np.array(list(patterns.values()), dtype=float)
    ps = ps / ps.sum()
    ent = -np.sum(ps * np.log2(ps + 1e-12))
    if normalize:
        ent /= np.log2(factorial(order))
    return float(ent)

def approximate_entropy(x, m=2, r=None):
    """Approximate Entropy (ApEn) — uses antropy if available, else basic implementation."""
    x = np.asarray(x, dtype=float)
    if r is None:
        r = 0.2 * _safe_std(x)
    if ant is not None:
        try:
            return float(ant.app_entropy(x, order=m, metric='chebyshev', r=r))
        except Exception:
            pass
    # Very basic ApEn fallback (not optimized)
    # Implementation adapted from standard definition
    N = len(x)
    if N <= m + 1:
        return np.nan
    def _phi(m_):
        C = []
        for i in range(N - m_ + 1):
            xi = x[i:i+m_]
            count = 0
            for j in range(N - m_ + 1):
                xj = x[j:j+m_]
                if np.max(np.abs(xi - xj)) <= r:
                    count += 1
            C.append(count / (N - m_ + 1.0))
        return np.mean(np.log(np.array(C) + 1e-12))
    try:
        return float(_phi(m) - _phi(m + 1))
    except Exception:
        return np.nan

def higuchi_fd(x, kmax=10):
    if ant is not None:
        try:
            return float(ant.higuchi_fd(x, kmax=kmax))
        except Exception:
            pass
    if nolds is not None and hasattr(nolds, 'higuchi_fd'):
        try:
            return float(nolds.higuchi_fd(x, kmax=kmax))
        except Exception:
            pass
    # fallback: return nan (Higuchi is nontrivial to implement robustly)
    return np.nan

def katz_fd(x):
    if ant is not None:
        try:
            return float(ant.katz_fd(x))
        except Exception:
            pass
    # fallback Katz formula
    x = np.asarray(x, dtype=float)
    L = np.sum(np.sqrt(np.diff(x) ** 2))  # curve length
    d = np.max(np.abs(x - x[0]))
    N = len(x)
    if d == 0 or N <= 1:
        return np.nan
    return float(np.log10(N) / (np.log10(N) + np.log10(d / L + 1e-12)))

def detrended_fluctuation(x, nvals=None):
    """DFA using nolds if available."""
    x = np.asarray(x, dtype=float)
    if nolds is not None:
        try:
            return float(nolds.dfa(x))
        except Exception:
            pass
    if ant is not None and hasattr(ant, 'detrended_fluctuation'):  # antropy has dfa?
        try:
            return float(ant.detrended_fluctuation(x))
        except Exception:
            pass
    return np.nan

def lziv_complexity(x):
    if ant is not None:
        try:
            return float(ant.lziv_complexity(x))
        except Exception:
            pass
    # fallback: naive LZC on binary thresholded signal
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    med = np.median(x)
    s = ''.join(['1' if v > med else '0' for v in x])
    i, k, l = 0, 1, 1
    c = 1
    n = len(s)
    while True:
        if i + k >= n:
            c += 1
            break
        if s[i:i + k] == s[l:l + k]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > l:
                i += 1
                k = 1
            else:
                l += 1
                c += 1
                k = 1
        if l >= n:
            break
    return float(c)

def hurst_exponent(x):
    if nolds is not None:
        try:
            return float(nolds.hurst_rs(x))
        except Exception:
            pass
    # fallback: use simple rescaled range estimate (rough)
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 20:
        return np.nan
    X = np.cumsum(x - np.mean(x))
    R = np.max(X) - np.min(X)
    S = np.std(x)
    if S == 0:
        return np.nan
    return float(np.log(R / S + 1e-12) / np.log(N))

# ---------- Top-level API ----------
def compute_nonlinear_features(epoch, fs, recalc_params=None):
    """
    Compute a dictionary of nonlinear features for a single epoch (1D array).
    Returns a dict: { 'sampen':..., 'perm_entropy':..., ... }
    """
    epoch = np.asarray(epoch, dtype=float)
    recalc_params = recalc_params or {}
    m = recalc_params.get('samp_m', 2)
    r = recalc_params.get('samp_r', 0.2 * _safe_std(epoch))
    perm_order = recalc_params.get('perm_order', 3)
    perm_delay = recalc_params.get('perm_delay', 1)
    higuchi_kmax = recalc_params.get('higuchi_kmax', 10)

    out = {}

    # only compute if epoch length is reasonable
    if not _check_length_min(epoch, min_len=30):
        # Return NaNs if too short
        out_keys = ['sampen', 'perm_entropy', 'app_entropy', 'higuchi_fd',
                    'katz_fd', 'dfa', 'lziv', 'hurst']
        for k in out_keys:
            out[k] = np.nan
        return out

    # remove linear trend for some measures (optional)
    x = detrend(epoch)

    out['sampen'] = sample_entropy(x, m=m, r=r)
    out['perm_entropy'] = permutation_entropy(x, order=perm_order, delay=perm_delay, normalize=True)
    out['app_entropy'] = approximate_entropy(x, m=m, r=r)
    out['higuchi_fd'] = higuchi_fd(x, kmax=higuchi_kmax)
    out['katz_fd'] = katz_fd(x)
    out['dfa'] = detrended_fluctuation(x)
    out['lziv'] = lziv_complexity(x)
    out['hurst'] = hurst_exponent(x)

    return out

def vectorized_nonlinear_features(epochs, fs, recalc_params=None, feature_order=None):
    """
    epochs: 2D array (n_epochs, n_samples).
    Returns: ndarray shape (n_epochs, n_features) and list of feature names.
    """
    epochs = np.asarray(epochs)
    n_epochs = epochs.shape[0]
    recalc_params = recalc_params or {}

    # compute on first epoch to get keys
    sample = compute_nonlinear_features(epochs[0], fs, recalc_params=recalc_params)
    keys = list(sample.keys()) if feature_order is None else feature_order
    arr = np.zeros((n_epochs, len(keys)), dtype=float)

    for i in range(n_epochs):
        res = compute_nonlinear_features(epochs[i], fs, recalc_params=recalc_params)
        arr[i, :] = [res.get(k, np.nan) for k in keys]

    return arr, keys

# End of file
