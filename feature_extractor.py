"""
feature_extractor.py
====================
Convierte ventanas temporales multivariadas en vectores de features
compactos aptos para SVM clásico y kernel cuántico.

Estrategia: features estadísticos por canal + features de forma temporal.
El vector final se normaliza a [-1, 1] para compatibilidad con
los mapas de features cuánticos (ZZFeatureMap espera datos acotados).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ── Features por canal ───────────────────────────────────────────────────────
def _channel_features(series: np.ndarray) -> np.ndarray:
    """
    Extrae features estadísticos de una serie 1D:
    [mean, std, min, max, range, skewness, kurtosis, slope]
    """
    from scipy.stats import skew, kurtosis
    n = len(series)
    t = np.arange(n)
    slope = np.polyfit(t, series, 1)[0]  # tendencia lineal

    return np.array([
        np.mean(series),
        np.std(series),
        np.min(series),
        np.max(series),
        np.ptp(series),           # peak-to-peak
        skew(series),
        kurtosis(series),
        slope,
    ])


def _onset_features(series: np.ndarray, onset_idx: int) -> np.ndarray:
    """
    Features relativos al onset del evento:
    [valor en onset, caída máxima post-onset, tiempo hasta mínimo post-onset]
    """
    pre  = series[:onset_idx]
    post = series[onset_idx:]

    val_onset    = series[onset_idx] if onset_idx < len(series) else np.nan
    drop         = np.min(post) - val_onset if len(post) > 0 else 0.0
    t_to_min     = np.argmin(post) / max(len(post), 1)

    return np.array([val_onset, drop, t_to_min])


def extract_features(
    X_windows: np.ndarray,
    onset_frac: float = 24 / 73,  # WINDOW_BEFORE_H / total_window_len
    channel_names: list[str] | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    X_windows : np.ndarray, shape (n_samples, window_len, n_channels)
    onset_frac : fracción temporal donde ocurre el onset (default: 24/73)
    channel_names : nombres de canales (para logging)

    Returns
    -------
    X_feat : np.ndarray, shape (n_samples, n_features)
    """
    n_samples, window_len, n_channels = X_windows.shape
    onset_idx = int(onset_frac * window_len)

    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]

    feat_list = []
    for i in range(n_samples):
        sample_feats = []
        for c in range(n_channels):
            ch = X_windows[i, :, c]
            # Imputar NaN en la serie antes de calcular features
            if np.isnan(ch).any():
                ch = pd.Series(ch).interpolate(method='linear').ffill().bfill().values
            sample_feats.append(_channel_features(ch))
            sample_feats.append(_onset_features(ch, onset_idx))
        feat_list.append(np.concatenate(sample_feats))

    X_feat = np.array(feat_list)

    # Limpiar NaN/inf residuales (series constantes → skew/kurtosis = nan)
    nan_cols = np.isnan(X_feat).any(axis=0) | np.isinf(X_feat).any(axis=0)
    if nan_cols.any():
        print(f"[feature_extractor] Imputando {nan_cols.sum()} columnas con NaN/inf residual")
        col_means = np.nanmean(X_feat, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        X_feat = np.where(np.isnan(X_feat) | np.isinf(X_feat),
                          col_means[np.newaxis, :], X_feat)

    print(f"[feature_extractor] Shape de features: {X_feat.shape}  "
          f"({X_feat.shape[1]} features por muestra)")
    return X_feat


def normalize_for_quantum(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_range: tuple = (-np.pi, np.pi),
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Escala features a [-π, π] para ZZFeatureMap.
    El scaler se ajusta SOLO en train para evitar data leakage.

    Returns
    -------
    X_train_sc, X_test_sc, scaler
    """
    lo, hi = feature_range
    scaler = MinMaxScaler(feature_range=(lo, hi))
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print(f"[feature_extractor] Features escaladas a [{lo:.2f}, {hi:.2f}]")
    return X_train_sc, X_test_sc, scaler


def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Selecciona los n_features más relevantes por mutual information.
    Importante para kernels cuánticos: cada feature = 1 qubit.

    Returns
    -------
    X_reduced : np.ndarray, shape (n_samples, n_features)
    selected_idx : np.ndarray de índices seleccionados
    """
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(X, y, random_state=42)
    selected_idx = np.argsort(mi)[::-1][:n_features]
    X_reduced = X[:, selected_idx]

    print(f"[feature_extractor] Seleccionados {n_features} features "
          f"de {X.shape[1]} por mutual information")
    return X_reduced, selected_idx