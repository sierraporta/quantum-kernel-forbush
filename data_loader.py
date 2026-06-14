"""
data_loader.py
===================
Loader adaptado a los archivos reales:

  OMNI  → columnas: 'Scalar B, nT', 'SW Plasma Temperature, K',
                    'SW Proton Density, N/cm3', 'SW Plasma Speed, kms',
                    'Kp index', 'Dst-index, nT', 'f10.7_index'
          índice: datetime

  JUNG  → columna: '1HCOR_E_JUNG'
          índice: datetime

  FEID  → columnas: 'OType','TVMax','VMax','Bzmin','Vpart','VmHm',
                    'THMax','HMax','Hpart','Betamin','BzmtoBm','ABzmax'
          índice: datetime  (onset del evento)
          → TODOS los registros son FD confirmados (label=1)
          → los no-eventos (label=0) se generan muestreando
            ventanas "quietas" (sin evento ±72h)

Uso:
    from data_loader_real import build_real_dataset
    X_raw, y, meta = build_real_dataset(omni, jung, feid)
"""

import numpy as np
import pandas as pd
from datetime import timedelta


# ── Parámetros de ventana ────────────────────────────────────────────────────
WINDOW_BEFORE_H = 24
WINDOW_AFTER_H  = 48
TOTAL_H         = WINDOW_BEFORE_H + WINDOW_AFTER_H   # 72 pasos + el onset = 73

# Columnas de OMNI que usaremos como features
OMNI_COLS = [
    'Scalar B, nT',
    "BX, nT (GSE)",
    "BY, nT (GSE)",
    "BZ, nT (GSE)",
    'SW Plasma Speed, kms',
    'SW Plasma Temperature, K',
    'Kp index',
    'Dst-index, nT',
    'SW Proton Density, N/cm3',
    'f10.7_index',
]

JUNG_COL  = '1HCOR_E_JUNG'

# Columnas de FEID para metadatos (no entran como features, pero son útiles)
FEID_META_COLS = ['OType', 'HMax', 'VMax', 'Bzmin']


# ── Funciones de limpieza ────────────────────────────────────────────────────
def clean_omni(omni: pd.DataFrame, fill_value: float = np.nan) -> pd.DataFrame:
    """
    OMNI tiene valores de relleno típicos (9999.9, 999.9, 99999, etc.).
    Los reemplazamos por NaN para luego interpolar.
    """
    df = omni[OMNI_COLS].copy()
    # Umbrales típicos de OMNI para valores faltantes
    thresholds = {
        'Scalar B, nT'            : 999.9,
        'BX, nT (GSE)'            : 999.9,
        'BY, nT (GSE)'            : 999.9,
        'BZ, nT (GSE)'            : 999.9,
        'SW Plasma Speed, kms'    : 9999.0,
        'SW Plasma Temperature, K' : 9999999.,
        'Kp index'                : 99.0,
        'Dst-index, nT'           : 99999.0,
        'SW Proton Density, N/cm3': 999.9,
        'f10.7_index'             : 999.9,
    }
    for col, thresh in thresholds.items():
        if col in df.columns:
            df.loc[df[col] >= thresh, col] = np.nan
    return df


def clean_jung(jung: pd.DataFrame) -> pd.DataFrame:
    """Elimina valores negativos o extremos del monitor de neutrones."""
    df = jung[[JUNG_COL]].copy()
    df.loc[df[JUNG_COL] <= 0, JUNG_COL] = np.nan
    # Outliers: más de 5 sigmas de la mediana
    med = df[JUNG_COL].median()
    std = df[JUNG_COL].std()
    df.loc[(df[JUNG_COL] - med).abs() > 5 * std, JUNG_COL] = np.nan
    return df


# ── Extracción de ventana ────────────────────────────────────────────────────
def _get_window(
    onset: pd.Timestamp,
    omni: pd.DataFrame,
    jung: pd.DataFrame,
    max_nan_frac: float = 0.20,
) -> np.ndarray | None:
    """
    Extrae ventana [onset - 24h, onset + 48h] y retorna array (73, 7).
    Retorna None si hay demasiados NaN.
    """
    t0 = onset - timedelta(hours=WINDOW_BEFORE_H)
    t1 = onset + timedelta(hours=WINDOW_AFTER_H)

    omni_w = omni.loc[t0:t1]
    jung_w = jung.loc[t0:t1]
    window = omni_w.join(jung_w, how='outer')

    # Verificar cobertura temporal mínima
    expected_len = TOTAL_H + 1  # 73
    if len(window) < 0.8 * expected_len:
        return None

    # Rechazar si hay demasiados NaN
    if window.isna().mean().mean() > max_nan_frac:
        return None

    # Interpolar huecos pequeños
    window = (window
              .interpolate(method='time')
              .ffill()
              .bfill())

    # Reindexar a exactamente 73 puntos horarios
    full_idx = pd.date_range(t0, t1, freq='1h')
    window   = window.reindex(full_idx).interpolate(method='time').ffill().bfill()

    return window.values.astype(np.float32)  # shape (73, 7)


# ── Generación de no-eventos ─────────────────────────────────────────────────
def _sample_quiet_windows(
    omni: pd.DataFrame,
    jung: pd.DataFrame,
    feid_dates: pd.DatetimeIndex,
    n_samples: int,
    min_gap_h: int = 72,
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Muestrea ventanas aleatorias que NO solapen con ningún evento FEID.
    Estas son los no-eventos (label=0).
    """
    rng          = np.random.default_rng(seed)
    all_times    = omni.index
    quiet_windows = []
    attempts      = 0
    max_attempts  = n_samples * 20

    while len(quiet_windows) < n_samples and attempts < max_attempts:
        attempts += 1
        # Elegir timestamp aleatorio dentro del rango de OMNI
        rand_idx = rng.integers(0, len(all_times))
        candidate = all_times[rand_idx]

        # Verificar que no haya ningún evento FEID a menos de min_gap_h
        diffs = np.abs((feid_dates - candidate)).total_seconds() / 3600
        if diffs.min() < min_gap_h:
            continue

        win = _get_window(candidate, omni, jung)
        if win is not None:
            quiet_windows.append(win)

    print(f"[data_loader_real] No-eventos generados: {len(quiet_windows)} "
          f"(intentos: {attempts})")
    return quiet_windows


# ── Pipeline principal ───────────────────────────────────────────────────────
def build_real_dataset(
    omni_raw: pd.DataFrame,
    jung_raw: pd.DataFrame,
    feid_raw: pd.DataFrame,
    min_magn: float = 0.0,     # filtrar FDs débiles (ej. HMax > 3.0 para solo significativos)
    max_nan_frac: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Construye dataset real para el clasificador.

    Parameters
    ----------
    omni_raw   : DataFrame con índice datetime y columnas OMNI
    jung_raw   : DataFrame con índice datetime y columna JUNG
    feid_raw   : DataFrame con índice datetime y columnas FEID
    min_magn   : magnitud mínima de FD a incluir (filtra eventos muy débiles)

    Returns
    -------
    X_raw : np.ndarray, shape (n_samples, 73, 7)
    y     : np.ndarray, shape (n_samples,)  — 1=FD, 0=no-FD
    meta  : pd.DataFrame con metadatos de cada evento (HMax, OType, etc.)
    """
    # Limpiar
    omni = clean_omni(omni_raw)
    jung = clean_jung(jung_raw)

    # Filtrar FEID por magnitud mínima
    feid = feid_raw.copy()
    if min_magn > 0:
        feid = feid[feid['Magn'] >= min_magn]
        print(f"[data_loader_real] FEID filtrado (HMax >= {min_magn}%): {len(feid)} eventos")

    # Limitar al rango temporal cubierto por OMNI y JUNG
    t_start = max(omni.index.min(), jung.index.min()) + timedelta(hours=WINDOW_BEFORE_H)
    t_end   = min(omni.index.max(), jung.index.max()) - timedelta(hours=WINDOW_AFTER_H)
    feid    = feid[(feid.index >= t_start) & (feid.index <= t_end)]
    print(f"[data_loader_real] Eventos FEID en rango temporal: {len(feid)}")

    # ── Extraer ventanas FD (label=1) ────────────────────────────────────────
    fd_windows, fd_meta = [], []
    skipped = 0
    for onset, row in feid.iterrows():
        win = _get_window(onset, omni, jung, max_nan_frac)
        if win is None:
            skipped += 1
            continue
        fd_windows.append(win)
        fd_meta.append({'datetime': onset, 'label': 1,
                        **{c: row[c] for c in FEID_META_COLS if c in row.index}})

    print(f"[data_loader_real] Ventanas FD válidas: {len(fd_windows)} "
          f"({skipped} descartadas por NaN)")

    # ── Generar no-eventos (label=0) — mismo número que FDs ─────────────────
    n_fd = len(fd_windows)
    quiet_windows = _sample_quiet_windows(
        omni, jung, feid.index, n_samples=n_fd, seed=seed
    )
    nfd_meta = [{'datetime': pd.NaT, 'label': 0,
                 'OType': 0, 'HMax': 0.0, 'VMax': 0.0, 'Bzmin': 0.0}
                for _ in quiet_windows]

    # ── Combinar y mezclar ────────────────────────────────────────────────────
    all_windows = fd_windows + quiet_windows
    all_labels  = [1] * len(fd_windows) + [0] * len(quiet_windows)
    all_meta    = fd_meta + nfd_meta

    X_raw = np.array(all_windows)   # (n, 73, 7)
    y     = np.array(all_labels)    # (n,)
    meta  = pd.DataFrame(all_meta)

    # Mezclar aleatoriamente
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X_raw, y, meta = X_raw[idx], y[idx], meta.iloc[idx].reset_index(drop=True)

    print(f"\n[data_loader_real] Dataset final: {X_raw.shape}")
    print(f"  FD (1): {y.sum()}  |  No-FD (0): {(y==0).sum()}")
    print(f"  Canales: {OMNI_COLS + [JUNG_COL]}")
    print(f"  Ventana: {WINDOW_BEFORE_H}h antes → {WINDOW_AFTER_H}h después del onset\n")

    return X_raw, y, meta


# ── Análisis rápido del dataset ──────────────────────────────────────────────
def describe_dataset(X_raw: np.ndarray, y: np.ndarray, meta: pd.DataFrame):
    """Imprime estadísticas básicas del dataset construido."""
    print("=" * 55)
    print("  ESTADÍSTICAS DEL DATASET REAL")
    print("=" * 55)
    print(f"  Shape X     : {X_raw.shape}")
    print(f"  FD events   : {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  No-FD events: {(y==0).sum()} ({100*(1-y.mean()):.1f}%)")

    fd_meta = meta[meta['label'] == 1]
    if 'HMax' in fd_meta.columns:
        print(f"\n  FD magnitude (HMax %):")
        print(f"    min={fd_meta['HMax'].min():.1f}  "
              f"median={fd_meta['HMax'].median():.1f}  "
              f"max={fd_meta['HMax'].max():.1f}")
    if 'OType' in fd_meta.columns:
        print(f"\n  OType distribution:")
        print(fd_meta['OType'].value_counts().to_string())

    # NaN residuales
    nan_frac = np.isnan(X_raw).mean()
    print(f"\n  NaN residuales en X: {nan_frac:.4f} (debería ser ~0)")
    print("=" * 55)
