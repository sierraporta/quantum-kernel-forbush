"""
main_real.py
============
Pipeline completo: datos reales FEID / OMNI / JUNG
→ tres modelos comparados:

  [1] RBF-SVM (full)          — clásico con todo el training set
  [2] RBF-SVM (sub)           — clásico con MISMO subconjunto que quantum
  [3] Quantum Kernel SVM (sub) — kernel cuántico ZZFeatureMap

Uso:
    python main_v2.py \
        --omni omni_clean.pkl \
        --jung jung_clean.pkl \
        --feid feid_clean.pkl \
        --min_magn 3.0        \
        --n_qubits 8          \
        --reps 2              \
        --max_q 250           \
        --save_fig resultados.png
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader       import build_real_dataset, describe_dataset
from feature_extractor import extract_features, normalize_for_quantum, select_top_features
from classical_baseline import (train_classical_svm, evaluate_classifier,
                                cross_validate_svm, plot_confusion_matrix)
from quantum_kernel    import QuantumKernelSVM


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Quantum Kernel SVM vs RBF-SVM — Forbush Decrease Classifier"
    )
    p.add_argument("--omni",      required=True,  help="omni_clean.pkl o .csv")
    p.add_argument("--jung",      required=True,  help="jung_clean.pkl o .csv")
    p.add_argument("--feid",      required=True,  help="feid_clean.pkl o .csv")
    p.add_argument("--min_magn", type=float, default=3.0,
               help="Excluir FDs con Magn < valor (caída CR en %%, ej. 3.0)")
    p.add_argument("--n_qubits",  type=int,   default=8,
                   help="Features seleccionados = número de qubits")
    p.add_argument("--reps",      type=int,   default=2,
                   help="Profundidad del ZZFeatureMap")
    p.add_argument("--C",         type=float, default=1.0,
                   help="Parámetro C del SVM")
    p.add_argument("--test_size", type=float, default=0.25,
                   help="Fracción del dataset para test")
    p.add_argument("--max_q",     type=int,   default=250,
                   help="Máximo de muestras de train para el kernel cuántico")
    p.add_argument("--cv_folds",  type=int,   default=5,
                   help="Folds para cross-validation del RBF-SVM full")
    p.add_argument("--save_fig",  type=str,   default=None,
                   help="Ruta para guardar figura de matrices de confusión")
    p.add_argument("--save_csv",  type=str,   default="experiment_results.csv",
                   help="Ruta para guardar resultados en CSV")
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_df(path: str) -> pd.DataFrame:
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Submuestra estratificada balanceada de n muestras totales."""
    rng  = np.random.default_rng(seed)
    half = n // 2
    idx_0 = rng.choice(np.where(y == 0)[0], half, replace=False)
    idx_1 = rng.choice(np.where(y == 1)[0], half, replace=False)
    idx   = np.concatenate([idx_0, idx_1])
    rng.shuffle(idx)
    return X[idx], y[idx]


def print_section(title: str):
    print("\n" + "─" * 55)
    print(f"  {title}")
    print("─" * 55)


# ── Pipeline principal ────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── 1. Cargar datos ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  QUANTUM FD CLASSIFIER — datos reales")
    print("=" * 55)
    print("\n[main] Cargando archivos...")
    omni = load_df(args.omni)
    jung = load_df(args.jung)
    feid = load_df(args.feid)

    # ── 2. Construir dataset (FD vs no-FD) ──────────────────────────────────
    X_raw, y, meta = build_real_dataset(omni, jung, feid, min_magn=args.min_magn)
    describe_dataset(X_raw, y, meta)

    # ── 3. Extracción de features ────────────────────────────────────────────
    print_section("Extracción de features")
    X_feat = extract_features(X_raw)

    # ── 4. Selección top-n_qubits por mutual information ────────────────────
    X_sel, sel_idx = select_top_features(X_feat, y, n_features=args.n_qubits)

    # ── 5. Train / test split estratificado ─────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sel, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )
    print(f"\n[main] Train: {len(y_tr)} muestras  |  Test: {len(y_te)} muestras")

    # ── 6. Normalización [-π, π] ─────────────────────────────────────────────
    X_tr_sc, X_te_sc, scaler = normalize_for_quantum(X_tr, X_te)

    # ── 7. Submuestra para el quantum kernel ─────────────────────────────────
    n_q = min(args.max_q, len(X_tr_sc))
    if n_q < len(X_tr_sc):
        X_tr_q, y_tr_q = stratified_subsample(X_tr_sc, y_tr, n=n_q)
        print(f"[main] Submuestra quantum: {n_q} de {len(X_tr_sc)} muestras de train")
    else:
        X_tr_q, y_tr_q = X_tr_sc, y_tr
        print(f"[main] Usando training set completo para quantum ({n_q} muestras)")

    # ── 8. MODELO 1: RBF-SVM full ────────────────────────────────────────────
    print_section(f"MODELO 1 — RBF-SVM  (train n={len(y_tr)})")
    clf_rbf_full = train_classical_svm(X_tr_sc, y_tr, C=args.C)
    met_rbf_full = evaluate_classifier(
        clf_rbf_full, X_te_sc, y_te, name=f"RBF-SVM (n={len(y_tr)})"
    )
    print(f"\n[main] Cross-validation RBF-SVM full ({args.cv_folds}-fold):")
    cross_validate_svm(X_sel, y, C=args.C, n_splits=args.cv_folds)

    # ── 9. MODELO 2: RBF-SVM submuestra (comparación justa) ─────────────────
    print_section(f"MODELO 2 — RBF-SVM  (train n={n_q}, mismo subconjunto que QSVM)")
    clf_rbf_sub = train_classical_svm(X_tr_q, y_tr_q, C=args.C)
    met_rbf_sub = evaluate_classifier(
        clf_rbf_sub, X_te_sc, y_te, name=f"RBF-SVM (n={n_q})"
    )

    # ── 10. MODELO 3: Quantum Kernel SVM ─────────────────────────────────────
    print_section(
        f"MODELO 3 — Quantum Kernel SVM  "
        f"({args.n_qubits} qubits, reps={args.reps}, train n={n_q})"
    )
    print("[main] Calculando matriz de kernel cuántico (tarda varios minutos)...\n")
    qsvm = QuantumKernelSVM(
        n_features=args.n_qubits,
        C=args.C,
        reps=args.reps,
    )
    qsvm.fit(X_tr_q, y_tr_q)
    met_qsvm = evaluate_classifier(
        qsvm, X_te_sc, y_te, name=f"Quantum Kernel SVM (n={n_q})"
    )

    # ── 11. Resumen comparativo ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RESUMEN COMPARATIVO")
    print("=" * 55)
    models = [met_rbf_full, met_rbf_sub, met_qsvm]
    for m in models:
        print(f"  {m['name']:42s}  Acc={m['accuracy']:.4f}  AUC={m['auc']:.4f}")

    d_fair = met_qsvm["auc"] - met_rbf_sub["auc"]
    d_full = met_qsvm["auc"] - met_rbf_full["auc"]
    print(f"\n  ΔAUC (QSVM − RBF_sub)  = {d_fair:+.4f}  ← comparación justa (mismo n)")
    print(f"  ΔAUC (QSVM − RBF_full) = {d_full:+.4f}  ← contexto vs. clásico sin restricción")

    if d_fair > 0.01:
        print("\n  → Quantum kernel supera al clásico en igual condición de datos.")
    elif d_fair < -0.01:
        print("\n  → RBF clásico supera al quantum kernel (resultado honesto y publicable).")
    else:
        print("\n  → Rendimiento comparable (quantum parity).")

    # ── 12. Figura ────────────────────────────────────────────────────────────
    plot_confusion_matrix(models, y_te, save_path=args.save_fig)

    # ── 13. Guardar CSV de resultados ─────────────────────────────────────────
    results = {
        "n_total"         : len(y),
        "n_fd"            : int(y.sum()),
        "n_nofd"          : int((y == 0).sum()),
        "n_train_full"    : len(y_tr),
        "n_train_quantum" : n_q,
        "n_test"          : len(y_te),
        "n_qubits"        : args.n_qubits,
        "reps"            : args.reps,
        "C"               : args.C,
        "min_magn"        : args.min_magn,
        "auc_rbf_full"    : met_rbf_full["auc"],
        "acc_rbf_full"    : met_rbf_full["accuracy"],
        "auc_rbf_sub"     : met_rbf_sub["auc"],
        "acc_rbf_sub"     : met_rbf_sub["accuracy"],
        "auc_qsvm"        : met_qsvm["auc"],
        "acc_qsvm"        : met_qsvm["accuracy"],
        "delta_auc_fair"  : d_fair,
        "delta_auc_full"  : d_full,
    }
    pd.DataFrame([results]).to_csv(args.save_csv, index=False)
    print(f"\n[main] Resultados guardados en {args.save_csv}")


if __name__ == "__main__":
    main()
