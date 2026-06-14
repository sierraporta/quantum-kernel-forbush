"""
classical_baseline.py
=====================
Baseline: SVM con kernel RBF clásico.
Sirve como referencia para comparar con el quantum kernel.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


def train_classical_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    gamma: str | float = "scale",
) -> SVC:
    """Entrena SVM-RBF clásico."""
    clf = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    print(f"[classical_baseline] SVM-RBF entrenado  (C={C}, gamma={gamma})")
    return clf


def evaluate_classifier(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str = "Classifier",
) -> dict:
    """
    Evalúa un clasificador (clásico o cuántico) y retorna métricas.
    """
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    print(f"\n{'='*50}")
    print(f"  {name}")
    print('='*50)
    print(classification_report(y_test, y_pred, target_names=["No-FD", "FD"]))

    metrics = {
        "name"    : name,
        "accuracy": np.mean(y_pred == y_test),
        "auc"     : roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "y_pred"  : y_pred,
        "y_prob"  : y_prob,
    }
    if metrics["auc"]:
        print(f"  ROC-AUC: {metrics['auc']:.4f}")
    return metrics


def cross_validate_svm(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    n_splits: int = 5,
) -> dict:
    """
    Validación cruzada estratificada para el SVM clásico.
    """
    clf = SVC(kernel="rbf", C=C, probability=True, random_state=42)
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        clf, X, y, cv=cv,
        scoring=["accuracy", "roc_auc", "f1"],
        return_train_score=False,
    )
    print(f"\n[classical_baseline] CV ({n_splits}-fold):")
    for metric in ["test_accuracy", "test_roc_auc", "test_f1"]:
        vals = scores[metric]
        print(f"  {metric:20s}: {vals.mean():.4f} ± {vals.std():.4f}")
    return scores


def plot_confusion_matrix(
    metrics_list: list[dict],
    y_test: np.ndarray,
    save_path: str | None = None,
):
    """Grafica matrices de confusión lado a lado para comparar modelos."""
    n = len(metrics_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics_list):
        cm = confusion_matrix(y_test, m["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["No-FD", "FD"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"{m['name']}\nAUC={m['auc']:.3f}" if m["auc"] else m["name"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[classical_baseline] Figura guardada en {save_path}")
    plt.show()
