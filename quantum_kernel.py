"""
quantum_kernel.py
=================
Quantum Kernel SVM para clasificación de Forbush Decreases.

Usa:
  - ZZFeatureMap  (entanglement = 'linear', reps = 2)  → feature map φ(x)
  - FidelityQuantumKernel                              → K(x,x') = |⟨φ(x)|φ(x')⟩|²
  - SVC (kernel="precomputed")                         → clasificador final

K_test se calcula en BATCHES para evitar OOM (el test set completo
mataría la memoria con >500 muestras de test × 150 de train).
"""

import numpy as np
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import Sampler          # V1 — requerido por ComputeUncompute


def build_quantum_kernel(
    n_features: int,
    reps: int = 2,
    entanglement: str = "linear",
) -> tuple:
    feature_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=reps,
        entanglement=entanglement,
    )
    print(f"[quantum_kernel] ZZFeatureMap: {n_features} qubits, "
          f"reps={reps}, entanglement='{entanglement}'")
    print(f"[quantum_kernel] Profundidad del circuito: {feature_map.decompose().depth()}")

    sampler  = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    return feature_map, quantum_kernel


def compute_kernel_matrix(
    quantum_kernel: FidelityQuantumKernel,
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
    batch_size: int = 50,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Calcula K_train y K_test por batches para evitar OOM.

    K_train : n_train × n_train  (simétrica)
    K_test  : n_test  × n_train  (calculada fila a fila en batches)

    batch_size: filas de X_test por llamada. Reducir a 25 si sigue muriendo.
    """
    print(f"[quantum_kernel] Calculando K_train ({len(X_train)}×{len(X_train)})...")
    K_train = quantum_kernel.evaluate(x_vec=X_train)

    K_test = None
    if X_test is not None:
        n_test    = len(X_test)
        n_train   = len(X_train)
        K_test    = np.zeros((n_test, n_train), dtype=np.float64)
        n_batches = int(np.ceil(n_test / batch_size))

        print(f"[quantum_kernel] Calculando K_test ({n_test}×{n_train}) "
              f"en {n_batches} batches de {batch_size}...")

        for i in range(n_batches):
            lo = i * batch_size
            hi = min(lo + batch_size, n_test)
            print(f"  batch {i+1}/{n_batches}  filas [{lo}:{hi}]", end="\r")
            K_test[lo:hi, :] = quantum_kernel.evaluate(
                x_vec=X_test[lo:hi], y_vec=X_train
            )
        print()

    return K_train, K_test


def train_quantum_svm(K_train, y_train, C=1.0):
    clf = SVC(kernel="precomputed", C=C, probability=True, random_state=42)
    clf.fit(K_train, y_train)
    print(f"[quantum_kernel] Quantum SVM entrenado  (C={C})")
    return clf


class QuantumKernelSVM:
    """
    Wrapper: feature_map → kernel matrix (batched) → SVC.
    Expone predict() y predict_proba() compatibles con evaluate_classifier().

    Parámetros
    ----------
    n_features  : número de qubits
    C           : regularización SVM
    reps        : profundidad ZZFeatureMap
    entanglement: 'linear' (default) o 'full'
    batch_size  : filas de test evaluadas por llamada al kernel cuántico
                  (bajar a 25 si el proceso sigue siendo killed)
    """

    def __init__(
        self,
        n_features: int,
        C: float = 1.0,
        reps: int = 2,
        entanglement: str = "linear",
        batch_size: int = 50,
    ):
        self.C          = C
        self.batch_size = batch_size
        self.feature_map, self.q_kernel = build_quantum_kernel(
            n_features, reps, entanglement
        )
        self.clf      = None
        self.X_train_ = None

    def fit(self, X_train, y_train):
        self.X_train_ = X_train
        K_train, _ = compute_kernel_matrix(
            self.q_kernel, X_train, batch_size=self.batch_size
        )
        self.clf = train_quantum_svm(K_train, y_train, C=self.C)
        return self

    def predict(self, X_test):
        _, K_test = compute_kernel_matrix(
            self.q_kernel, self.X_train_, X_test, batch_size=self.batch_size
        )
        return self.clf.predict(K_test)

    def predict_proba(self, X_test):
        _, K_test = compute_kernel_matrix(
            self.q_kernel, self.X_train_, X_test, batch_size=self.batch_size
        )
        return self.clf.predict_proba(K_test)
