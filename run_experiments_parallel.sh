#!/bin/bash
# run_experiments_parallel.sh
# Uso: bash run_experiments_parallel.sh [n_jobs]

export OMNI="omni_clean.pkl"
export JUNG="jung_clean.pkl"
export FEID="feid_clean.pkl"
export RESULTS_DIR="Results"
N_JOBS=${1:-4}

mkdir -p "$RESULTS_DIR"

MIN_MAGNS=(0.0 3.0 4.0 5.0 6.0 7.0)
N_QUBITS=(4 6 8)
REPS=(1 2 3)
MAX_QS=(50 100 150 200 250)

run_one() {
    local magn=$1 qubits=$2 reps=$3 maxq=$4
    local TAG="${magn}_${qubits}_${reps}_${maxq}"
    local FIG="${RESULTS_DIR}/results_${TAG}.png"
    local CSV="${RESULTS_DIR}/exp_${TAG}.csv"

    if [ -f "$CSV" ]; then
        echo "[SKIP] $TAG — ya existe"
        return 0
    fi

    echo "[START] min_magn=$magn  n_qubits=$qubits  reps=$reps  max_q=$maxq"

    python3 main_v2.py \
        --omni     "$OMNI"   \
        --jung     "$JUNG"   \
        --feid     "$FEID"   \
        --min_magn "$magn"   \
        --n_qubits "$qubits" \
        --reps     "$reps"   \
        --max_q    "$maxq"   \
        --save_fig "$FIG"    \
        --save_csv "$CSV"

    if [ $? -eq 0 ]; then
        echo "[DONE] $TAG"
    else
        echo "[ERROR] $TAG"
    fi
}

export -f run_one

echo "============================================"
echo "  Quantum FD Classifier — Parallel Batch"
echo "  Jobs en paralelo: $N_JOBS"
echo "============================================"

{
for magn in "${MIN_MAGNS[@]}"; do
for qubits in "${N_QUBITS[@]}"; do
for reps in "${REPS[@]}"; do
for maxq in "${MAX_QS[@]}"; do
    echo "$magn $qubits $reps $maxq"
done; done; done; done
} | parallel --jobs "$N_JOBS" \
             --colsep ' ' \
             --progress \
             run_one {1} {2} {3} {4}

echo ""
echo "============================================"
echo "  Batch completado. Resultados en: $RESULTS_DIR/"
echo "============================================"
