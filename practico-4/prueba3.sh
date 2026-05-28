#!/bin/bash
#SBATCH --job-name=pruebas_pr4_ej3
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00

#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:n2080ti:1

#SBATCH --partition=cursos
#SBATCH --qos=gpgpu

PATH=$PATH:/usr/local/cuda/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi
echo "[INFO] Directorio de trabajo: $(pwd)"

PROGRAM="ej3_bins_thrust"
SOURCE="ej3_bins_thrust.cu"
echo "Compilando $SOURCE ..."
nvcc -O3 -std=c++14 "$SOURCE" -o "$PROGRAM"
echo "Compilacion completada."
echo ""

TEST_N=1000
echo "Ejecucion de prueba simple con N = $TEST_N"
./"$PROGRAM" "$TEST_N" --csv
echo ""

echo "==== EJECUCIONES EJERCICIO 3 ===="
echo "Se ejecutaran los casos N = 1024 * 2^k para k = 1..10 sin generar CSV"
echo ""

for k in {1..10}; do
    n=$((1024 * (2 ** k)))

    echo "Ejecutando para k = $k, N = $n"
    ./"$PROGRAM" "$n"
    echo ""
done

echo "==== FIN ===="
echo "El CSV del caso simple queda en: resultado_ej3_thrust.csv"
