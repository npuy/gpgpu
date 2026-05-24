#!/bin/bash
#SBATCH --job-name=pruebas_pr4
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

PROGRAM="ej1_scan"
SOURCE="ej1_scan.cu"
OUTPUT_DIR="./outputs"

mkdir -p "$OUTPUT_DIR"

echo "Compilando $SOURCE ..."
nvcc -O3 -std=c++14 "$SOURCE" -o "$PROGRAM"
echo "Compilacion completada."
echo ""

TEST_N=2048
echo "Ejecucion de prueba simple con N = $TEST_N"
./"$PROGRAM" "$TEST_N"
echo ""

echo "==== PROFILING EJERCICIO 1 ===="
echo "Se ejecutaran los casos N = 1024 * 2^k para k = 1..10"
echo ""

for k in {1..10}; do
    n=$((1024 * (2 ** k)))
    label="k${k}_N${n}"

    echo "Ejecutando profiling para k = $k, N = $n"
    nsys profile -t cuda -o "$OUTPUT_DIR/reporte_${label}" ./"$PROGRAM" "$n" > /dev/null 2>&1
    nsys stats --report cuda_gpu_kern_sum "$OUTPUT_DIR/reporte_${label}.nsys-rep"
    echo ""
done

echo "==== FIN ===="
echo "Los reportes quedaron en: $OUTPUT_DIR"
