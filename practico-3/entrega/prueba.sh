#!/bin/bash
#SBATCH --job-name=primera_prueba
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00

#SBATCH --gres=gpu:1
# para ejecutar en la gtx1060 ---> #SBATCH --gres=gpu:n1060:1
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

OUTPUT_DIR="./outputs"
MATRIX_N="16384"   # 2^14
ARRAY_CASES=("4096" "4194304" "268435456")   # 2^12, 2^22, 2^28
ARRAY_LABELS=("2^12" "2^22" "2^28")

EJ11GLOBAL_OUTPUT="$OUTPUT_DIR/output11global.txt"
EJ11SHARED_OUTPUT="$OUTPUT_DIR/output11shared.txt"
EJ12PADDING_OUTPUT="$OUTPUT_DIR/output12padding.txt"

echo "[INFO] Compilando los 7 programas de la entrega..."
nvcc ej11global.cu -o ej11global
nvcc ej11shared.cu -o ej11shared
nvcc ej12padding.cu -o ej12padding
nvcc ej21mem.cu -o ej21mem
nvcc ej21sec.cu -o ej21sec
nvcc ej22prim.cu -o ej22prim
nvcc ej22ejemploForo.cu -o ej22ejemploForo

echo "[INFO] Compilacion completada para los 7 programas."

mkdir -p "$OUTPUT_DIR"
echo "[INFO] Directorio de salida listo: $OUTPUT_DIR"

echo "[INFO] Iniciando pruebas de matrices con n = 2^14 ($MATRIX_N)."
echo "[INFO] Ejecutando ej11global: traspuesta con accesos no coalesced en memoria global."
./ej11global "$MATRIX_N" "$EJ11GLOBAL_OUTPUT"
echo "[INFO] Archivo generado: $EJ11GLOBAL_OUTPUT"

echo "[INFO] Ejecutando ej11shared: traspuesta usando memoria compartida."
./ej11shared "$MATRIX_N" "$EJ11SHARED_OUTPUT"
echo "[INFO] Archivo generado: $EJ11SHARED_OUTPUT"

echo "[INFO] Ejecutando ej12padding: traspuesta con memoria compartida y padding."
./ej12padding "$MATRIX_N" "$EJ12PADDING_OUTPUT"
echo "[INFO] Archivo generado: $EJ12PADDING_OUTPUT"

echo "[INFO] Iniciando pruebas de arrays para cada programa con n = 2^12, 2^22 y 2^28."

for i in "${!ARRAY_CASES[@]}"; do
    n="${ARRAY_CASES[$i]}"
    label="${ARRAY_LABELS[$i]}"

    echo "[INFO] Ejecutando ej21sec con n = $label ($n): version secuencial por bloque."
    ./ej21sec "$n" "$OUTPUT_DIR/output21sec_${label}.txt"
    echo "[INFO] Archivo generado: $OUTPUT_DIR/output21sec_${label}.txt"

    echo "[INFO] Ejecutando ej21mem con n = $label ($n): reduccion en memoria compartida."
    ./ej21mem "$n" "$OUTPUT_DIR/output21mem_${label}.txt"
    echo "[INFO] Archivo generado: $OUTPUT_DIR/output21mem_${label}.txt"

    echo "[INFO] Ejecutando ej22prim con n = $label ($n): version con primitivas warp-level."
    ./ej22prim "$n" "$OUTPUT_DIR/output22prim_${label}.txt"
    echo "[INFO] Archivo generado: $OUTPUT_DIR/output22prim_${label}.txt"
done

echo "[INFO] Ejecutando ej22ejemploForo: caso borde fijo con el array [1, -2, 3, ..., 32]."
./ej22ejemploForo

echo "[INFO] Todas las ejecuciones finalizaron correctamente."
echo "[INFO] Los outputs quedaron almacenados en: $OUTPUT_DIR"
