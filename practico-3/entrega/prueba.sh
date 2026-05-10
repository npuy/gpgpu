#!/bin/bash
#SBATCH --job-name=pruebas_pr3
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00

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

OUTPUT_DIR="./outputs"
MATRIX_N="16384"   # 2^14
ARRAY_CASES=("4194304" "268435456")   # 2^22, 2^28
ARRAY_LABELS=("2^22" "2^28")

EJ11GLOBAL_OUTPUT="$OUTPUT_DIR/output11global.txt"
EJ11SHARED_OUTPUT="$OUTPUT_DIR/output11shared.txt"
EJ12PADDING_OUTPUT="$OUTPUT_DIR/output12padding.txt"

echo "Compilando los 7 programas de la entrega..."
echo ""
nvcc ej11global.cu -o ej11global
nvcc ej11shared.cu -o ej11shared
nvcc ej12padding.cu -o ej12padding
nvcc ej21mem.cu -o ej21mem
nvcc ej21sec.cu -o ej21sec
nvcc ej22prim.cu -o ej22prim
nvcc ej22prueba.cu -o ej22prueba

echo "Compilacion completada para los 7 programas."
echo ""

mkdir -p "$OUTPUT_DIR"

echo "==== PRUEBAS EJERCICIO 1 ===="
echo ""

echo "Iniciando pruebas con matrices N = 2^14 ($MATRIX_N)."
echo ""

echo "Ejecutando ej11global: traspuesta con accesos no coalesced en memoria global."
nsys profile -t cuda -o reporte1 ./ej11global "$MATRIX_N" "$EJ11GLOBAL_OUTPUT" > /dev/null 2>&1
nsys stats --report cuda_gpu_kern_sum reporte1.nsys-rep
echo "Archivo generado: $EJ11GLOBAL_OUTPUT"
echo ""

echo "Ejecutando ej11shared: traspuesta usando memoria compartida."
nsys profile -t cuda -o reporte2 ./ej11shared "$MATRIX_N" "$EJ11SHARED_OUTPUT" > /dev/null 2>&1
nsys stats --report cuda_gpu_kern_sum reporte2.nsys-rep
echo "Archivo generado: $EJ11SHARED_OUTPUT"
echo ""

echo "Ejecutando ej12padding: traspuesta con memoria compartida y padding."
nsys profile -t cuda -o reporte3 ./ej12padding "$MATRIX_N" "$EJ12PADDING_OUTPUT" > /dev/null 2>&1
nsys stats --report cuda_gpu_kern_sum reporte3.nsys-rep
echo "Archivo generado: $EJ12PADDING_OUTPUT"

echo "==== PRUEBAS EJERCICIO 2 ===="
echo ""

echo "Iniciando pruebas con arrat n = 2^28 ($ARRAY_CASES) para medir tiempos de ejecución."


for i in "${!ARRAY_CASES[@]}"; do
    n="${ARRAY_CASES[$i]}"
    label="${ARRAY_LABELS[$i]}"

    echo "Ejecutando ej21sec con n = $label ($n): version secuencial por bloque."
    nsys profile -t cuda -o reporte21sec_${label} ./ej21sec "$n" "$OUTPUT_DIR/output21sec_${label}.txt" > /dev/null 2>&1
    nsys stats --report cuda_gpu_kern_sum reporte21sec_${label}.nsys-rep
    echo "Archivo generado: $OUTPUT_DIR/output21sec_${label}.txt"
    echo ""

    echo "Ejecutando ej21mem con n = $label ($n): reduccion en memoria compartida."
    nsys profile -t cuda -o reporte21mem_${label} ./ej21mem "$n" "$OUTPUT_DIR/output21mem_${label}.txt" > /dev/null 2>&1
    nsys stats --report cuda_gpu_kern_sum reporte21mem_${label}.nsys-rep
    echo "Archivo generado: $OUTPUT_DIR/output21mem_${label}.txt"
    echo ""

    echo "Ejecutando ej22prim con n = $label ($n): version con primitivas warp-level."
    nsys profile -t cuda -o reporte22prim_${label} ./ej22prim "$n" "$OUTPUT_DIR/output22prim_${label}.txt" > /dev/null 2>&1
    nsys stats --report cuda_gpu_kern_sum reporte22prim_${label}.nsys-rep
    echo "Archivo generado: $OUTPUT_DIR/output22prim_${label}.txt"
    echo ""

done

echo "Ejecutando ej22prueba: caso borde fijo con el array [1, -2, 3, ..., 32]."
./ej22prueba

echo "==== FIN ===="
echo "Todas las ejecuciones finalizaron."
echo "Los outputs quedaron almacenados en: $OUTPUT_DIR"
echo ""