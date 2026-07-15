#!/bin/bash
#SBATCH --job-name=distancias_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00

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

echo "==========================================="
echo "Compilando..."
echo "==========================================="

nvcc -O3 -arch=sm_75 versionCublas.cu     -o versionCublas     -lcublas
nvcc -O3 -arch=sm_75 versionCuda.cu       -o versionCuda       -lcublas
nvcc -O3 -arch=sm_75 versionBitPacking.cu -o versionBitPacking -lcublas
nvcc -O3 -arch=sm_75 versionTensor.cu     -o versionTensor     -lcublas
nvcc -O3 -arch=sm_75 versionBitTensor.cu  -o versionBitTensor  -lcublas

echo ""
echo "Compilacion finalizada."
echo ""

####################################################
# Tamanos de prueba
####################################################
#
# GPU objetivo: 12 GB de memoria global.
#
# Cada version se ejecuta como un proceso independiente, por lo que la memoria
# del device se libera entre corridas: el limite lo fija la version mas pesada
# en modo verify. Las matrices n x n en float dominan el consumo. En verify las
# versiones mantienen 5 buffers n x n (d_C/d_W, d_D2, d_D, d_G, d_Dref) mas la
# entrada, por lo que el pico aproximado es:
#
#     pico ~= 20 * n^2 + 7 * n * m   bytes
#
# Se apunta a un pico <= ~9 GB para dejar margen al contexto CUDA y al
# workspace de cuBLAS en una placa de 12 GB. Todos los tamanos son multiplos
# de 32 (la letra asume que X es divisible en bloques de 32x32).
#
# Los pares (N, M) barren tres ejes:
#   - Barrido de N (individuos) con M fijo: escalado del producto X*X^T y de D.
#   - Barrido de M (SNPs) con N fijo: escalado de la dimension de reduccion.
#   - Casos grandes combinados: estres cercano al limite de 12 GB.
#
# Comentario a la derecha: pico de memoria GPU estimado.

N_VALUES=(2048 4096 8192 16384   4096  4096  4096   4096    8192  12288 16384 16384) # cantidad de individuos
M_VALUES=(8192 8192 8192 8192    4096  16384 65536  131072  16384 16384 16384 32768) # cantidad de SNPs por individuo
# pico GPU (GB):  0.2  0.6  1.8  6.3    0.5   0.8   2.2    4.1     2.3   4.4   7.3   9.1

####################################################
# Comparacion versionCublas (referencia) vs versiones propias
#   Time(ms) = pipeline completo (H->D + kernels + D->H)
#   Mult(ms) = solo el kernel de multiplicacion
#   Speedup  = TimeCublas / TimeVersion
#   ErrRel   = ||D - D_cuBLAS|| / ||D_cuBLAS||
####################################################

echo "==================================================================="
echo " Comparacion: versionCublas (referencia) vs versiones propias"
echo " Time = pipeline completo | desglose por paso del for de medicion:"
echo "   HtoD = copia H->D | Norm = normas/popcounts | Mult = kernel mult"
echo "   Dist = distance_squared | Sqrt = sqrt | DtoH = copia D->H"
echo "==================================================================="
echo ""

# Versiones propias (se corren en modo verify). versionCublas es la referencia.
VERSIONS=(versionCuda versionBitPacking versionTensor versionBitTensor)

# Extrae del stdout de una version las metricas por paso (una linea "clave valor").
# Imprime: TIME HTOD NORMAS MULT DIST SQRT DTOH ERRREL (ERRREL = "-" si no hay).
parse_steps() {
    awk '/^TIME_MS/{t=$2} /^HTOD_MS/{h=$2} /^NORMAS_MS/{no=$2} /^MULT_MS/{mu=$2} \
         /^DIST_MS/{d=$2} /^SQRT_MS/{s=$2} /^DTOH_MS/{dh=$2} /^ERR_REL/{e=$2} \
         END{ if (e=="") e="-"; print t, h, no, mu, d, s, dh, e }'
}

HDR="%-6s %-7s %-17s %-9s %-8s %-8s %-9s %-8s %-8s %-8s %-9s %-12s\n"
printf "$HDR" "N" "M" "Version" "Time(ms)" "HtoD" "Norm" "Mult(ms)" "Dist" "Sqrt" "DtoH" "Speedup" "ErrRel"
printf "$HDR" "------" "-------" "-----------------" "---------" "--------" "--------" "---------" "--------" "--------" "--------" "---------" "------------"

for idx in "${!N_VALUES[@]}"
do
    N=${N_VALUES[$idx]}
    M=${M_VALUES[$idx]}

    # Referencia cuBLAS: tiempo total y desglose por paso, sin error
    read -r base_ms base_htod base_norm base_mult base_dist base_sqrt base_dtoh base_err \
        < <(./versionCublas "$N" "$M" | parse_steps)
    printf "$HDR" "$N" "$M" "Cublas(ref)" \
        "$base_ms" "$base_htod" "$base_norm" "$base_mult" "$base_dist" "$base_sqrt" "$base_dtoh" "1.00x" "-"

    # Versiones propias en modo verify: total + desglose + ERR_REL
    for ver in "${VERSIONS[@]}"
    do
        read -r ms htod norm mult dist sqrt dtoh err \
            < <(./"$ver" "$N" "$M" verify | parse_steps)
        sp=$(awk -v b="$base_ms" -v v="$ms" 'BEGIN{ if (v+0>0) printf "%.2fx", b/v; else printf "NA" }')
        printf "$HDR" "$N" "$M" "$ver" "$ms" "$htod" "$norm" "$mult" "$dist" "$sqrt" "$dtoh" "$sp" "$err"
    done

    echo ""
done

echo ""
echo "==========================================="
echo "FIN"
echo "==========================================="
