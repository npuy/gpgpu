#!/bin/bash
#SBATCH --job-name=distancias_gpu
#SBATCH --ntasks=1
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

cleanup_nsys() {
    rm -f "$1.nsys-rep" "$1.sqlite"
}

echo "==========================================="
echo "Compilando..."
echo "==========================================="

nvcc -O3 -arch=sm_75 version1.cu -o version1
nvcc -O3 -arch=sm_75 version2.cu -o version2
nvcc -O3 -arch=sm_75 version22.cu -o version22
nvcc -O3 -arch=sm_75 version22gpu.cu -o version22gpu
nvcc -O3 -arch=sm_75 version3.cu -o version3


echo ""
echo "Compilacion finalizada."
echo ""

####################################################
# Tamanos de prueba
####################################################

N_VALUES=(1024 1024 2048 2048 4096 1024) # cantidad de individuos

M_VALUES=(1024 4096 2048 4096 4096 32768) # cantidad de SNPs por individuo

####################################################
# VERSION 1
####################################################

echo "==========================================="
echo "VERSION 1"
echo "==========================================="

for idx in "${!N_VALUES[@]}"
do

    N=${N_VALUES[$idx]}
    M=${M_VALUES[$idx]}

    echo ""
    echo "-------------------------------------------"
    echo "N=$N   M=$M"
    echo "-------------------------------------------"

    ./version1 $N $M

    echo ""
    echo "Nsight Systems"

    nsys profile \
        -t cuda \
        -o reporte_v1_${N}_${M} \
        ./version1 $N $M \
        > /dev/null

    nsys stats \
        --report cuda_gpu_kern_sum \
        reporte_v1_${N}_${M}.nsys-rep \
        2>&1 | sed \
        '/Generating SQLite file/d;
         /Processing \[/d;
         /\*\* CUDA GPU Kernel Summary/d'

    cleanup_nsys reporte_v1_${N}_${M}

done

####################################################
# VERSION 2
####################################################

echo ""
echo "==========================================="
echo "VERSION 2.1 (BIT PACKING 8 SNPs)"
echo "==========================================="

for idx in "${!N_VALUES[@]}"
do

    N=${N_VALUES[$idx]}
    M=${M_VALUES[$idx]}

    echo ""
    echo "-------------------------------------------"
    echo "N=$N   M=$M"
    echo "-------------------------------------------"

    ./version2 $N $M

    echo ""
    echo "Nsight Systems"

    nsys profile \
        -t cuda \
        -o reporte_v2_${N}_${M} \
        ./version2 $N $M \
        > /dev/null

    nsys stats \
        --report cuda_gpu_kern_sum \
        reporte_v2_${N}_${M}.nsys-rep \
        2>&1 | sed \
        '/Generating SQLite file/d;
         /Processing \[/d;
         /\*\* CUDA GPU Kernel Summary/d'

    cleanup_nsys reporte_v2_${N}_${M}

done

echo ""
echo "==========================================="
echo "VERSION 2.2 (BIT PACKING 16 SNPs)"
echo "==========================================="

for idx in "${!N_VALUES[@]}"
do

    N=${N_VALUES[$idx]}
    M=${M_VALUES[$idx]}

    echo ""
    echo "-------------------------------------------"
    echo "N=$N   M=$M"
    echo "-------------------------------------------"

    ./version22 $N $M

    echo ""
    echo "Nsight Systems"

    nsys profile \
        -t cuda \
        -o reporte_v2_${N}_${M} \
        ./version2 $N $M \
        > /dev/null

    nsys stats \
        --report cuda_gpu_kern_sum \
        reporte_v2_${N}_${M}.nsys-rep \
        2>&1 | sed \
        '/Generating SQLite file/d;
         /Processing \[/d;
         /\*\* CUDA GPU Kernel Summary/d'

    cleanup_nsys reporte_v2_${N}_${M}

done

echo ""
echo "==========================================="
echo "VERSION 2.2 GPU (BIT PACKING 16 SNPs EN GPU)"
echo "==========================================="

for idx in "${!N_VALUES[@]}"
do

    N=${N_VALUES[$idx]}
    M=${M_VALUES[$idx]}

    echo ""
    echo "-------------------------------------------"
    echo "N=$N   M=$M"
    echo "-------------------------------------------"

    ./version22gpu $N $M

    echo ""
    echo "Nsight Systems"

    nsys profile \
        -t cuda \
        -o reporte_v2gpu_${N}_${M} \
        ./version2 $N $M \
        > /dev/null

    nsys stats \
        --report cuda_gpu_kern_sum \
        reporte_v2gpu_${N}_${M}.nsys-rep \
        2>&1 | sed \
        '/Generating SQLite file/d;
         /Processing \[/d;
         /\*\* CUDA GPU Kernel Summary/d'

    cleanup_nsys reporte_v2gpu_${N}_${M}

done

echo ""
echo "==========================================="
echo "VERSION 3 (TENSOR CORES WMMA)"
echo "==========================================="

for idx in "${!N_VALUES[@]}"
do

    N=${N_VALUES[$idx]}
    M=${M_VALUES[$idx]}

    echo ""
    echo "-------------------------------------------"
    echo "N=$N   M=$M"
    echo "-------------------------------------------"

    ./version3 $N $M

    echo ""
    echo "Nsight Systems"

    nsys profile \
        -t cuda \
        -o reporte_v3_${N}_${M} \
        ./version3 $N $M \
        > /dev/null

    nsys stats \
        --report cuda_gpu_kern_sum \
        reporte_v3_${N}_${M}.nsys-rep \
        2>&1 | sed \
        '/Generating SQLite file/d;
         /Processing \[/d;
         /\*\* CUDA GPU Kernel Summary/d'

    cleanup_nsys reporte_v3_${N}_${M}

done

echo ""
echo "==========================================="
echo "FIN"
echo "==========================================="