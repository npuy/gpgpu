#!/bin/bash
#SBATCH --job-name=primera_prueba
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00

#SBATCH --gres=gpu:1
# para ejecutar en la gtx1060 ---> #SBATCH --gres=gpu:n1060:1
# para ejecutar en la rtx2080ti ---> #SBATCH --gres=gpu:n2080ti:1

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
echo "Current directory: $(pwd)"

TEST_DIR="../test"

# Default input files in practico-2/test/
EJ1_INPUT="secreto.txt"
EJ2_INPUT="matrix2.txt"
EJ3_INPUT="matrix3.txt"

# Output files produced by each program in the test directory.
EJ1_OUTPUT="output1.txt"
EJ2_OUTPUT="output2.txt"
EJ3_OUTPUT="output3.txt"

function ensure_file_exists() {
    local file_path="$1"
    local description="$2"
    if [[ ! -f "$file_path" ]]; then
        echo "Error: $description not found: $file_path"
        exit 1
    fi
}

# Compile all exercises
nvcc ej1.cu -o ej1
nvcc ej2.cu -o ej2
nvcc ej3.cu -o ej3

echo "Compiled ej1, ej2, ej3"

# Verify input and expected files
ensure_file_exists "$TEST_DIR/$EJ1_INPUT" "Input file for ej1"
ensure_file_exists "$TEST_DIR/$EJ2_INPUT" "Input file for ej2"
ensure_file_exists "$TEST_DIR/$EJ3_INPUT" "Input file for ej3"

pushd "$TEST_DIR" >/dev/null

echo "Running ej1 with input $EJ1_INPUT..."
../src/ej1 "$EJ1_INPUT"
echo "Output for ej1 written to $EJ1_OUTPUT"

echo "Running ej2 with input $EJ2_INPUT..."
../src/ej2 "$EJ2_INPUT"
echo "Output for ej2 written to $EJ2_OUTPUT"

echo "Running ej3 with input $EJ3_INPUT..."
../src/ej3 "$EJ3_INPUT" "$EJ3_OUTPUT"
echo "Output for ej3 written to $EJ3_OUTPUT"

popd >/dev/null