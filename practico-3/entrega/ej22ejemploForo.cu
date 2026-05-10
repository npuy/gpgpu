#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define CUDA_CHK(ans) {gpuAssert((ans), __FILE__, __LINE__)}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

void parse_input(int argc, char *argv[], const char **output_path);
int *generate_array(int n);
void write_file(const char *path, int *output, int length);

__global__ void shuffle_separado(int *array, int n)
{
    int tid = threadIdx.x;
    int id_global = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (id_global < n)
    {
        int valor = array[id_global];

        unsigned mask_pos = __ballot_sync(0xFFFFFFFF, valor > 0);
        unsigned mask_neg = __ballot_sync(0xFFFFFFFF, valor < 0);

        int suma_neg = 0;
        int max_pos = 0;

        // Reduccion para negativos - solo participan los negativos
        if (valor < 0)
        {
            suma_neg = valor;
            for (int stride = 16; stride > 0; stride /= 2)
            {
                int src_lane = tid + stride;
                int temp = __shfl_down_sync(mask_neg, suma_neg, stride);
                // Solo sumar si leimos de un lane valido
                if ((mask_neg >> src_lane) & 1)
                    suma_neg += temp;
            }
            // Broadcast del resultado a todos los negativos
            suma_neg = __shfl_sync(mask_neg, suma_neg, __ffs(mask_neg) - 1);
        }

        // Reduccion para positivos - solo participan los positivos
        if (valor > 0)
        {
            max_pos = valor;
            for (int stride = 16; stride > 0; stride /= 2)
            {
                int src_lane = tid + stride;
                int temp = __shfl_down_sync(mask_pos, max_pos, stride);
                if ((mask_pos >> src_lane) & 1)
                    max_pos = max(max_pos, temp);
            }
            max_pos = __shfl_sync(mask_pos, max_pos, __ffs(mask_pos) - 1);
        }

        // Compartir resultados entre todos
        int suma_broadcast = __shfl_sync(0xFFFFFFFF, suma_neg, __ffs(mask_neg) - 1);
        int max_broadcast = __shfl_sync(0xFFFFFFFF, max_pos, __ffs(mask_pos) - 1);

        if (valor < 0)
        {
            array[id_global] = suma_broadcast + max_broadcast;
        }
    }
}

int main(int argc, char *argv[])
{
    const int n = 32;
    const int size = n * sizeof(int);

    int *h_array = generate_array(n);
    int *d_array;

    printf("Entrada:\n");
    for (int i = 0; i < n; i++)
    {
        printf("%d ", h_array[i]);
    }
    printf("\n\n");

    cudaMalloc(&d_array, size);
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    shuffle_separado<<<1, 32>>>(d_array, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de ejecucion del caso borde: %f ms\n", milliseconds);

    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    printf("Salida:\n");
    for (int i = 0; i < n; i++)
    {
        printf("%d ", h_array[i]);
    }
    printf("\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_array);
    free(h_array);

    return 0;
}

int *generate_array(int n)
{
    int *array = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++)
    {
        array[i] = i + 1;
    }
    array[1] = -2;

    return array;
}
