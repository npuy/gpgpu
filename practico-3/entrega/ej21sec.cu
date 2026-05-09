#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <cmath>
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

/*
En este codigo hacemos la reduccion usando la mask 0xFFFFFFFF. Es la primer version que se nos ocurrio.

Si bien funciona, usar la mask 0xFFFFFFFF no parece lo mejor. Estamos pidiendo que todos los hilos participen por ejemplo en la suma de negativos,
incluso aquellos hilos que no tienen negativos. Lo mismo con el maximo de positivos.

Esto hace que todos los hilos participen de ambas reducciones, incluso los que no aportan a una de las dos.  Es como si se "serializaran" las reducciones en vez de paralelizarlas.
En los otros codigos para la parte 2.2 probamos formas de mejorar esto.

Si bien la mask indica que hilos van a participar, si usamos por ejemplo la mask_neg para la reduccion de negativos,
de todos modos leemos valores de hilos no indicados en la mask al hacer la reduccion dado el offset, y eso puede dar valores indefinidos.
Es por eso que mantenemos esta version con mask 0xFFFFFFFF como una version inicial de todos modos.
*/

__global__ void funcion_secuencial(int *in, int *out, int n)
{
    __shared__ int tile[32];
    __shared__ int sum_neg;
    __shared__ int max_pos;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n)
        tile[tid] = in[idx];

    __syncthreads();

    if (tid == 0)
    {
        sum_neg = 0;
        max_pos = 0;

        for (int i = 0; i < 32; i++)
        {
            if (tile[i] < 0)
                sum_neg += tile[i];
            else if (tile[i] > max_pos)
                max_pos = tile[i];
        }
    }

    __syncthreads();

    if (idx < n)
    {
        if (tile[tid] < 0)
            out[idx] = sum_neg + max_pos;
        else
            out[idx] = tile[tid];
    }
}

int main()
{

    // int N = 8192; // 2^13 chico
    // int N = 4194304; // 2^20 grande
    int N = 268435456; // 2^28 enorme
    int size = N * sizeof(int);

    int *h_in = (int *)malloc(size);
    int *h_out = (int *)malloc(size);

    // mezcla de positivos y negativos
    for (int i = 0; i < N; i++)
    {
        if (i % 5 == 0)
            h_in[i] = -i;
        else
            h_in[i] = i;
    }

    /*
    printf("Entrada:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_in[i]);
    }
    printf("\n\n");
    */

    int *d_in, *d_out;

    cudaMalloc(&d_in, size);

    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int blockSize = 32;
    int gridSize = N / 32;

    const int repeticiones = 10;
    std::vector<float> tiempos(repeticiones);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < repeticiones; i++)
    {
        cudaEventRecord(start);
        funcion_secuencial<<<gridSize, blockSize>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tiempos[i], start, stop);
    }

    float suma = std::accumulate(tiempos.begin(), tiempos.end(), 0.0f);
    float promedio = suma / repeticiones;
    float varianza = 0.0;
    for (float t : tiempos)
        varianza += pow(t - promedio, 2);
    float desv_est = sqrt(varianza / repeticiones);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    /*
    printf("Salida:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n\n");
    */

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}