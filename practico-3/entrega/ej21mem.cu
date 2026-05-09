#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <time.h>
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

__global__ void reduccion_mem(int *array_in, int *array_out, int n)
{
    __shared__ int tile_negativos[32];
    __shared__ int tile_positivos[32];

    // Cada hilo carga un elemento en mem compartida
    int tid = threadIdx.x;
    int id_global = (blockIdx.x * blockDim.x) + threadIdx.x;
    int valor = 0;
    if (id_global < n)
    {
        valor = array_in[id_global];
        int neg = (valor < 0) ? valor : 0;
        int pos = (valor > 0) ? valor : 0; // Aca asumo que siempre hay un positivo, sino el maximo podria ser un negativo
        tile_negativos[tid] = neg;
        tile_positivos[tid] = pos;
    }
    else
    {
        tile_negativos[tid] = 0;
        tile_positivos[tid] = 0;
    }

    // Espero todos los hilos hayan escrito en shared
    __syncthreads();

    // Reduccion en memoria compartida
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            tile_negativos[tid] += tile_negativos[tid + stride];
            tile_positivos[tid] = max(tile_positivos[tid], tile_positivos[tid + stride]);
        }
        __syncthreads();
    }

    __syncthreads();

    // Segun el valor original en la celda, veo si reemplazo por el resultado de la operacion
    if (id_global < n)
    {
        if (valor < 0)
        {
            array_out[id_global] = tile_negativos[0] + tile_positivos[0]; // suma paso 1 y paso 2
        }
        else
        {
            array_out[id_global] = valor;
        }
    }
}

int main()
{

    // int N = 8192; // 2^13 chico
    int N = 4194304; // 2^20 grande
    // int N = 268435456; // 2^28 enorme
    int size = N * sizeof(int);

    int *h_in = (int *)malloc(size);
    int *h_out = (int *)malloc(size);

    // mezcla de positivos y negativos
    /*
    for (int i = 0; i < N; i++) {
        if (i % 5 == 0)
            h_in[i] = -i;
        else
            h_in[i] = i;
    }
    */
    // arreglo aleatorio de enteros entre -100 y 100
    for (int i = 0; i < N; i++)
    {
        h_in[i] = (rand() % 201) - 100;
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
        reduccion_mem<<<gridSize, blockSize>>>(d_in, d_out, N);
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