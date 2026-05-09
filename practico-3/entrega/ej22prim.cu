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

__global__ void reduccion_primitivas(int *matriz_origen, int *matriz_destino, int n)
{
    int lane = threadIdx.x % 32;
    int id_global = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (id_global < n)
    {
        // Guardo el valor original asignado al hilo
        // Si aporta a suma_neg o max_pos lo guardo, sino guardo 0 para que no aporte en la reduccion
        int valor = matriz_origen[id_global];
        int suma_neg = (valor < 0) ? valor : 0;
        int max_pos = (valor > 0) ? valor : 0; // asumo hay positivos, por letra

        for (int offset = 16; offset > 0; offset /= 2)
        {
            suma_neg += __shfl_down_sync(0xFFFFFFFF, suma_neg, offset);
            max_pos = max(max_pos, __shfl_down_sync(0xFFFFFFFF, max_pos, offset));
        }

        // lane 0 tiene el resultado, lo difunde a los demas hilos
        int resultado = 0;
        if (lane == 0)
        {
            resultado = suma_neg + max_pos;
        }
        resultado = __shfl_sync(0xFFFFFFFF, resultado, 0);

        unsigned mask_neg = __ballot_sync(0xFFFFFFFF, valor < 0);

        // para el hilo que soy, me fijo si mi bit en el mask_neg esta prendido
        if ((mask_neg >> lane) & 1)
        {
            // corresponde reemplazar mi valor en matriz
            matriz_destino[id_global] = resultado;
        }
        else
        {
            matriz_destino[id_global] = valor;
        }
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
        reduccion_primitivas<<<gridSize, blockSize>>>(d_in, d_out, N);
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