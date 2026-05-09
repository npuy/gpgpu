#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <cmath>
#include "cuda.h"

#define CUDA_CHK(ans)                         \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void traspuesta(int *matriz_in, int *matriz_out, int n)
{
    __shared__ int tile[32][32];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < n)
    {
        tile[threadIdx.y][threadIdx.x] = matriz_in[y * n + x];
    }
    __syncthreads();

    x = blockIdx.y * blockDim.y + threadIdx.x; // intercambiamos x e y para escribir la transpuesta
    y = blockIdx.x * blockDim.x + threadIdx.y;
    if (x < n && y < n)
    {
        matriz_out[y * n + x] = tile[threadIdx.x][threadIdx.y]; // escribimos la transpuesta desde el tile
    }
}

int main(int argc, char *argv[])
{
    // int n = 8192; // 2^13
    int n = 16384; // 2^14
    unsigned int size = n * n * sizeof(int);

    int *A = (int *)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = i;
        }
    }

    // variables en host y device
    int *h_entrada;
    int *h_traspuesta;

    int *d_entrada;
    int *d_traspuesta;

    // asigno memoria en host
    h_entrada = A;
    h_traspuesta = (int *)malloc(size);

    // asigno memoria en device
    cudaMalloc((void **)&d_entrada, size);
    cudaMalloc((void **)&d_traspuesta, size);

    // paso la data a device, a la memoria recien asignada
    cudaMemcpy(d_entrada, h_entrada, size, cudaMemcpyHostToDevice);

    // tamaño bloques y grilla
    dim3 block_size(32, 32);
    dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y); // Esto es para que alcance a cubrir toda la matriz, aunque el bloque no divida exactamente a n

    // ejecuto el kernel 10 veces para hacer el promedio queme piden
    const int repeticiones = 10;
    std::vector<float> tiempos(repeticiones);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < repeticiones; i++)
    {
        cudaEventRecord(start);

        traspuesta<<<grid_size, block_size>>>(d_entrada, d_traspuesta, n);

        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tiempos[i], start, stop);
    }

    /*
    float suma = std::accumulate(tiempos.begin(), tiempos.end(), 0.0f);
    float promedio = suma / repeticiones;
    float varianza = 0.0;
    for(float t : tiempos) varianza += pow(t - promedio, 2);
    float desv_est = sqrt(varianza / repeticiones);

    printf("Promedio (10 iteraciones): %f ms\n", promedio);
    printf("Desviación estándar: %f ms\n", desv_est);
    */

    cudaMemcpy(h_traspuesta, d_traspuesta, size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_entrada);
    cudaFree(d_traspuesta);
    free(h_entrada);
    free(h_traspuesta);

    return 0;
}