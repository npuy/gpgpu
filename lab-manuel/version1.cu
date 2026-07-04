#include <stdio.h>
#include <stdlib.h>
#include <numeric>
#include <time.h>
#include <cmath>
#include <stdint.h>
#include "cuda.h"

#define CUDA_CHK(ans) {gpuAssert((ans), __FILE__, __LINE__);}
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
La idea es:
1. Kernel XXt, el que voy a optimizar
2. Kernel normas N
3. Kernels suma final y raiz cuadrada

En esta version inicial, XXt se hace aprovechando memoria compartida y haciendo accesos coalesced, y aprovechando simetria
*/

// Kernel Normas
// Para cada fila de X, calcula la norma ||xi||^2 y la guarda en normas[i]
// Solo se calcula un valor para cada fila (individuo), la matriz Ñ seria en cada fila repetir el valor j veces, no tiene sentido almacenar todo
__global__ void kernel_normas(const uint8_t* X, float* normas, int n, int m) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (fila < n) {
        float suma = 0.0f;
        const uint8_t* indice_fila = X + fila * m;
        
        for (int col = 0; col < m; col++) {
            uint8_t val = indice_fila[col];
            suma += (float)(val * val);
        }
        
        normas[fila] = suma;
    }
}

#define TILE 32

__global__ void kernel_xxt(const uint8_t* X, float* C, int n, int m) {

    // Agrego esto para que solo se calcule la mitad triangular superior
    // Otra opcion es if row > col return, pero hay divergencia de hilos. 
    // En este caso con los bloques en la diagonal se calculan algunos valores extra de mas, pero la gpu es rapida y asi no sufre divergencia, me parece mas prolijo
    if (blockIdx.x < blockIdx.y) {
        return;
    }

    __shared__ uint8_t tile_A[TILE][TILE + 1];
    __shared__ uint8_t tile_B[TILE][TILE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    int cant_tiles = (m + TILE - 1) / TILE;

    for (int t = 0; t < cant_tiles; t++) {
        int kIdx = t * TILE + tx;

        // Cargar la fila 'row'' de X
        if (row < n && kIdx < m)
            tile_A[ty][tx] = X[row * m + kIdx]; // coalesced
        else
            tile_A[ty][tx] = 0;

        // Cargar la fila "col" de X
        int bRow = blockIdx.x * TILE + ty;

        if (bRow < n && kIdx < m)
            tile_B[ty][tx] = X[bRow * m + kIdx]; // coalesced
        else
            tile_B[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += tile_A[ty][k] * tile_B[tx][k];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Kernel distancia al cuadrado
// Aca hago N + Nt - 2XXt, y guardo el resultado en D
// en particular, como XXt esta en triangulo superior, tengo que acceder de forma espejada a los elementos triang inferiores
__global__ void kernel_distance_squared(const float* norms, const float* C, float* D2, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= n || col >= n) {
        return;
    }
    
    float c_val;
    if (row <= col) {
        // C en triang superior
        c_val = C[row * n + col];
    } else {
        // C en traing inferior, accedo espejado
        c_val = C[col * n + row];
    }
    
    D2[row * n + col] = norms[row] + norms[col] - 2.0f * c_val;
}

__global__ void kernel_sqrt(float *D2, float *D, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < n && col < n) {
        float x = D2[row*n + col];

        if(x < 0.0f) {
            x = 0.0f;
        }
        D[row*n + col] = sqrtf(x);
    }
}


int main(int argc, char *argv[]) {
    int n = atoi(argv[1]); // cant individuos
    int m = atoi(argv[2]); // cant SNPs por individuo

    // Generar matriz aleatoria de SNPs
    uint8_t *h_X = (uint8_t*)malloc(n*m*sizeof(uint8_t));
    srand(0);
    for(int i=0;i<n*m;i++) {
        h_X[i] = rand()%3;
    }


    float *h_D = (float*)malloc(n*n*sizeof(float));

    /* Reservo memoria en device para X, N, C = XXt, D y D2 */

    uint8_t *d_X;
    float *d_normas;
    float *d_C;
    float *d_D2;
    float *d_D;

    CUDA_CHK(cudaMalloc(&d_X, n*m*sizeof(uint8_t)));
    CUDA_CHK(cudaMalloc(&d_normas, n*sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_C, n*n*sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_D2, n*n*sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_D, n*n*sizeof(float)));

    /* Copio X de host a device */

    cudaEvent_t startMemcpy, stopMemcpy;
    cudaEventCreate(&startMemcpy);
    cudaEventCreate(&stopMemcpy);
    cudaEventRecord(startMemcpy);
    CUDA_CHK(cudaMemcpy(d_X, h_X, n*m*sizeof(uint8_t), cudaMemcpyHostToDevice));
    cudaEventRecord(stopMemcpy);
    cudaEventSynchronize(stopMemcpy);
    float tiempoMemcpy;
    cudaEventElapsedTime(&tiempoMemcpy, startMemcpy,stopMemcpy);

    /* Para elegir grillas y bloques lo hago personalizado a cada tarea */
    dim3 blockNorms(256);
    dim3 gridNorms((n + blockNorms.x - 1) / blockNorms.x);

    dim3 block2D(32,32);
    dim3 grid2D((n + 31)/32, (n + 31)/32);

    /* LANZO KERNELS */

    /* Eventos para medir itempos de los kernels */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float tiempoNormas = 0.0f;
    float tiempoXXT    = 0.0f;
    float tiempoDist   = 0.0f;
    float tiempoSqrt   = 0.0f;
    float tiempo       = 0.0f;


    const int REPETICIONES = 10;
    for (int rep = 0; rep < REPETICIONES; rep++) {

        // Normas
        cudaEventRecord(start);
        kernel_normas<<<gridNorms, blockNorms>>>(d_X, d_normas, n, m);
        CUDA_CHK(cudaGetLastError());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tiempo, start, stop);
        tiempoNormas += tiempo;

        // XXT
        cudaEventRecord(start);
        kernel_xxt<<<grid2D, block2D>>>(d_X, d_C, n, m);
        CUDA_CHK(cudaGetLastError());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tiempo, start, stop);
        tiempoXXT += tiempo;

        // Distancia^2
        cudaEventRecord(start);
        kernel_distance_squared<<<grid2D, block2D>>>(d_normas, d_C, d_D2, n);
        CUDA_CHK(cudaGetLastError());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tiempo, start, stop);
        tiempoDist += tiempo;

        // Sqrt
        cudaEventRecord(start);
        kernel_sqrt<<<grid2D, block2D>>>(d_D2, d_D, n);
        CUDA_CHK(cudaGetLastError());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tiempo, start, stop);
        tiempoSqrt += tiempo;
    }

    tiempoNormas /= REPETICIONES;
    tiempoXXT /= REPETICIONES;
    tiempoDist /= REPETICIONES;
    tiempoSqrt /= REPETICIONES;

    /* Imprimir resultados */
    printf("Memcpy H->D: %.3f ms\n\n", tiempoMemcpy);
    printf("Normas: %.3f ms\n\n", tiempoNormas);
    printf("XXT: %.3f ms\n\n", tiempoXXT);
    printf("Distance^2: %.3f ms\n\n", tiempoDist);
    printf("Sqrt: %.3f ms\n\n", tiempoSqrt);

    /* Transfiero el resultado de device a host */

    CUDA_CHK(cudaMemcpy(h_D, d_D, n*n*sizeof(float), cudaMemcpyDeviceToHost));

    /* Elimino estos eventos x prolijidad */
    cudaEventDestroy(startMemcpy);
    cudaEventDestroy(stopMemcpy);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* Libero memoria */

    cudaFree(d_X);
    cudaFree(d_normas);
    cudaFree(d_C);
    cudaFree(d_D2);
    cudaFree(d_D);

    free(h_X);
    free(h_D);


    return 0;
}