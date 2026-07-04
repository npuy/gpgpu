#include <stdio.h>
#include <stdlib.h>
#include <numeric>
#include <time.h>
#include <cmath>
#include <stdint.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cuda.h"

using namespace nvcuda;

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
Version 3 con WMMA

La idea va a ser pasar el kernel XXt a que use WMMA, el resto de los kernels los dejo igual (por lo menos x ahora)

Hcer grafica comparativa de tiepo XXt 
*/

// Kernel Normas
// Para cada fila de X, calcula la norma ||xi||^2 y la guarda en normas[i]
// Solo se calcula un valor para cada fila (individuo), la matriz Ñ seria en cada fila repetir el valor j veces, no tiene sentido almacenar todo
/*
__global__ void kernel_normas(const half* X, float* normas, int n, int m) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (fila < n) {
        float suma = 0.0f;
        const half* indice_fila = X + fila * m;
        
        for (int col = 0; col < m; col++) {
            half val = indice_fila[col];
            float f = __half2float(val); // cambio aca, al usar half los tengo que pasar a float
            suma += f*f;
        }
        
        normas[fila] = suma;
    }
}
*/
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

// Kernel XXt con WMMA
/*
__global__ void wmma_xxt(const half *X, float *C, int n, int m) {

    // Cada bloque tiene 4 warps
    int warpId = threadIdx.x / 32;   // 0..3
    int lane   = threadIdx.x % 32;

    // Distribución 2x2 de warps dentro del bloque
    int warpRow = warpId / 2;
    int warpCol = warpId % 2;

    // Tile de 16x16 que calcula este warp
    int row = blockIdx.y * 32 + warpRow * 16;
    int col = blockIdx.x * 32 + warpCol * 16;

    if (blockIdx.x < blockIdx.y)
        return;

    int row = blockIdx.y * 16;
    int col = blockIdx.x * 16;

    if (row >= n || col >= n)
        return;

    __shared__ half As[16][17];
    __shared__ half Bs[16][17];

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,float> c_frag;

    wmma::fill_fragment(c_frag,0.0f);

    int lane = threadIdx.x;

    for(int k=0;k<m;k+=16)
    {
        // cada hilo copia 8 elementos
        for(int idx=lane; idx<256; idx+=32)
        {
            int i = idx / 16;
            int j = idx % 16;

            As[i][j] = X[(row+i)*m + (k+j)];
            Bs[i][j] = X[(col+i)*m + (k+j)];
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, &As[0][0], 16);
        wmma::load_matrix_sync(b_frag, &Bs[0][0], 16);

        wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(
        C + row*n + col,
        c_frag,
        n,
        wmma::mem_row_major);
}
*/

/*
La idea es esta: 
Se lanza un bloque de 128 hilos = 4 warps, cada warp calcula un tile independiente de 16x16 usando
tensor cores, por lo que el bloque completo produce un tile de salida de 32x32
*/

__global__ void wmma_xxt(const half *X, float *C, int n, int m) {
    int warpId = threadIdx.x / 32;  
    int warpRow = warpId / 2;
    int warpCol = warpId % 2;

    // salteo los bloques debajo de diagonal (simetria)
    if (blockIdx.x < blockIdx.y) {
        return;
    }

    int blockRow = blockIdx.y * 32; // coords del tile que procesa este bloque
    int blockCol = blockIdx.x * 32;

    int row = blockRow + warpRow * 16;   // fila base del sub-tile de este warp
    int col = blockCol + warpCol * 16;   // col  base del sub-tile de este warp

    // Shared memory para las 32 filas que necesita TODO el bloque (A y B),
    // un tile de K=16 por vez
    // Se almacenan las 32 filas que necesita todo el bloque pero solamente un tile de K=16 columnas por vez.
    // As contiene filas de X correspondientes al lado izquierdo.
    // Bs contiene filas de X correspondientes al lado derecho que wmma la slee como columnas, como si fuese Xt
    __shared__ half As[32][16];
    __shared__ half Bs[32][16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // recorro los k SNPs de a bloques de 16 columnas
    for (int k = 0; k < m; k += 16) {

        // Los 128 hilos del bloque copian conjuntamente un tile de 32x16 para A y otro tile de 32x16 para B
        // De esta forma cada dato se lee UNA sola vez desde mem global y luego es reutilizado por los cuatro
        // warps desde shared memory
        for (int idx = threadIdx.x; idx < 32 * 16; idx += 128) {
            int i = idx / 16; // coords en el tile
            int j = idx % 16;

            int globalRowA = blockRow + i;   // filas de X para lado A
            int globalRowB = blockCol + i;   // filas de X para lado B ("X^T")
            int globalCol  = k + j;

            As[i][j] = X[(blockRow + i) * m + (k + j)];
            Bs[i][j] = X[(blockCol + i) * m + (k + j)];
        }

        __syncthreads();

        // Cada warp toma su sub-bloque de 16x16 dentro de As/Bs
        wmma::load_matrix_sync(a_frag, &As[warpRow * 16][0], 16);
        wmma::load_matrix_sync(b_frag, &Bs[warpCol * 16][0], 16);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    if (row < n && col < n) {
        wmma::store_matrix_sync(C + row * n + col, c_frag, n, wmma::mem_row_major);
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

__global__ void uint8_to_half(const uint8_t *src, half *dst, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size) {
        dst[i] = __int2half_rn(src[i]);
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

    uint8_t *d_X; // el formato general para normas, como en las otras versiones
    half *d_X_half; // el formato que le paso a wmma
    float *d_normas;
    float *d_C;
    float *d_D2;
    float *d_D;

    CUDA_CHK(cudaMalloc(&d_X, n * m * sizeof(uint8_t)));
    CUDA_CHK(cudaMalloc(&d_X_half, n*m*sizeof(half)));
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

    // Por ahora pruebo asi,
    dim3 blockWMMA(128);
    dim3 gridWMMA((n + 31)/32,
                (n + 31)/32);

    // Para convertir de float a half la matriz X
    int total = n * m;
    dim3 blockConvert(256);
    dim3 gridConvert((total + blockConvert.x - 1) / blockConvert.x);
    uint8_to_half<<<gridConvert, blockConvert>>>(d_X, d_X_half, total);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

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
        wmma_xxt<<<gridWMMA, blockWMMA>>>(d_X_half, d_C, n, m);
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
    cudaFree(d_X_half);
    cudaFree(d_normas);
    cudaFree(d_C);
    cudaFree(d_D2);
    cudaFree(d_D);

    free(h_X);
    free(h_D);


    return 0;
}


