#include <stdio.h>
#include <stdlib.h>
#include <numeric>
#include <chrono>
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
Esta es la version 2 modificada para usar enteros de 32 bits, y asi empaquetamos 16 SNPs por enteero
*/


__global__ void kernel_normas(const uint32_t* X, float* normas, int n, int m) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < n) {
        float suma = 0.0f;
        int packed_cols = (m + 15) / 16;
        const uint32_t* indice_fila = X + fila * packed_cols;

        for (int col = 0; col < packed_cols; col++) {
            uint32_t packed = indice_fila[col];

            // Extraigo los 16 SNPs almacenados en los 32 bits del entero
            for (int i = 0; i < 16; i++) {
                int snp_index = col * 16 + i;
                // el ultimo byte puede no estar todo lleno. Seguramente si porque vamos a asumir multiplos de 32 pero por las dudas...
                if (snp_index < m) {
                    uint32_t snp = (packed >> (2*i)) & 0x03;
                    suma += snp * snp;
                }
            }
        }
        normas[fila] = suma;
    }
}

#define TILE 32

__global__ void kernel_xxt(const uint32_t* X, float* C, int n, int m) {

    // Agrego esto para que solo se calcule la mitad triangular superior
    // Otra opcion es if row > col return, pero hay divergencia de hilos. 
    // En este caso con los bloques en la diagonal se calculan algunos valores extra de mas, pero la gpu es rapida y asi no sufre divergencia, me parece mas prolijo
    if (blockIdx.x < blockIdx.y) {
        return;
    }

    __shared__ uint32_t tile_A[TILE][TILE + 1];
    __shared__ uint32_t tile_B[TILE][TILE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    int packed_cols = (m + 15) / 16;
    int cant_tiles = (packed_cols + TILE - 1) / TILE;

    for (int t = 0; t < cant_tiles; t++) {
        int packedIdx = t * TILE + tx;

        // Cargar la fila 'row'' de X
        if (row < n && packedIdx < packed_cols) {
            tile_A[ty][tx] = X[row * packed_cols + packedIdx];

        } else {
            tile_A[ty][tx] = 0;
        }

        // Cargar la fila "col" de X
        int bRow = blockIdx.x * TILE + ty;

        if (bRow < n && packedIdx < packed_cols) {
            tile_B[ty][tx] = X[bRow * packed_cols + packedIdx];

        } else {
            tile_B[ty][tx] = 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            uint32_t a = tile_A[ty][k];
            uint32_t b = tile_B[tx][k];

            for (int i = 0; i < 16; i++) {
                int sa = (a >> (2*i)) & 0x3;
                int sb = (b >> (2*i)) & 0x3;
                sum += sa * sb;
            }
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

void bit_packing(uint8_t *h_X, uint32_t *h_X_packed, int n, int m) {
    int packed_cols = (m + 15) / 16; // Cada uint32_t tiene 32 bits, puede almacenar 16 SNPs
    for (int i = 0; i < n; ++i) { // para cada individuo
        for (int j = 0; j < packed_cols; j++) { // para cada grupo de 16 SNPs
            uint32_t valor_packed = 0;
            for (int k = 0; k < 16; ++k) {
                int snp_index = j * 16 + k;
                if (snp_index < m) {
                    uint32_t snp_aux = h_X[i*m + snp_index] & 0x03; // tomo los 2 bits del SNP, que estaban almacenados en 8 bits
                    valor_packed |= snp_aux << (2 * k); // los muevo dos lugares a la izquierda y los agrego al valor empaquetado haciendo un or bit a bit
                }
            }
            h_X_packed[i * packed_cols + j] = valor_packed;
        }
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

    // A este punto ya tengo X, ahora tengo que hacer el bit packing, y luego transferir a GPU la version empaquetada
    int packed_cols = (m + 15) / 16;

    /* BIT PACKING EN CPU Ejecuto bit packing y mido tiempos */
    uint32_t *h_X_packed = (uint32_t*)malloc(n * packed_cols * sizeof(uint32_t));
    auto packing_inicio = std::chrono::high_resolution_clock::now();
    bit_packing(h_X, h_X_packed, n, m);
    auto packing_fin = std::chrono::high_resolution_clock::now();
    double tiempo_packing = std::chrono::duration<double, std::milli>(packing_fin-packing_inicio).count();
    printf("Bit packing CPU: %.3f ms\n", tiempo_packing);
    

    /* Mem en host para resultado final */
    float *h_D = (float*)malloc(n*n*sizeof(float));

    /* Reservo memoria en device para X, N, C = XXt, D y D2 */
    uint32_t *d_X;
    float *d_normas;
    float *d_C;
    float *d_D2;
    float *d_D;

    CUDA_CHK(cudaMalloc(&d_X, n*packed_cols*sizeof(uint32_t)));
    CUDA_CHK(cudaMalloc(&d_normas, n*sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_C, n*n*sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_D2, n*n*sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_D, n*n*sizeof(float)));

    /* Copio X PACKED de host a device y mido tiempos */
    cudaEvent_t startMemcpy, stopMemcpy;
    cudaEventCreate(&startMemcpy);
    cudaEventCreate(&stopMemcpy);
    cudaEventRecord(startMemcpy);
    CUDA_CHK(cudaMemcpy(d_X, h_X_packed, n*packed_cols*sizeof(uint32_t), cudaMemcpyHostToDevice));
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

        cudaEventRecord(start);
        kernel_normas<<<gridNorms, blockNorms>>>(d_X, d_normas, n, m);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        CUDA_CHK(cudaGetLastError());
        //CUDA_CHK(cudaDeviceSynchronize());
        cudaEventElapsedTime(&tiempo,start,stop);
        tiempoNormas += tiempo;


        cudaEventRecord(start);
        kernel_xxt<<<grid2D,block2D>>>(d_X, d_C, n, m);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        CUDA_CHK(cudaGetLastError());
        //CUDA_CHK(cudaDeviceSynchronize());
        cudaEventElapsedTime(&tiempo,start,stop);
        tiempoXXT += tiempo;


        cudaEventRecord(start);
        kernel_distance_squared<<<grid2D, block2D>>>(d_normas, d_C, d_D2, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        CUDA_CHK(cudaGetLastError());
        //CUDA_CHK(cudaDeviceSynchronize());
        cudaEventElapsedTime(&tiempo,start,stop);
        tiempoDist += tiempo;


        cudaEventRecord(start);
        kernel_sqrt<<<grid2D, block2D>>>(d_D2, d_D, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        CUDA_CHK(cudaGetLastError());
        //CUDA_CHK(cudaDeviceSynchronize());
        cudaEventElapsedTime(&tiempo,start,stop);
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
    //cudaFree(d_X_raw);
    cudaFree(d_X);
    cudaFree(d_normas);
    cudaFree(d_C);
    cudaFree(d_D2);
    cudaFree(d_D);

    free(h_X_packed);
    free(h_X);
    free(h_D);


    return 0;
}