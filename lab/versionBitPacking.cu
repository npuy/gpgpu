#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numeric>
#include <chrono>
#include <cmath>
#include <stdint.h>
#include "cuda.h"
#include "cublas_v2.h"

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

// Kernel Normas sobre datos empaquetados: ||xi||^2 = 4*popc(a1) + 4*popc(a1&a0) + popc(a0)
__global__ void kernel_normas(const uint32_t *X, float *normas, int n, int m)
{
    int fila = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < n)
    {
        float suma = 0.0f;
        int packed_cols = (m + 15) / 16;
        const uint32_t *indice_fila = X + fila * packed_cols;

        for (int col = 0; col < packed_cols; col++)
        {
            uint32_t packed = indice_fila[col];

            uint32_t a0 = packed & 0x55555555u;        // bits bajos de los 16 SNPs
            uint32_t a1 = (packed >> 1) & 0x55555555u; // bits altos de los 16 SNPs

            suma += 4.0f * __popc(a1) + 4.0f * __popc(a1 & a0) + __popc(a0);
        }
        normas[fila] = suma;
    }
}

#define TILE 32

// Kernel multiplicacion X * X^T con bit-slicing + __popc (solo triangulo superior)
__global__ void kernel_xxt(const uint32_t *X, float *C, int n, int m)
{
    // Salteo los bloques bajo la diagonal (simetria); evita divergencia dentro del bloque.
    if (blockIdx.x < blockIdx.y)
    {
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

    for (int t = 0; t < cant_tiles; t++)
    {
        int packedIdx = t * TILE + tx;

        // Cargar la fila 'row' de X
        if (row < n && packedIdx < packed_cols)
        {
            tile_A[ty][tx] = X[row * packed_cols + packedIdx];
        }
        else
        {
            tile_A[ty][tx] = 0;
        }

        // Cargar la fila 'col' de X
        int bRow = blockIdx.x * TILE + ty;

        if (bRow < n && packedIdx < packed_cols)
        {
            tile_B[ty][tx] = X[bRow * packed_cols + packedIdx];
        }
        else
        {
            tile_B[ty][tx] = 0;
        }

        __syncthreads();

        // Producto punto de los 16 SNPs empaquetados de una sola vez (bit-slicing + __popc)
        for (int k = 0; k < TILE; k++)
        {
            uint32_t a = tile_A[ty][k];
            uint32_t b = tile_B[tx][k];

            uint32_t a0 = a & 0x55555555u;        // bits bajos de cada SNP en a
            uint32_t a1 = (a >> 1) & 0x55555555u; // bits altos de cada SNP en a
            uint32_t b0 = b & 0x55555555u;        // bits bajos de cada SNP en b
            uint32_t b1 = (b >> 1) & 0x55555555u; // bits altos de cada SNP en b

            sum += 4.0f * __popc(a1 & b1) + 2.0f * __popc(a1 & b0) + 2.0f * __popc(a0 & b1) + __popc(a0 & b0);
        }

        __syncthreads();
    }

    if (row < n && col < n)
    {
        C[row * n + col] = sum;
    }
}

// Kernel distancia al cuadrado: D2 = N + N^T - 2*XXt (C en triangulo superior, se accede espejado)
__global__ void kernel_distance_squared(const float *norms, const float *C, float *D2, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n)
    {
        return;
    }

    float c_val;
    if (row <= col)
    {
        c_val = C[row * n + col];
    }
    else
    {
        c_val = C[col * n + row];
    }

    D2[row * n + col] = norms[row] + norms[col] - 2.0f * c_val;
}

__global__ void kernel_sqrt(float *D2, float *D, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float x = D2[row * n + col];

        if (x < 0.0f)
        {
            x = 0.0f;
        }
        D[row * n + col] = sqrtf(x);
    }
}

// Conversion uint8 (SNPs originales 0/1/2 sin empaquetar) -> float, para la referencia cuBLAS (FP32)
__global__ void kernel_u8_to_float(const uint8_t *X, float *Xf, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        Xf[idx] = (float)X[idx];
    }
}

// Bit-packing en CPU: 16 SNPs (2 bits c/u) por uint32_t. Conversion de entrada, NO se mide.
void bit_packing(uint8_t *h_X, uint32_t *h_X_packed, int n, int m)
{
    int packed_cols = (m + 15) / 16; // cada uint32_t almacena 16 SNPs
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < packed_cols; j++)
        {
            uint32_t valor_packed = 0;
            for (int k = 0; k < 16; ++k)
            {
                int snp_index = j * 16 + k;
                if (snp_index < m)
                {
                    uint32_t snp_aux = h_X[i * m + snp_index] & 0x03;
                    valor_packed |= snp_aux << (2 * k);
                }
            }
            h_X_packed[i * packed_cols + j] = valor_packed;
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, "Uso: %s N M [verify]\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]); // cant individuos
    int m = atoi(argv[2]); // cant SNPs por individuo
    int verify = (argc > 3 && strcmp(argv[3], "verify") == 0);

    // Generar matriz aleatoria de SNPs
    uint8_t *h_X = (uint8_t *)malloc(n * m * sizeof(uint8_t));
    srand(0);
    for (int i = 0; i < n * m; i++)
    {
        h_X[i] = rand() % 3;
    }

    int packed_cols = (m + 15) / 16;

    /* BIT PACKING EN CPU (conversion de entrada, una sola vez, NO se mide) */
    uint32_t *h_X_packed = (uint32_t *)malloc(n * packed_cols * sizeof(uint32_t));
    auto packing_inicio = std::chrono::high_resolution_clock::now();
    bit_packing(h_X, h_X_packed, n, m);
    auto packing_fin = std::chrono::high_resolution_clock::now();
    double tiempo_packing = std::chrono::duration<double, std::milli>(packing_fin - packing_inicio).count();
    printf("Bit packing CPU: %.3f ms\n", tiempo_packing);

    /* Mem en host para resultado final */
    float *h_D = (float *)malloc(n * n * sizeof(float));

    /* Reservo memoria en device para X, N, C = XXt, D2 y D */
    uint32_t *d_X;
    float *d_normas;
    float *d_C;
    float *d_D2;
    float *d_D;

    CUDA_CHK(cudaMalloc(&d_X, n * packed_cols * sizeof(uint32_t)));
    CUDA_CHK(cudaMalloc(&d_normas, n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_C, n * n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_D2, n * n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_D, n * n * sizeof(float)));

    /* Grillas y bloques */
    dim3 blockNorms(256);
    dim3 gridNorms((n + blockNorms.x - 1) / blockNorms.x);

    dim3 block2D(32, 32);
    dim3 grid2D((n + 31) / 32, (n + 31) / 32);

    /* Eventos: total del pipeline + un evento por cada paso (para desglosar tiempos).
       ev[0..6] delimitan los 6 pasos: H->D, normas, mult, distance, sqrt, D->H */
    cudaEvent_t start, stop, ev[7];
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 7; i++)
        cudaEventCreate(&ev[i]);

    /* Warm-up (no medido) */
    cudaMemcpy(d_X, h_X_packed, n * packed_cols * sizeof(uint32_t), cudaMemcpyHostToDevice);
    kernel_normas<<<gridNorms, blockNorms>>>(d_X, d_normas, n, m);
    kernel_xxt<<<grid2D, block2D>>>(d_X, d_C, n, m);
    kernel_distance_squared<<<grid2D, block2D>>>(d_normas, d_C, d_D2, n);
    kernel_sqrt<<<grid2D, block2D>>>(d_D2, d_D, n);
    cudaMemcpy(h_D, d_D, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    /* Medicion: H->D (empaquetado) + normas + xxt + distance + sqrt + D->H (desglosado por paso) */
    const int REPETICIONES = 10;
    float tHtoD = 0, tNormas = 0, tMult = 0, tDist = 0, tSqrt = 0, tDtoH = 0;
    cudaEventRecord(start);
    for (int rep = 0; rep < REPETICIONES; rep++)
    {
        cudaEventRecord(ev[0]);
        cudaMemcpy(d_X, h_X_packed, n * packed_cols * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaEventRecord(ev[1]);
        kernel_normas<<<gridNorms, blockNorms>>>(d_X, d_normas, n, m);

        cudaEventRecord(ev[2]);
        kernel_xxt<<<grid2D, block2D>>>(d_X, d_C, n, m);

        cudaEventRecord(ev[3]);
        kernel_distance_squared<<<grid2D, block2D>>>(d_normas, d_C, d_D2, n);

        cudaEventRecord(ev[4]);
        kernel_sqrt<<<grid2D, block2D>>>(d_D2, d_D, n);

        cudaEventRecord(ev[5]);
        cudaMemcpy(h_D, d_D, n * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(ev[6]);

        cudaEventSynchronize(ev[6]);
        float dt;
        cudaEventElapsedTime(&dt, ev[0], ev[1]);
        tHtoD += dt;
        cudaEventElapsedTime(&dt, ev[1], ev[2]);
        tNormas += dt;
        cudaEventElapsedTime(&dt, ev[2], ev[3]);
        tMult += dt;
        cudaEventElapsedTime(&dt, ev[3], ev[4]);
        tDist += dt;
        cudaEventElapsedTime(&dt, ev[4], ev[5]);
        tSqrt += dt;
        cudaEventElapsedTime(&dt, ev[5], ev[6]);
        tDtoH += dt;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_CHK(cudaGetLastError());

    float tiempoTotal = 0.0f;
    cudaEventElapsedTime(&tiempoTotal, start, stop);
    tiempoTotal /= REPETICIONES;
    tHtoD /= REPETICIONES;
    tNormas /= REPETICIONES;
    tMult /= REPETICIONES;
    tDist /= REPETICIONES;
    tSqrt /= REPETICIONES;
    tDtoH /= REPETICIONES;

    printf("Tiempo total (H->D + kernels + D->H): %.3f ms\n", tiempoTotal);
    printf("  H->D            : %.3f ms\n", tHtoD);
    printf("  normas          : %.3f ms\n", tNormas);
    printf("  multiplicacion  : %.3f ms\n", tMult);
    printf("  distance_squared: %.3f ms\n", tDist);
    printf("  sqrt            : %.3f ms\n", tSqrt);
    printf("  D->H            : %.3f ms\n", tDtoH);
    printf("TIME_MS %.3f\n", tiempoTotal);
    printf("HTOD_MS %.3f\n", tHtoD);
    printf("NORMAS_MS %.3f\n", tNormas);
    printf("MULT_MS %.3f\n", tMult);
    printf("DIST_MS %.3f\n", tDist);
    printf("SQRT_MS %.3f\n", tSqrt);
    printf("DTOH_MS %.3f\n", tDtoH);

    /* Verificacion opcional contra referencia cuBLAS (cublasSsyrk) sobre datos SIN empaquetar */
    if (verify)
    {
        cublasHandle_t handle;
        cublasCreate(&handle);

        uint8_t *d_Xorig;
        float *d_Xf, *d_G, *d_Dref;
        CUDA_CHK(cudaMalloc(&d_Xorig, n * m * sizeof(uint8_t)));
        CUDA_CHK(cudaMalloc(&d_Xf, n * m * sizeof(float)));
        CUDA_CHK(cudaMalloc(&d_G, n * n * sizeof(float)));
        CUDA_CHK(cudaMalloc(&d_Dref, n * n * sizeof(float)));

        // Datos originales (0/1/2) -> float
        cudaMemcpy(d_Xorig, h_X, n * m * sizeof(uint8_t), cudaMemcpyHostToDevice);
        int total = n * m;
        kernel_u8_to_float<<<(total + 255) / 256, 256>>>(d_Xorig, d_Xf, total);

        // G = X X^T con cuBLAS. FILL_MODE_LOWER (column-major) == triangulo superior row-major.
        const float uno = 1.0f, cero = 0.0f;
        cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                    n, m, &uno, d_Xf, m, &cero, d_G, n);

        kernel_distance_squared<<<grid2D, block2D>>>(d_normas, d_G, d_D2, n);
        kernel_sqrt<<<grid2D, block2D>>>(d_D2, d_Dref, n);
        CUDA_CHK(cudaGetLastError());

        // Refresco d_D con el resultado de esta version (el loop dejo d_D2 pisado)
        cudaMemcpy(d_X, h_X_packed, n * packed_cols * sizeof(uint32_t), cudaMemcpyHostToDevice);
        kernel_normas<<<gridNorms, blockNorms>>>(d_X, d_normas, n, m);
        kernel_xxt<<<grid2D, block2D>>>(d_X, d_C, n, m);
        kernel_distance_squared<<<grid2D, block2D>>>(d_normas, d_C, d_D2, n);
        kernel_sqrt<<<grid2D, block2D>>>(d_D2, d_D, n);

        // ||D - D_ref||_2
        const float menosUno = -1.0f;
        float err_abs = 0.0f, norm_ref = 0.0f;
        cublasSnrm2(handle, n * n, d_Dref, 1, &norm_ref);
        cublasSaxpy(handle, n * n, &menosUno, d_Dref, 1, d_D, 1); // d_D = d_D - d_Dref
        cublasSnrm2(handle, n * n, d_D, 1, &err_abs);
        CUDA_CHK(cudaDeviceSynchronize());

        float err_rel = (norm_ref > 0.0f) ? err_abs / norm_ref : err_abs;
        printf("ERR_ABS %.6e\n", err_abs);
        printf("ERR_REL %.6e\n", err_rel);

        cudaFree(d_Xorig);
        cudaFree(d_Xf);
        cudaFree(d_G);
        cudaFree(d_Dref);
        cublasDestroy(handle);
    }

    /* Elimino eventos */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < 7; i++)
        cudaEventDestroy(ev[i]);

    /* Libero memoria */
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
