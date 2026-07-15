#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numeric>
#include <time.h>
#include <cmath>
#include <stdint.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cuda.h"
#include "cublas_v2.h"

using namespace nvcuda;

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

// Kernel Normas: ||xi||^2 por fila, sobre los datos uint8 originales
__global__ void kernel_normas(const uint8_t *X, float *normas, int n, int m)
{
    int fila = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < n)
    {
        float suma = 0.0f;
        const uint8_t *indice_fila = X + fila * m;

        for (int col = 0; col < m; col++)
        {
            uint8_t val = indice_fila[col];
            suma += (float)(val * val);
        }

        normas[fila] = suma;
    }
}

// Kernel multiplicacion X * X^T con WMMA (bloque de 128 hilos = 4 warps -> tile de salida 32x32).
// Cada warp calcula un sub-tile 16x16 con tensor cores; se saltean los bloques bajo la diagonal.
__global__ void wmma_xxt(const half *X, float *C, int n, int m)
{
    int warpId = threadIdx.x / 32;
    int warpRow = warpId / 2;
    int warpCol = warpId % 2;

    // salteo los bloques debajo de la diagonal (simetria)
    if (blockIdx.x < blockIdx.y)
    {
        return;
    }

    int blockRow = blockIdx.y * 32; // coords del tile que procesa este bloque
    int blockCol = blockIdx.x * 32;

    int row = blockRow + warpRow * 16; // fila base del sub-tile de este warp
    int col = blockCol + warpCol * 16; // col  base del sub-tile de este warp

    // Shared memory: 32 filas (para A y B), un tile de K=16 por vez.
    // As guarda las filas del lado izquierdo; Bs las del lado derecho (WMMA las lee como columnas, X^T).
    __shared__ half As[32][16];
    __shared__ half Bs[32][16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // recorro los K SNPs de a bloques de 16 columnas
    for (int k = 0; k < m; k += 16)
    {

        // Los 128 hilos copian conjuntamente un tile 32x16 para A y otro para B (cada dato se lee una
        // sola vez desde global y luego lo reutilizan los 4 warps desde shared).
        for (int idx = threadIdx.x; idx < 32 * 16; idx += 128)
        {
            int i = idx / 16;
            int j = idx % 16;

            As[i][j] = X[(blockRow + i) * m + (k + j)];
            Bs[i][j] = X[(blockCol + i) * m + (k + j)];
        }

        __syncthreads();

        // Cada warp toma su sub-bloque 16x16 dentro de As/Bs
        wmma::load_matrix_sync(a_frag, &As[warpRow * 16][0], 16);
        wmma::load_matrix_sync(b_frag, &Bs[warpCol * 16][0], 16);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    if (row < n && col < n)
    {
        wmma::store_matrix_sync(C + row * n + col, c_frag, n, wmma::mem_row_major);
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

// Conversion de entrada uint8 -> half (para WMMA). NO se mide (una sola vez).
__global__ void uint8_to_half(const uint8_t *src, half *dst, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        dst[i] = __int2half_rn(src[i]);
    }
}

// Conversion uint8 -> float, para la referencia cuBLAS (FP32) del modo verify
__global__ void kernel_u8_to_float(const uint8_t *X, float *Xf, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        Xf[idx] = (float)X[idx];
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

    float *h_D = (float *)malloc(n * n * sizeof(float));

    /* Reservo memoria en device para X, N, C = XXt, D2 y D */
    uint8_t *d_X;   // formato general para normas, como en las otras versiones
    half *d_X_half; // formato que consume WMMA
    float *d_normas;
    float *d_C;
    float *d_D2;
    float *d_D;

    CUDA_CHK(cudaMalloc(&d_X, n * m * sizeof(uint8_t)));
    CUDA_CHK(cudaMalloc(&d_X_half, n * m * sizeof(half)));
    CUDA_CHK(cudaMalloc(&d_normas, n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_C, n * n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_D2, n * n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_D, n * n * sizeof(float)));

    /* Grillas y bloques */
    dim3 blockNorms(256);
    dim3 gridNorms((n + blockNorms.x - 1) / blockNorms.x);

    dim3 block2D(32, 32);
    dim3 grid2D((n + 31) / 32, (n + 31) / 32);

    dim3 blockWMMA(128);
    dim3 gridWMMA((n + 31) / 32, (n + 31) / 32);

    // Para convertir uint8 -> half la matriz X (una sola vez)
    int total = n * m;
    dim3 blockConvert(256);
    dim3 gridConvert((total + blockConvert.x - 1) / blockConvert.x);

    /* Eventos: total del pipeline + un evento por cada paso (para desglosar tiempos).
       ev[0..6] delimitan los 6 pasos: H->D, normas, mult, distance, sqrt, D->H */
    cudaEvent_t start, stop, ev[7];
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 7; i++)
        cudaEventCreate(&ev[i]);

    /* Conversion de entrada uint8 -> half: NO se mide, se hace una sola vez.
       El buffer d_X_half queda residente y es el que consume wmma_xxt. */
    cudaMemcpy(d_X, h_X, n * m * sizeof(uint8_t), cudaMemcpyHostToDevice);
    uint8_to_half<<<gridConvert, blockConvert>>>(d_X, d_X_half, total);

    /* Warm-up (no medido) */
    cudaMemcpy(d_X, h_X, n * m * sizeof(uint8_t), cudaMemcpyHostToDevice);
    kernel_normas<<<gridNorms, blockNorms>>>(d_X, d_normas, n, m);
    wmma_xxt<<<gridWMMA, blockWMMA>>>(d_X_half, d_C, n, m);
    kernel_distance_squared<<<grid2D, block2D>>>(d_normas, d_C, d_D2, n);
    kernel_sqrt<<<grid2D, block2D>>>(d_D2, d_D, n);
    cudaMemcpy(h_D, d_D, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    /* Medicion: H->D (uint8) + normas + wmma + distance + sqrt + D->H (desglosado por paso) */
    const int REPETICIONES = 10;
    float tHtoD = 0, tNormas = 0, tMult = 0, tDist = 0, tSqrt = 0, tDtoH = 0;
    cudaEventRecord(start);
    for (int rep = 0; rep < REPETICIONES; rep++)
    {
        cudaEventRecord(ev[0]);
        cudaMemcpy(d_X, h_X, n * m * sizeof(uint8_t), cudaMemcpyHostToDevice);

        cudaEventRecord(ev[1]);
        kernel_normas<<<gridNorms, blockNorms>>>(d_X, d_normas, n, m);

        cudaEventRecord(ev[2]);
        wmma_xxt<<<gridWMMA, blockWMMA>>>(d_X_half, d_C, n, m);

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

    /* Verificacion opcional contra referencia cuBLAS (cublasSsyrk) */
    if (verify)
    {
        cublasHandle_t handle;
        cublasCreate(&handle);

        float *d_Xf, *d_G, *d_Dref;
        CUDA_CHK(cudaMalloc(&d_Xf, n * m * sizeof(float)));
        CUDA_CHK(cudaMalloc(&d_G, n * n * sizeof(float)));
        CUDA_CHK(cudaMalloc(&d_Dref, n * n * sizeof(float)));

        // d_X ya tiene los datos originales (0/1/2) en device -> float
        kernel_u8_to_float<<<gridConvert, blockConvert>>>(d_X, d_Xf, total);

        // G = X X^T con cuBLAS. FILL_MODE_LOWER (column-major) == triangulo superior row-major.
        const float uno = 1.0f, cero = 0.0f;
        cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                    n, m, &uno, d_Xf, m, &cero, d_G, n);

        kernel_distance_squared<<<grid2D, block2D>>>(d_normas, d_G, d_D2, n);
        kernel_sqrt<<<grid2D, block2D>>>(d_D2, d_Dref, n);
        CUDA_CHK(cudaGetLastError());

        // Refresco d_D con el resultado de esta version (el loop dejo d_D2 pisado)
        uint8_to_half<<<gridConvert, blockConvert>>>(d_X, d_X_half, total);
        kernel_normas<<<gridNorms, blockNorms>>>(d_X, d_normas, n, m);
        wmma_xxt<<<gridWMMA, blockWMMA>>>(d_X_half, d_C, n, m);
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
    cudaFree(d_X_half);
    cudaFree(d_normas);
    cudaFree(d_C);
    cudaFree(d_D2);
    cudaFree(d_D);

    free(h_X);
    free(h_D);

    return 0;
}
