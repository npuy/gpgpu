#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

// Norma ||xi||^2 por fila (a partir de los datos uint8)
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

// D2 = norms_i + norms_j - 2 * G_ij (G simetrica, se lee el triangulo superior)
__global__ void kernel_distance_squared(const float *norms, const float *G, float *D2, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n)
    {
        return;
    }

    float g_val;
    if (row <= col)
    {
        g_val = G[row * n + col];
    }
    else
    {
        g_val = G[col * n + row];
    }

    D2[row * n + col] = norms[row] + norms[col] - 2.0f * g_val;
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

// Conversion de entrada uint8 -> float (cublasSsyrk es FP32). NO se mide (una sola vez).
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
        fprintf(stderr, "Uso: %s N M\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]); // cant individuos
    int m = atoi(argv[2]); // cant SNPs por individuo

    // Generar matriz aleatoria de SNPs (misma semilla que las demas versiones)
    uint8_t *h_X = (uint8_t *)malloc(n * m * sizeof(uint8_t));
    srand(0);
    for (int i = 0; i < n * m; i++)
    {
        h_X[i] = rand() % 3;
    }

    float *h_D = (float *)malloc(n * n * sizeof(float));

    /* Memoria en device */
    uint8_t *d_X;
    float *d_Xf;
    float *d_normas;
    float *d_G;
    float *d_D2;
    float *d_D;

    CUDA_CHK(cudaMalloc(&d_X, n * m * sizeof(uint8_t)));
    CUDA_CHK(cudaMalloc(&d_Xf, n * m * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_normas, n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_G, n * n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_D2, n * n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_D, n * n * sizeof(float)));

    /* cuBLAS handle (fuera del loop de medicion) */
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float uno = 1.0f, cero = 0.0f;

    /* Grillas y bloques */
    dim3 blockNorms(256);
    dim3 gridNorms((n + blockNorms.x - 1) / blockNorms.x);
    dim3 block2D(32, 32);
    dim3 grid2D((n + 31) / 32, (n + 31) / 32);
    int total = n * m;

    /* Eventos: total del pipeline + un evento por cada paso (para desglosar tiempos).
       ev[0..6] delimitan los 6 pasos: H->D, normas, mult, distance, sqrt, D->H */
    cudaEvent_t start, stop, ev[7];
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 7; i++)
        cudaEventCreate(&ev[i]);

    /* Conversion de entrada uint8 -> float: NO se mide, se hace una sola vez.
       El buffer d_Xf queda residente y es el que consume cublasSsyrk. */
    cudaMemcpy(d_X, h_X, n * m * sizeof(uint8_t), cudaMemcpyHostToDevice);
    kernel_u8_to_float<<<(total + 255) / 256, 256>>>(d_X, d_Xf, total);

    /* Warm-up (no medido) */
    cudaMemcpy(d_X, h_X, n * m * sizeof(uint8_t), cudaMemcpyHostToDevice);
    kernel_normas<<<gridNorms, blockNorms>>>(d_X, d_normas, n, m);
    cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                n, m, &uno, d_Xf, m, &cero, d_G, n);
    kernel_distance_squared<<<grid2D, block2D>>>(d_normas, d_G, d_D2, n);
    kernel_sqrt<<<grid2D, block2D>>>(d_D2, d_D, n);
    cudaMemcpy(h_D, d_D, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    /* Medicion: H->D + normas + syrk + distance + sqrt + D->H (desglosado por paso) */
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
        cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                    n, m, &uno, d_Xf, m, &cero, d_G, n);

        cudaEventRecord(ev[3]);
        kernel_distance_squared<<<grid2D, block2D>>>(d_normas, d_G, d_D2, n);

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

    /* Limpieza */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < 7; i++)
        cudaEventDestroy(ev[i]);
    cublasDestroy(handle);

    cudaFree(d_X);
    cudaFree(d_Xf);
    cudaFree(d_normas);
    cudaFree(d_G);
    cudaFree(d_D2);
    cudaFree(d_D);

    free(h_X);
    free(h_D);

    return 0;
}
