#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numeric>
#include <chrono>
#include <cmath>
#include <stdint.h>
#include <mma.h>
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

// Norma (popcounts por fila de cada plano): S1[i] = popc(fila i de X1), S0[i] = popc(fila i de X0).
// Sirven para las normas (||xi||^2 = 4*S1+S0) y para la correccion XOR->AND.
__global__ void kernel_plane_popcounts(const uint32_t *X1, const uint32_t *X0,
                                       int *S1, int *S0, int n, int m)
{
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    if (fila < n)
    {
        int wpr = (m + 31) / 32; // palabras por fila (32 SNPs/uint32 por plano)
        int c1 = 0, c0 = 0;
        const uint32_t *f1 = X1 + fila * wpr;
        const uint32_t *f0 = X0 + fila * wpr;
        for (int c = 0; c < wpr; c++)
        { // los bits de padding son 0 -> no aportan
            c1 += __popc(f1[c]);
            c0 += __popc(f0[c]);
        }
        S1[fila] = c1;
        S0[fila] = c0;
    }
}

// Bit-packing en CPU a dos planos de 1 bit (a1 = bit alto, a0 = bit bajo), 32 SNPs/uint32.
// Conversion de entrada, NO se mide.
void bit_packing_planes(const uint8_t *h_X, uint32_t *h_X1, uint32_t *h_X0, int n, int m)
{
    int wpr = (m + 31) / 32;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < wpr; j++)
        {
            uint32_t w1 = 0, w0 = 0;
            int base = j * 32;
            for (int k = 0; k < 32; k++)
            {
                int idx = base + k;
                if (idx < m)
                {
                    uint32_t v = h_X[i * m + idx] & 0x3u;
                    w0 |= (v & 1u) << k;        // bit bajo
                    w1 |= ((v >> 1) & 1u) << k; // bit alto
                }
            }
            h_X1[i * wpr + j] = w1;
            h_X0[i * wpr + j] = w0;
        }
    }
}

#define WORDS_PER_KSTEP 4 // 128 bits de K = 4 uint32

// Kernel multiplicacion con Tensor Cores binarios (b1, XOR). Guarda en W el acumulado ponderado de
// XOR-popcounts (4*X11 + 2*X10 + 2*X01 + X00) en int32; la reconstruccion del Gram se hace en el
// post-proceso (kernel_distance_squared).
__global__ void wmma_xxt_b1(const uint32_t *X1, const uint32_t *X0, int *W, int n, int m)
{
    int warpId = threadIdx.x / 32; // 0..15
    int warpRow = warpId / 4;      // 0..3
    int warpCol = warpId % 4;      // 0..3

    // simetria: salteo bloques bajo la diagonal
    if (blockIdx.x < blockIdx.y)
        return;

    int blockRow = blockIdx.y * 32;
    int blockCol = blockIdx.x * 32;
    int row = blockRow + warpRow * 8;
    int col = blockCol + warpCol * 8;

    int wpr = (m + 31) / 32;

    // 32 filas x 4 uint32 (= 128 bits de K) por tramo, para A y B, en ambos planos
    __shared__ uint32_t As1[32][WORDS_PER_KSTEP];
    __shared__ uint32_t As0[32][WORDS_PER_KSTEP];
    __shared__ uint32_t Bs1[32][WORDS_PER_KSTEP];
    __shared__ uint32_t Bs0[32][WORDS_PER_KSTEP];

    wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::experimental::precision::b1, wmma::row_major> a1_frag, a0_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1, wmma::col_major> b1_frag, b0_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c11, c10, c01, c00;
    wmma::fill_fragment(c11, 0);
    wmma::fill_fragment(c10, 0);
    wmma::fill_fragment(c01, 0);
    wmma::fill_fragment(c00, 0);

    int kSteps = (m + 127) / 128;
    for (int ks = 0; ks < kSteps; ks++)
    {
        // Los 512 hilos cargan 32x4 = 128 palabras por buffer (coalesced).
        for (int idx = threadIdx.x; idx < 32 * WORDS_PER_KSTEP; idx += blockDim.x)
        {
            int r = idx / WORDS_PER_KSTEP;
            int w = idx % WORDS_PER_KSTEP;
            int gword = ks * WORDS_PER_KSTEP + w;

            uint32_t va1 = 0, va0 = 0, vb1 = 0, vb0 = 0;
            if (gword < wpr)
            {
                if (blockRow + r < n)
                {
                    va1 = X1[(blockRow + r) * wpr + gword];
                    va0 = X0[(blockRow + r) * wpr + gword];
                }
                if (blockCol + r < n)
                {
                    vb1 = X1[(blockCol + r) * wpr + gword];
                    vb0 = X0[(blockCol + r) * wpr + gword];
                }
            }
            As1[r][w] = va1;
            As0[r][w] = va0;
            Bs1[r][w] = vb1;
            Bs0[r][w] = vb0;
        }
        __syncthreads();

        // ldm = 128 bits (= 4 uint32), multiplo de 128 requerido por WMMA b1
        wmma::load_matrix_sync(a1_frag, &As1[warpRow * 8][0], 128);
        wmma::load_matrix_sync(a0_frag, &As0[warpRow * 8][0], 128);
        wmma::load_matrix_sync(b1_frag, &Bs1[warpCol * 8][0], 128);
        wmma::load_matrix_sync(b0_frag, &Bs0[warpCol * 8][0], 128);

        // 4 matmuls binarios XOR + popcount
        wmma::bmma_sync(c11, a1_frag, b1_frag, c11,
                        wmma::experimental::bmmaBitOpXOR, wmma::experimental::bmmaAccumulateOpPOPC);
        wmma::bmma_sync(c10, a1_frag, b0_frag, c10,
                        wmma::experimental::bmmaBitOpXOR, wmma::experimental::bmmaAccumulateOpPOPC);
        wmma::bmma_sync(c01, a0_frag, b1_frag, c01,
                        wmma::experimental::bmmaBitOpXOR, wmma::experimental::bmmaAccumulateOpPOPC);
        wmma::bmma_sync(c00, a0_frag, b0_frag, c00,
                        wmma::experimental::bmmaBitOpXOR, wmma::experimental::bmmaAccumulateOpPOPC);

        __syncthreads();
    }

    // W = 4*X11 + 2*X10 + 2*X01 + X00 (mismos indices de elemento por shape identico)
    for (int i = 0; i < c00.num_elements; i++)
    {
        c00.x[i] = 4 * c11.x[i] + 2 * c10.x[i] + 2 * c01.x[i] + c00.x[i];
    }

    if (row < n && col < n)
    {
        wmma::store_matrix_sync(W + row * n + col, c00, n, wmma::mem_row_major);
    }
}

// Kernel distancia al cuadrado: reconstruye el Gram desde W (corrige XOR->AND) y arma D2 = N + N^T - 2*XXt.
//   2*G_ij = 6*(S1i+S1j) + 3*(S0i+S0j) - W_ij   (entero, par -> /2 exacto)
//   ||xi||^2 = 4*S1i + S0i
// W solo tiene el triangulo superior; para el inferior se accede espejado (W es simetrica).
__global__ void kernel_distance_squared(const int *S1, const int *S0, const int *W, float *D2, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n)
        return;

    int w_val = (row <= col) ? W[row * n + col] : W[col * n + row];

    int s1i = S1[row], s1j = S1[col];
    int s0i = S0[row], s0j = S0[col];

    int twoG = 6 * (s1i + s1j) + 3 * (s0i + s0j) - w_val; // = 2 * <xi,xj>
    int G = twoG >> 1;                                    // exacto (twoG es par)

    float normi = 4.0f * s1i + s0i;
    float normj = 4.0f * s1j + s0j;

    D2[row * n + col] = normi + normj - 2.0f * (float)G;
}

__global__ void kernel_sqrt(float *D2, float *D, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float x = D2[row * n + col];
        if (x < 0.0f)
            x = 0.0f;
        D[row * n + col] = sqrtf(x);
    }
}

// Conversion uint8 (SNPs originales 0/1/2) -> float, para la referencia cuBLAS (FP32) del modo verify
__global__ void kernel_u8_to_float(const uint8_t *X, float *Xf, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
        Xf[idx] = (float)X[idx];
}

// Post-proceso de la referencia: C (Gram) en float (salida de cublasSsyrk); normas desde S1,S0.
__global__ void kernel_distance_squared_ref(const int *S1, const int *S0, const float *C, float *D2, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n)
        return;
    float c_val = (row <= col) ? C[row * n + col] : C[col * n + row];
    float normi = 4.0f * S1[row] + S0[row];
    float normj = 4.0f * S1[col] + S0[col];
    D2[row * n + col] = normi + normj - 2.0f * c_val;
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
        h_X[i] = rand() % 3;

    float *h_D = (float *)malloc(n * n * sizeof(float));

    int wpr = (m + 31) / 32; // palabras por fila por plano

    /* BIT PACKING EN CPU a dos planos (conversion de entrada, una sola vez, NO se mide) */
    uint32_t *h_X1 = (uint32_t *)malloc(n * wpr * sizeof(uint32_t));
    uint32_t *h_X0 = (uint32_t *)malloc(n * wpr * sizeof(uint32_t));
    auto packing_inicio = std::chrono::high_resolution_clock::now();
    bit_packing_planes(h_X, h_X1, h_X0, n, m);
    auto packing_fin = std::chrono::high_resolution_clock::now();
    double tiempo_packing = std::chrono::duration<double, std::milli>(packing_fin - packing_inicio).count();
    printf("Bit packing CPU (b1, 2 planos): %.3f ms\n", tiempo_packing);

    /* Memoria en device */
    uint32_t *d_X1, *d_X0; // planos empaquetados (1 bit/SNP c/u)
    int *d_S1, *d_S0;      // popcounts por fila de cada plano
    int *d_W;              // acumulado ponderado de XOR-popcounts
    float *d_D2;
    float *d_D;

    CUDA_CHK(cudaMalloc(&d_X1, n * wpr * sizeof(uint32_t)));
    CUDA_CHK(cudaMalloc(&d_X0, n * wpr * sizeof(uint32_t)));
    CUDA_CHK(cudaMalloc(&d_S1, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_S0, n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_W, n * n * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_D2, n * n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_D, n * n * sizeof(float)));

    /* Grillas y bloques */
    dim3 blockNorms(256);
    dim3 gridNorms((n + blockNorms.x - 1) / blockNorms.x);

    dim3 block2D(32, 32);
    dim3 grid2D((n + 31) / 32, (n + 31) / 32);

    dim3 blockWMMA(512); // 16 warps -> tile de salida 32x32 (16 tiles de 8x8)
    dim3 gridWMMA((n + 31) / 32, (n + 31) / 32);

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

    /* Warm-up (no medido) */
    cudaMemcpy(d_X1, h_X1, n * wpr * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X0, h_X0, n * wpr * sizeof(uint32_t), cudaMemcpyHostToDevice);
    kernel_plane_popcounts<<<gridNorms, blockNorms>>>(d_X1, d_X0, d_S1, d_S0, n, m);
    wmma_xxt_b1<<<gridWMMA, blockWMMA>>>(d_X1, d_X0, d_W, n, m);
    kernel_distance_squared<<<grid2D, block2D>>>(d_S1, d_S0, d_W, d_D2, n);
    kernel_sqrt<<<grid2D, block2D>>>(d_D2, d_D, n);
    cudaMemcpy(h_D, d_D, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    /* Medicion: H->D (planos) + norma + wmma + distance + sqrt + D->H.
       El packing NO esta aca (se hizo en CPU una vez). */
    const int REPETICIONES = 10;
    float tHtoD = 0, tNormas = 0, tMult = 0, tDist = 0, tSqrt = 0, tDtoH = 0;
    cudaEventRecord(start);
    for (int rep = 0; rep < REPETICIONES; rep++)
    {
        cudaEventRecord(ev[0]);
        cudaMemcpy(d_X1, h_X1, n * wpr * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_X0, h_X0, n * wpr * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaEventRecord(ev[1]);
        kernel_plane_popcounts<<<gridNorms, blockNorms>>>(d_X1, d_X0, d_S1, d_S0, n, m);

        cudaEventRecord(ev[2]);
        wmma_xxt_b1<<<gridWMMA, blockWMMA>>>(d_X1, d_X0, d_W, n, m);

        cudaEventRecord(ev[3]);
        kernel_distance_squared<<<grid2D, block2D>>>(d_S1, d_S0, d_W, d_D2, n);

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
    printf("  popcounts       : %.3f ms\n", tNormas);
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

    /* Verificacion opcional contra cuBLAS (cublasSsyrk, FP32) sobre datos SIN empaquetar */
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
        kernel_u8_to_float<<<gridConvert, blockConvert>>>(d_Xorig, d_Xf, total);

        // G = X X^T con cuBLAS. FILL_MODE_LOWER (col-major) == triangulo superior row-major.
        const float uno = 1.0f, cero = 0.0f;
        cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                    n, m, &uno, d_Xf, m, &cero, d_G, n);

        kernel_distance_squared_ref<<<grid2D, block2D>>>(d_S1, d_S0, d_G, d_D2, n);
        kernel_sqrt<<<grid2D, block2D>>>(d_D2, d_Dref, n);
        CUDA_CHK(cudaGetLastError());

        // Refresco d_D con el resultado de esta version (el loop dejo d_D2 pisado)
        cudaMemcpy(d_X1, h_X1, n * wpr * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_X0, h_X0, n * wpr * sizeof(uint32_t), cudaMemcpyHostToDevice);
        kernel_plane_popcounts<<<gridNorms, blockNorms>>>(d_X1, d_X0, d_S1, d_S0, n, m);
        wmma_xxt_b1<<<gridWMMA, blockWMMA>>>(d_X1, d_X0, d_W, n, m);
        kernel_distance_squared<<<grid2D, block2D>>>(d_S1, d_S0, d_W, d_D2, n);
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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < 7; i++)
        cudaEventDestroy(ev[i]);

    cudaFree(d_X1);
    cudaFree(d_X0);
    cudaFree(d_S1);
    cudaFree(d_S0);
    cudaFree(d_W);
    cudaFree(d_D2);
    cudaFree(d_D);

    free(h_X1);
    free(h_X0);
    free(h_X);
    free(h_D);

    return 0;
}
