#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"

void init_matrix(float *M, int N) {
    for (int i = 0; i < N * N; i++) {
        M[i] = (float)(rand() % 100) / 10.0f;
    }
}

void matrix_mult_blocked(float *A, float *B, float *C, int N, int BS) {
    int i, j, k, ii, jj, kk;
    for (ii = 0; ii < N; ii += BS)
        for (jj = 0; jj < N; jj += BS)
            for (kk = 0; kk < N; kk += BS)
                for (i = ii; i < ii + BS && i < N; i++)
                    for (j = jj; j < jj + BS && j < N; j++)
                        for (k = kk; k < kk + BS && k < N; k++)
                            C[i * N + j] += A[i * N + k] * B[k * N + j];
}

void matrix_mult_ikj(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

void matrix_mult_ijk(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

void matrix_mult_jik(float *A, float *B, float *C, int N) {
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

void matrix_mult_kij(float *A, float *B, float *C, int N) {
    for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

void matrix_mult_jki(float *A, float *B, float *C, int N) {
    for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
            for (int i = 0; i < N; i++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

void matrix_mult_kji(float *A, float *B, float *C, int N) {
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

double benchmark_blocked(float *A, float *B, float *C, int N, int BS) {
    memset(C, 0, N * N * sizeof(float));
    double start = get_time();
    matrix_mult_blocked(A, B, C, N, BS);
    double end = get_time();
    return end - start;
}

double benchmark_variant(void (*func)(float*, float*, float*, int), 
                         float *A, float *B, float *C, int N) {
    memset(C, 0, N * N * sizeof(float));
    double start = get_time();
    func(A, B, C, N);
    double end = get_time();
    return end - start;
}

int test_block_sizes(int N) {
    printf("\n=== Parte 1: Determinación del BS óptimo ===\n");
    printf("Tamaño de matriz: %d x %d\n", N, N);
    
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));
    
    if (!A || !B || !C) {
        fprintf(stderr, "Error al reservar memoria\n");
        return 64;
    }
    
    init_matrix(A, N);
    init_matrix(B, N);
    
    int block_sizes[] = {8, 16, 32, 64, 128, 256};
    int num_bs = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    int optimal_bs = block_sizes[0];
    double best_time = 1e9;
    
    FILE *fp = fopen("results/ejercicio2_block_sizes.csv", "w");
    if (fp) {
        fprintf(fp, "block_size,tiempo_segundos,mflops\n");
    }
    
    printf("\nBS\tTiempo(s)\tMFLOPS\n");
    printf("---\t---------\t------\n");
    
    for (int i = 0; i < num_bs; i++) {
        int BS = block_sizes[i];
        double time = benchmark_blocked(A, B, C, N, BS);
        double mflops = (2.0 * N * N * N) / (time * 1e6);
        
        printf("%d\t%.4f\t\t%.2f", BS, time, mflops);
        
        if (time < best_time) {
            best_time = time;
            optimal_bs = BS;
            printf(" ← ÓPTIMO");
        }
        printf("\n");
        
        if (fp) {
            fprintf(fp, "%d,%.6f,%.2f\n", BS, time, mflops);
        }
    }
    
    if (fp) {
        fclose(fp);
        printf("\nResultados guardados en results/ejercicio2_block_sizes.csv\n");
    }
    
    printf("\n** BS óptimo detectado: %d (%.4f s, %.2f MFLOPS) **\n", 
           optimal_bs, best_time, (2.0 * N * N * N) / (best_time * 1e6));
    
    free(A);
    free(B);
    free(C);
    
    return optimal_bs;
}

void compare_blocked_vs_ikj(int N, int optimal_BS) {
    printf("\n=== Parte 2: Comparación Blocked vs IKJ ===\n");
    printf("Tamaño de matriz: %d x %d\n", N, N);
    printf("BS óptimo: %d\n", optimal_BS);
    
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));
    
    if (!A || !B || !C) {
        fprintf(stderr, "Error al reservar memoria\n");
        return;
    }
    
    init_matrix(A, N);
    init_matrix(B, N);
    
    double time_blocked = benchmark_blocked(A, B, C, N, optimal_BS);
    double mflops_blocked = (2.0 * N * N * N) / (time_blocked * 1e6);
    
    double time_ikj = benchmark_variant(matrix_mult_ikj, A, B, C, N);
    double mflops_ikj = (2.0 * N * N * N) / (time_ikj * 1e6);
    
    printf("\nResultados:\n");
    printf("Blocked (BS=%d): %.4f s, %.2f MFLOPS\n", optimal_BS, time_blocked, mflops_blocked);
    printf("IKJ:             %.4f s, %.2f MFLOPS\n", time_ikj, mflops_ikj);
    printf("Speedup (IKJ/Blocked): %.2fx\n", time_blocked / time_ikj);
    
    FILE *fp = fopen("results/ejercicio2_blocked_vs_ikj.csv", "w");
    if (fp) {
        fprintf(fp, "variante,block_size,tiempo_segundos,mflops\n");
        fprintf(fp, "blocked,%d,%.6f,%.2f\n", optimal_BS, time_blocked, mflops_blocked);
        fprintf(fp, "ikj,0,%.6f,%.2f\n", time_ikj, mflops_ikj);
        fclose(fp);
    }
    
    free(A);
    free(B);
    free(C);
}

void test_loop_orders() {
    printf("\n=== Parte 3: Comparación de órdenes de loop ===\n");
    
    int sizes[] = {256, 260, 512, 550, 1024, 1050};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    const char *variants[] = {"ijk", "jik", "ikj", "kij", "jki", "kji"};
    void (*functions[])(float*, float*, float*, int) = {
        matrix_mult_ijk, matrix_mult_jik, matrix_mult_ikj,
        matrix_mult_kij, matrix_mult_jki, matrix_mult_kji
    };
    int num_variants = 6;
    
    FILE *fp = fopen("results/ejercicio2_loop_orders.csv", "w");
    if (fp) {
        fprintf(fp, "size,variant,tiempo_segundos,mflops\n");
    }
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        printf("\nTamaño de matriz: %d x %d\n", N, N);
        
        float *A = (float *)malloc(N * N * sizeof(float));
        float *B = (float *)malloc(N * N * sizeof(float));
        float *C = (float *)malloc(N * N * sizeof(float));
        
        if (!A || !B || !C) {
            fprintf(stderr, "Error al reservar memoria para N=%d\n", N);
            continue;
        }
        
        init_matrix(A, N);
        init_matrix(B, N);
        
        printf("Variante\tTiempo(s)\tMFLOPS\n");
        printf("--------\t---------\t------\n");
        
        for (int v = 0; v < num_variants; v++) {
            double time = benchmark_variant(functions[v], A, B, C, N);
            double mflops = (2.0 * N * N * N) / (time * 1e6);
            
            printf("%s\t\t%.4f\t\t%.2f\n", variants[v], time, mflops);
            
            if (fp) {
                fprintf(fp, "%d,%s,%.6f,%.2f\n", N, variants[v], time, mflops);
            }
        }
        
        free(A);
        free(B);
        free(C);
    }
    
    if (fp) {
        fclose(fp);
        printf("\nResultados guardados en results/ejercicio2_loop_orders.csv\n");
    }
}

int main() {
    printf("=== Ejercicio 2: Multiplicación de Matrices ===\n");
    
    srand(42);
    
    int optimal_bs = test_block_sizes(1024);
    
    compare_blocked_vs_ikj(1024, optimal_bs);
    
    test_loop_orders();
    
    printf("\n=== Ejercicio 2 completado ===\n");
    printf("\nPara generar gráficas e informe, ejecuta:\n");
    printf("  python3 scripts/generate_ejercicio2_report.py\n");
    
    return 0;
}
