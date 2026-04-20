#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "utils.h"

#define ARRAY_SIZE (100 * 1024 * 1024) //100mb para que no entre en cache 
#define ITERATIONS 15

void shuffle_indices(int *indices, int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

void sequential_access(char *array, int *indices, int size, double *times) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double start = get_time();
        
        for (int i = 0; i < size; i++) {
            int idx = indices[i];
            array[idx]++;
        }
        
        double end = get_time();
        times[iter] = end - start;
    }
}

void random_access(char *array, int *indices, int size, double *times) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double start = get_time();
        
        for (int i = 0; i < size; i++) {
            int idx = indices[i];
            array[idx]++;
        }
        
        double end = get_time();
        times[iter] = end - start;
    }
}

void stride_access(char *array, int size, int stride, double *times) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        double start = get_time();
        
        for (long long i = 0; i < size; i++) { //stride = 1 es secuencial, stride = 64 es saltando 64 posiciones 0 64 128 192...
            long long idx = (i * stride) % size;
            array[idx]++;
        }
        
        double end = get_time();
        times[iter] = end - start;
    }
}

int main() {
    printf("=== Ejercicio 1: Localidad Espacial ===\n");
    printf("Tamaño del arreglo: %d MB\n", ARRAY_SIZE / (1024 * 1024));
    printf("Iteraciones: %d\n\n", ITERATIONS);
    
    char *array = (char *)malloc(ARRAY_SIZE * sizeof(char));
    if (!array) {
        fprintf(stderr, "Error al reservar memoria\n");
        return 1;
    }
    
    memset(array, 0, ARRAY_SIZE);
    
    int *seq_indices = (int *)malloc(ARRAY_SIZE * sizeof(int));
    int *rand_indices = (int *)malloc(ARRAY_SIZE * sizeof(int));
    
    if (!seq_indices || !rand_indices) {
        fprintf(stderr, "Error al reservar memoria para índices\n");
        free(array);
        return 1;
    }
    
    for (int i = 0; i < ARRAY_SIZE; i++) {
        seq_indices[i] = i;
        rand_indices[i] = i;
    }
    
    srand(time(NULL));
    shuffle_indices(rand_indices, ARRAY_SIZE);
    
    double seq_times[ITERATIONS];
    double rand_times[ITERATIONS];
    
    printf("Ejecutando acceso secuencial...\n");
    sequential_access(array, seq_indices, ARRAY_SIZE, seq_times);
    
    printf("\nTiempos por iteración (Secuencial):\n");
    printf("Iter\tTiempo(s)\tEstado Cache\n");
    printf("----\t---------\t------------\n");
    for (int i = 0; i < ITERATIONS; i++) {
        const char *estado = (i == 0) ? "FRÍA" : "CALIENTE";
        printf("%d\t%.4f\t\t%s\n", i + 1, seq_times[i], estado);
    }
    
    memset(array, 0, ARRAY_SIZE);
    
    printf("\nEjecutando acceso aleatorio...\n");
    random_access(array, rand_indices, ARRAY_SIZE, rand_times);
    
    printf("\nTiempos por iteración (Aleatorio):\n");
    printf("Iter\tTiempo(s)\tEstado Cache\n");
    printf("----\t---------\t------------\n");
    for (int i = 0; i < ITERATIONS; i++) {
        const char *estado = (i == 0) ? "FRÍA" : "CALIENTE";
        printf("%d\t%.4f\t\t%s\n", i + 1, rand_times[i], estado);
    }
    
    printf("\n=== Pruebas con diferentes STRIDES ===\n");
    printf("Analizando el impacto del tamaño de salto en el rendimiento\n\n");
    
    int strides[] = {1, 16, 64, 128, 256, 512, 1024, 4096};
    int num_strides = sizeof(strides) / sizeof(strides[0]);
    double stride_times[8][ITERATIONS];
    
    for (int s = 0; s < num_strides; s++) {
        int stride = strides[s];
        memset(array, 0, ARRAY_SIZE);
        
        printf("Ejecutando acceso con stride=%d bytes...\n", stride);
        stride_access(array, ARRAY_SIZE, stride, stride_times[s]);
        
        printf("Tiempos por iteración (Stride=%d):\n", stride);
        printf("Iter\tTiempo(s)\tEstado Cache\n");
        printf("----\t---------\t------------\n");
        for (int i = 0; i < ITERATIONS; i++) {
            const char *estado = (i == 0) ? "FRÍA" : "CALIENTE";
            printf("%d\t%.4f\t\t%s\n", i + 1, stride_times[s][i], estado);
        }
        printf("\n");
    }
    
    double seq_warm = 0, rand_warm = 0;
    
    for (int i = 0; i < ITERATIONS; i++) {
        if (i > 0) {
            seq_warm += seq_times[i];
            rand_warm += rand_times[i];
        }
    }
    
    double seq_avg_warm = seq_warm / (ITERATIONS - 1);
    double rand_avg_warm = rand_warm / (ITERATIONS - 1);
    
    printf("\n=== Resumen de Resultados ===\n");
    printf("\nCACHÉ FRÍA (Iteración 1):\n");
    printf("  Secuencial: %.4f s\n", seq_times[0]);
    printf("  Aleatorio:  %.4f s\n", rand_times[0]);
    printf("  Slowdown:   %.2fx\n", rand_times[0] / seq_times[0]);
    
    printf("\nCACHÉ CALIENTE (Promedio iter 2-%d):\n", ITERATIONS);
    printf("  Secuencial: %.4f s\n", seq_avg_warm);
    printf("  Aleatorio:  %.4f s\n", rand_avg_warm);
    printf("  Slowdown:   %.2fx\n", rand_avg_warm / seq_avg_warm);
    
    printf("\nMEJORA POR CALENTAMIENTO:\n");
    printf("  Secuencial: %.2fx más rápido\n", seq_times[0] / seq_avg_warm);
    printf("  Aleatorio:  %.2fx más rápido\n", rand_times[0] / rand_avg_warm);
    
    printf("\n=== Análisis de STRIDES (Caché Caliente) ===\n");
    printf("Stride\tTiempo(s)\tSlowdown vs Seq\tEficiencia Cache\n");
    printf("------\t---------\t---------------\t----------------\n");
    
    for (int s = 0; s < num_strides; s++) {
        double stride_warm = 0;
        for (int i = 1; i < ITERATIONS; i++) {
            stride_warm += stride_times[s][i];
        }
        double stride_avg_warm = stride_warm / (ITERATIONS - 1);
        double slowdown = stride_avg_warm / seq_avg_warm;
        const char *eficiencia;
        
        if (slowdown < 1.5) eficiencia = "EXCELENTE";
        else if (slowdown < 3.0) eficiencia = "BUENA";
        else if (slowdown < 10.0) eficiencia = "REGULAR";
        else eficiencia = "POBRE";
        
        printf("%d\t%.4f\t\t%.2fx\t\t%s\n", strides[s], stride_avg_warm, slowdown, eficiencia);
    }
    
    FILE *fp = fopen("results/ejercicio1_results.csv", "w");
    if (fp) {
        fprintf(fp, "tipo,tiempo_segundos\n");
        fprintf(fp, "secuencial_fria,%.6f\n", seq_times[0]);
        fprintf(fp, "aleatorio_fria,%.6f\n", rand_times[0]);
        fprintf(fp, "secuencial_caliente,%.6f\n", seq_avg_warm);
        fprintf(fp, "aleatorio_caliente,%.6f\n", rand_avg_warm);
        
        for (int s = 0; s < num_strides; s++) {
            double stride_warm = 0;
            for (int i = 1; i < ITERATIONS; i++) {
                stride_warm += stride_times[s][i];
            }
            double stride_avg_warm = stride_warm / (ITERATIONS - 1);
            fprintf(fp, "stride_%d_caliente,%.6f\n", strides[s], stride_avg_warm);
        }
        fclose(fp);
    }
    
    FILE *fp_detail = fopen("results/ejercicio1_detailed.csv", "w");
    if (fp_detail) {
        fprintf(fp_detail, "iteracion,secuencial,aleatorio\n");
        for (int i = 0; i < ITERATIONS; i++) {
            fprintf(fp_detail, "%d,%.6f,%.6f\n", i + 1, seq_times[i], rand_times[i]);
        }
        fclose(fp_detail);
    }
    
    FILE *fp_stride = fopen("results/ejercicio1_strides.csv", "w");
    if (fp_stride) {
        fprintf(fp_stride, "stride,iteracion,tiempo_segundos\n");
        for (int s = 0; s < num_strides; s++) {
            for (int i = 0; i < ITERATIONS; i++) {
                fprintf(fp_stride, "%d,%d,%.6f\n", strides[s], i + 1, stride_times[s][i]);
            }
        }
        fclose(fp_stride);
        printf("\nResultados guardados en:\n");
        printf("  - results/ejercicio1_results.csv\n");
        printf("  - results/ejercicio1_detailed.csv\n");
        printf("  - results/ejercicio1_strides.csv\n");
    }
    
    free(array);
    free(seq_indices);
    free(rand_indices);
    
    return 0;
}
