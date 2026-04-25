#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Configura el tamaño de la matriz y el rango de valores aqui
#define N 2048
#define MIN_VALUE -10
#define MAX_VALUE 10

int main(void) {
    FILE *file = fopen("matrix.txt", "w");
    if (file == NULL) {
        perror("Error al abrir matrix.txt");
        return EXIT_FAILURE;
    }

    srand((unsigned int)time(NULL));

    for (int i = 0; i < N * N; i++) {
        int value = MIN_VALUE + rand() % (MAX_VALUE - MIN_VALUE + 1);
        fprintf(file, "%d", value);
        if (i < N * N - 1) {
            fprintf(file, " ");
        }
    }
    fprintf(file, "\n");

    fclose(file);
    printf("Array de %d valores generado en matrix.txt\n", N * N);
    return EXIT_SUCCESS;
}
