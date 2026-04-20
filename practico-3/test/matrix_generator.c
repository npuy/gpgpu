#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Configura el tamaño de la matriz y el rango de valores aquí
#define N 2048      
#define MAX_VALUE 100

int main(void) {
    FILE *file = fopen("matrix.txt", "w");
    if (file == NULL) {
        perror("Error al abrir matrix.txt");
        return EXIT_FAILURE;
    }

    srand((unsigned int)time(NULL));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int value = i;
            fprintf(file, "%d", value);
            if (j < N - 1) {
                fprintf(file, " ");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Matriz %dx%d generada en matrix.txt\n", N, N);
    return EXIT_SUCCESS;
}
