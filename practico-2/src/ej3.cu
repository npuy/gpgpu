#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define CUDA_CHK(ans) {gpuAssert((ans), __FILE__, __LINE__)}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

void write_file(const char *, int *, int);

__global__ void traspuesta(int *matriz_origen, int *matriz_destino, int n)
{
    int row = threadIdx.y + (blockIdx.y * blockDim.y);
    int col = threadIdx.x + (blockIdx.x * blockDim.x);
    if (col < n && row < n)
    {                                   // Asumiendo matriz cuadrada x ahora
        int id_origen = row * n + col;  // Asi accedo coalesced a origen, siguiendo row-major order
        int id_destino = col * n + row; // Pero esto ya no es coalesced
        matriz_destino[id_destino] = matriz_origen[id_origen];
    }
}

// Dentro de c/bloque 32 x 32 los hilos se organizan asi (x,y):
/*
(0,0) (1,0) (2,0) ... (31,0)   ← warp 0
(0,1) (1,1) (2,1) ... (31,1)   ← warp 1
(0,2) (1,2) (2,2) ... (31,2)   ← warp 2

El acceso al kernel para el primer warp seria
blockIdx.x = 0, blockIdx.y = 0, threadIdx.x = 0, threadIdx.y = 0 -> row = 0, col = 0 -> id_origen = 0, id_destino = 0
blockIdx.x = 0, blockIdx.y = 0, threadIdx.x = 1, threadIdx.y = 0 -> row = 0, col = 1 -> id_origen = 1, id_destino = 5
blockIdx.x = 0, blockIdx.y = 0, threadIdx.x = 2, threadIdx.y = 0 -> row = 0, col = 2 -> id_origen = 2, id_destino = 10
blockIdx.x = 0, blockIdx.y = 0, threadIdx.x = 3, threadIdx.y = 0 -> row = 0, col = 3 -> id_origen = 3, id_destino = 15
blockIdx.x = 0, blockIdx.y = 0, threadIdx.x = 4, threadIdx.y = 0 -> row = 0, col = 4 -> id_origen = 4, id_destino = 20
blockIdx.x = 0, blockIdx.y = 0, threadIdx.x = 5, threadIdx.y = 0 -> row = 0, col = 5 -> id_origen = 5, id_destino = 25 (fuera de rango en mi caso de matriz 5x5, hasta 24)
....

blockIdx.x = 0, blockIdx.y = 0, threadIdx.x = 0, threadIdx.y = 1 -> row = 1, col = 0 -> id_origen = 5, id_destino = 1
blockIdx.x = 0, blockIdx.y = 0, threadIdx.x = 1, threadIdx.y = 1 -> row = 1, col = 1 -> id_origen = 6, id_destino = 6
...

O sea el warp 0 accede a toda la fila 0, luego el warp 1 accede a toda la fila 1
(0,0)
(0,1)
(0,2)
(0,3)
...
(0,31)

Acceso a origen para lectura (id_origen) es coalesced, contiguos por fila.

Pero el acceso a destino para escritura (id_destino) no lo es, porque se accede por columna:
(0,0) -> id_destino = 0
(1,0) -> id_destino = 5
(2,0) -> id_destino = 10
(3,0) -> id_destino = 15
(4,0) -> id_destino = 20
(5,0) -> id_destino = 25 (fuera de rango en mi ej 5x5)

El stride es n (5 en el ejemplo que use), entonces cada hilo accede a una fila diferente en destino, lo que hace que el acceso a memoria no sea coalesced, y por lo tanto el kernel sea mas lento.

Esto es un problema de rendimiento, porque la GPU tiene que hacer varias transacciones de memoria para acceder a la misma fila en destino, lo que hace que el kernel sea mas lento.
Para solucionar esto, se podria usar memoria compartida para almacenar un bloque de la matriz origen, y luego escribir la transpuesta de ese bloque en destino, lo que haria que el acceso a memoria sea coalesced tanto para origen como para destino (TILING)
*/

int main(int argc, char *argv[])
{
    int *h_entrada;
    int *h_traspuesta;

    int *d_entrada;
    int *d_traspuesta;

    unsigned int size;
    int n, bx, by;

    /**
     * ------------------------------------------------------------------------------------
     * Obtener matriz desde un archivo de texto, el formato del archivo es el siguiente:
     * la primera linea contiene el numero n, bx y by (tamaño de bloque en x e y) separados por espacios
     * las siguientes n lineas contienen cada una un renglon de la matriz, con los elementos separados por espacios
     * Ejemplo:
     * 5 32 8
     * 1 0 0 0 0
     * 0 1 0 0 0
     * 0 0 1 0 0
     * 0 0 0 1 0
     * 0 0 0 0 1
     */
    const char *fname;

    if (argc < 2)
    {
        printf("Debe ingresar el nombre del archivo\n");
        exit(1);
    }
    else
    {
        fname = argv[1];
        printf("Archivo ingresado: %s\n", fname);
    }
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "../test/%s", fname);
    FILE *file = fopen(filepath, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error: Could not find %s file \n", filepath);
        return 1;
    }

    fscanf(file, "%d %d %d", &n, &bx, &by);
    size = n * n * sizeof(int);

    // reservar memoria para la matriz
    h_entrada = (int *)malloc(size);

    for (int i = 0; i < n * n; i++)
    {
        fscanf(file, "%d", &h_entrada[i]);
    }
    fclose(file);
    // ------------------------------------------------------------------------------------

    // asigno memoria en host
    h_traspuesta = (int *)malloc(size);

    // asigno memoria en device
    cudaMalloc((void **)&d_entrada, size);
    cudaMalloc((void **)&d_traspuesta, size);

    // paso la data a device, a la memoria recien asignada
    cudaMemcpy(d_entrada, h_entrada, size, cudaMemcpyHostToDevice);

    // tamaño bloques y grilla
    dim3 block_size(bx, by);

    dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y); // Esto es para que alcance a cubrir toda la matriz, aunque el bloque no divida exactamente a n

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    traspuesta<<<grid_size, block_size>>>(d_entrada, d_traspuesta, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_traspuesta, d_traspuesta, size, cudaMemcpyDeviceToHost);

    printf("Kernel execution time: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // escribo la matriz resultante en el archivo de salida en el mismo formato que el de entrada, pero sin la primera linea con los parametros
    write_file("output3.txt", h_traspuesta, n);

    free(h_entrada);
    free(h_traspuesta);

    return 0;
}

void write_file(const char *fname, int *output, int length)
{
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "../test/%s", fname);
    FILE *f = NULL;
    f = fopen(filepath, "w"); // write and binary flags
    if (f == NULL)
    {
        fprintf(stderr, "Error: Could not create %s file \n", filepath);
        exit(1);
    }

    int size = length * length;
    for (int i = 0; i < size; i++)
    {
        fprintf(f, "%d ", output[i]);
        if ((i + 1) % (length) == 0) // salto de linea cada n elementos, donde n es el tamaño de la matriz
        {
            fprintf(f, "\n");
        }
    }

    fclose(f);
}