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
    write_file("output1.txt", h_traspuesta, n);

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