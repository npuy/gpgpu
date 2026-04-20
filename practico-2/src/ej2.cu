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

__global__ void sumar_val_submatriz(int *matriz, int n, int i1, int j1, int i2, int j2, int val)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x); // indice del hilo en el bloque
    if (idx < n * n)
    {
        int i = idx / n;
        int j = idx % n;
        if (i >= i1 && i <= i2 && j >= j1 && j <= j2) // esto para que no se pase de la submatriz
        {
            matriz[i * n + j] += val;
        }
    }
}

__global__ void sumar_val_submatriz_bidimensional(int *matriz, int n, int i1, int j1, int i2, int j2, int val)
{
    int fila = threadIdx.y + (blockIdx.y * blockDim.y);
    int columna = threadIdx.x + (blockIdx.x * blockDim.x);
    if ((columna < n) && (fila < n))
    {
        if (fila >= i1 && fila <= i2 && columna >= j1 && columna <= j2) // esto para que no se pase de la submatriz
        {
            matriz[fila * n + columna] += val;
        }
    }
}

int main(int argc, char *argv[])
{
    int *h_matriz; // puntero a matriz en host
    int *d_matriz; // y puntero a matriz en device
    unsigned int size;

    int n, i1, j1, i2, j2, val, bidimensional;

    /**
     * ------------------------------------------------------------------------------------
     * Obtener matriz desde un archivo de texto, el formato del archivo es el siguiente:
     * la primera linea contiene el numero n, i1, j1, i2, j2, val y bidimensional (0,1) separados por espacios
     * las siguientes n lineas contienen cada una un renglon de la matriz, con los elementos separados por espacios
     * Ejemplo:
     * 5 1 1 3 3 5 0
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

    fscanf(file, "%d %d %d %d %d %d %d", &n, &i1, &j1, &i2, &j2, &val, &bidimensional);
    size = n * n * sizeof(int);

    // reservar memoria para la matriz
    h_matriz = (int *)malloc(size);

    for (int i = 0; i < n * n; i++)
    {
        fscanf(file, "%d", &h_matriz[i]);
    }
    fclose(file);
    // ------------------------------------------------------------------------------------

    // 2. Reservar memoria en la GPU para la matriz
    // d_matriz puntero destino en device
    cudaMalloc((void **)&d_matriz, size);

    // 3. Copiar la matriz a la memoria de la GPU.
    cudaMemcpy(d_matriz, h_matriz, size, cudaMemcpyHostToDevice);

    // 4.Configurar la grilla de threads con un tama˜no de bloque fijo (a elecci´on)
    // y suficientes hilos para cubrir toda la matriz.
    // Luego invocar el kernel con dicha configuraci´on.

    if (bidimensional)
    {
        // Configurar la grilla de threads para el kernel bidimensional
        dim3 block_size(16, 16); // bloque de 16x16 hilos

        dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y); // grilla de tamaño suficiente para cubrir toda la matriz

        sumar_val_submatriz_bidimensional<<<grid_size, block_size>>>(d_matriz, n, i1, j1, i2, j2, val);
    }
    else
    {
        int blockSize = 256;
        int total = n * n;
        int numBlocks = (total + blockSize - 1) / blockSize;

        dim3 grid_size(numBlocks);
        dim3 block_size(blockSize);

        sumar_val_submatriz<<<grid_size, block_size>>>(d_matriz, n, i1, j1, i2, j2, val);
    }

    // sincronizar para esperar a que terminen los threads
    cudaDeviceSynchronize();

    cudaMemcpy(h_matriz, d_matriz, size, cudaMemcpyDeviceToHost);

    // escribo la matriz resultante en el archivo de salida en el mismo formato que el de entrada, pero sin la primera linea con los parametros
    write_file("output2.txt", h_matriz, n);

    // libero la memoria en la CPU
    free(h_matriz);

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