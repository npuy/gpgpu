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

/**
 * Configure una grilla de bloques de tama ̃no 32×32. Reserve un espacio en memoria compartida de
 * tama ̃no igual al del bloque y utilice este espacio para evitar los accesos no-coalesced a la memoria
 * global (es decir, realizar los accesos que ser ́ıan no-coalesced sobre este espacio en lugar de la memoria
 * global). Compare el desempe ̃no del kernel con las versiones del pr ́actico anterior
 */
__global__ void traspuesta(int *matriz_origen, int *matriz_destino, int n)
{
    __shared__ int tile[32][32];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < n)
    {
        tile[threadIdx.y][threadIdx.x] = matriz_origen[y * n + x];
    }
    __syncthreads();

    x = blockIdx.y * blockDim.y + threadIdx.x; // intercambiamos x e y para escribir la transpuesta
    y = blockIdx.x * blockDim.x + threadIdx.y;
    if (x < n && y < n)
    {
        matriz_destino[y * n + x] = tile[threadIdx.x][threadIdx.y]; // escribimos la transpuesta desde el tile
    }
}

__global__ void traspuesta_padding(int *matriz_origen, int *matriz_destino, int n)
{
    __shared__ int tile[32][33];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < n)
    {
        tile[threadIdx.y][threadIdx.x] = matriz_origen[y * n + x];
    }
    __syncthreads();

    x = blockIdx.y * blockDim.y + threadIdx.x; // intercambiamos x e y para escribir la transpuesta
    y = blockIdx.x * blockDim.x + threadIdx.y;
    if (x < n && y < n)
    {
        matriz_destino[y * n + x] = tile[threadIdx.x][threadIdx.y]; // escribimos la transpuesta desde el tile
    }
}

int main(int argc, char *argv[])
{
    int *h_entrada;
    int *h_traspuesta;

    int *d_entrada;
    int *d_traspuesta;
    int *d_transpuesta_padding;

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
    cudaMalloc((void **)&d_transpuesta_padding, size);

    // paso la data a device, a la memoria recien asignada
    cudaMemcpy(d_entrada, h_entrada, size, cudaMemcpyHostToDevice);

    // tamaño bloques y grilla
    dim3 block_size(bx, by);

    dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y); // Esto es para que alcance a cubrir toda la matriz, aunque el bloque no divida exactamente a n

    for (int i = 0; i < 10; i++)
        traspuesta<<<grid_size, block_size>>>(d_entrada, d_traspuesta, n);
    for (int i = 0; i < 10; i++)
        traspuesta_padding<<<grid_size, block_size>>>(d_entrada, d_transpuesta_padding, n);

    cudaMemcpy(h_traspuesta, d_traspuesta, size, cudaMemcpyDeviceToHost);
    // escribo la matriz resultante en el archivo de salida en el mismo formato que el de entrada, pero sin la primera linea con los parametros
    write_file("output1.txt", h_traspuesta, n);

    cudaMemcpy(h_traspuesta, d_transpuesta_padding, size, cudaMemcpyDeviceToHost);
    write_file("output1_padding.txt", h_traspuesta, n);

    cudaFree(d_entrada);
    cudaFree(d_traspuesta);
    cudaFree(d_transpuesta_padding);

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