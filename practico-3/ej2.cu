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
 * Partiremos de un arreglo de enteros, de tama ̃no m ́ultiplo de 32, en memoria global, en el que algunos
 * son positivos y otros negativos. Se trabajar ́a con una grilla unidimensional y se mapear ́a cada hilo a un
 * elemento del arreglo. Para cada segmento de 32 enteros consecutivos se debe hacer lo siguiente: Los elementos
 * negativos del segmento deben ser reemplazados por la suma de estos m ́as el m ́aximo de los elementos positivos
 * del segmento.
 */
__global__ void func(int *array, int n)
{
    __shared__ int tile[32];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        tile[threadIdx.x] = array[idx];
    }

    __syncthreads();

    int sum_negatives = 0;
    int max_positive = 0;
    for (int i = 0; i < blockDim.x; i++)
    {
        if (tile[i] < 0)
            sum_negatives += tile[i];
        else if (tile[i] > max_positive)
            max_positive = tile[i];
    }
    if (idx < n && tile[threadIdx.x] < 0)
    {
        array[idx] = sum_negatives + max_positive;
    }
}

__global__ void reduccion_suma_neg_prim(int *array, int n)
{
    int tid = threadIdx.x; // la "lane" de la que hablan en las diapos teorico
    int id_global = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (id_global < n)
    {
        int valor = array[id_global];
        int suma_neg = (valor < 0) ? valor : 0;
        int max_pos = (valor > 0) ? valor : 0; // asumo hay positivos, por letra

        for (int stride = 16; stride > 0; stride /= 2)
        {
            suma_neg += __shfl_down_sync(0xFFFFFFFF, suma_neg, stride);
            max_pos = max(max_pos, __shfl_down_sync(0xFFFFFFFF, max_pos, stride));
        }

        // lane 0 tiene el resultad (o sea el hilo 0)
        int resultado = suma_neg + max_pos;
        resultado = __shfl_sync(0xFFFFFFFF, resultado, 0);

        // uso el ballot sync para ver cual elemento tiene negativo y asi se cuales reemplazar por el resultado de la suma negs + max pos
        unsigned mask_neg = __ballot_sync(0xFFFFFFFF, valor < 0);

        // para el hilo que soy, me fijo si mi bit en el mask_neg esta prendido
        if ((mask_neg >> tid) & 1)
        {
            // corresponde reemplazar mi valor en matriz
            array[id_global] = resultado;
        }
        else
        {
            array[id_global] = valor;
        }
    }
}

__global__ void reduccion_suma_neg(int *array, int n)
{
    __shared__ int tile_negativos[32];
    __shared__ int tile_positivos[32];

    // Cada hilo carga un elemento en mem compartida
    int tid = threadIdx.x;
    int id_global = (blockIdx.x * blockDim.x) + threadIdx.x;
    int valor = 0;
    if (id_global < n)
    {
        valor = array[id_global];
        int neg = (valor < 0) ? valor : 0;
        int pos = (valor > 0) ? valor : 0; // Aca asumo que siempre hay un positivo, sino el maximo podria ser un negativo
        tile_negativos[tid] = neg;
        tile_positivos[tid] = pos;
    }
    else
    {
        tile_negativos[tid] = 0;
        tile_positivos[tid] = 0;
    }

    // Espero todos los hilos hayan escrito en shared
    __syncthreads();

    // Reduccion en memoria compartida
    /*
    for(int iter = 1; iter < 32; iter *= 2) {
        int indice = 2 * iter * tid;
        if (indice + iter < blockDim.x) {
            tile_negativos[indice] += tile_negativos[indice + iter];
            tile_positivos[indice] = max(tile_positivos[indice], tile_positivos[indice + iter]);
        }
        __syncthreads();
    }
    */
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            tile_negativos[tid] += tile_negativos[tid + stride];
            tile_positivos[tid] = max(tile_positivos[tid], tile_positivos[tid + stride]);
        }
        __syncthreads();
    }

    __syncthreads();

    if (valor < 0)
    {
        array[id_global] = tile_negativos[0] + tile_positivos[0]; // suma paso 1 y paso 2
    }
    else
    {
        array[id_global] = valor;
    }
}

int main(int argc, char *argv[])
{
    int *h_array;

    int *d_array;
    int *d_array2;
    int *d_array3;

    int size, n;

    /**
     * ------------------------------------------------------------------------------------
     * Obtener array desde un archivo de texto, el formato del archivo es el siguiente:
     * la primera linea contiene el numero n (tamaño del array) separado por un espacio
     * la segunda linea contiene n enteros separados por espacios
     * Ejemplo:
     * 64
     * 1 -2 3 -4 5 -6 7 -8 9 -10 11 -12 13 -14 15 -16 17 -18 19 -20 21 -22 23 -24 25 -26 27 -28 29 -30 31 ...
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

    fscanf(file, "%d", &n);
    size = n * sizeof(int);

    h_array = (int *)malloc(size);

    for (int i = 0; i < n; i++)
    {
        fscanf(file, "%d", &h_array[i]);
    }
    fclose(file);
    // ------------------------------------------------------------------------------------

    // asigno memoria en device
    cudaMalloc((void **)&d_array, size);
    cudaMalloc((void **)&d_array2, size);
    cudaMalloc((void **)&d_array3, size);

    // paso la data a device, a la memoria recien asignada
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, h_array, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array3, h_array, size, cudaMemcpyHostToDevice);

    // tamaño bloques y grilla
    dim3 block_size(32);

    dim3 grid_size(n / block_size.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        func<<<grid_size, block_size>>>(d_array, n);
    for (int i = 0; i < 10; i++)
        reduccion_suma_neg<<<grid_size, block_size>>>(d_array2, n);
    for (int i = 0; i < 10; i++)
        reduccion_suma_neg_prim<<<grid_size, block_size>>>(d_array3, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);
    // escribo el array resultante en el archivo de salida en el mismo formato que el de entrada, pero sin la primera linea con los parametros
    write_file("output2.txt", h_array, n);

    cudaMemcpy(h_array, d_array2, size, cudaMemcpyDeviceToHost);
    write_file("output_reduccion_suma_neg.txt", h_array, n);
    cudaMemcpy(h_array, d_array3, size, cudaMemcpyDeviceToHost);
    write_file("output_reduccion_suma_neg_prim.txt", h_array, n);

    // libero memoria en device
    cudaFree(d_array);
    cudaFree(d_array2);
    cudaFree(d_array3);

    free(h_array);

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

    for (int i = 0; i < length; i++)
    {
        fprintf(f, "%d ", output[i]);
    }

    fclose(f);
}