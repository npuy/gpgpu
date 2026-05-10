#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <cmath>
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

void write_file(const char *path, int *output, int length);
void parse_input(int argc, char *argv[], int *n, unsigned int *size, const char **output_path);
int *generate_matrix(int n);
void print_timing_stats(const std::vector<float> &tiempos);

__global__ void traspuesta(int *matriz_origen, int *matriz_destino, int n)
{
    __shared__ int tile[32][33]; // Con columna dummy

    int row = threadIdx.y + (blockIdx.y * blockDim.y); // esto es lineal, para acceder a la global
    int col = threadIdx.x + (blockIdx.x * blockDim.x);
    if (col < n && row < n)
    { // Asumiendo matriz cuadrada
        int id_origen = row * n + col;
        tile[threadIdx.y][threadIdx.x] = matriz_origen[id_origen];
    }

    __syncthreads();

    int t_row = threadIdx.y + blockIdx.x * blockDim.y;
    int t_col = threadIdx.x + blockIdx.y * blockDim.x;
    if (t_col < n && t_row < n)
    {
        // Aca con id_destino hago la trasposicion, pero con el tile ya cargado en shared memory, asi accedo coalesced a destino
        // Basicamente doy vuelta el tile, y escribo por fila, que es coalesced

        // Entonces, lo accesos NO COALESCED los hago sobre la memoria compartida, mas rapida
        int id_destino = t_row * n + t_col;
        matriz_destino[id_destino] = tile[threadIdx.x][threadIdx.y];
    }
}

int main(int argc, char *argv[])
{
    const char *output_path = NULL;

    int n;
    unsigned int size;

    // variables en host y device
    int *h_entrada;
    int *h_traspuesta;

    int *d_entrada;
    int *d_traspuesta;

    parse_input(argc, argv, &n, &size, &output_path);
    h_entrada = generate_matrix(n);
    h_traspuesta = (int *)malloc(size);

    // asigno memoria en device
    cudaMalloc((void **)&d_entrada, size);
    cudaMalloc((void **)&d_traspuesta, size);

    // paso la data a device, a la memoria recien asignada
    cudaMemcpy(d_entrada, h_entrada, size, cudaMemcpyHostToDevice);

    // tamaño bloques y grilla
    dim3 block_size(32, 32);
    dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y); // Esto es para que alcance a cubrir toda la matriz, aunque el bloque no divida exactamente a n

    // ejecuto el kernel 10 veces para hacer el promedio queme piden
    const int repeticiones = 10;
    std::vector<float> tiempos(repeticiones);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < repeticiones; i++)
    {
        cudaEventRecord(start);
        traspuesta<<<grid_size, block_size>>>(d_entrada, d_traspuesta, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tiempos[i], start, stop);
    }

    print_timing_stats(tiempos);

    cudaMemcpy(h_traspuesta, d_traspuesta, size, cudaMemcpyDeviceToHost);
    write_file(output_path, h_traspuesta, n);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_entrada);
    cudaFree(d_traspuesta);
    free(h_entrada);
    free(h_traspuesta);

    return 0;
}

void parse_input(int argc, char *argv[], int *n, unsigned int *size, const char **output_path)
{
    if (argc < 3)
    {
        fprintf(stderr, "Uso: %s <n> <output_path>\n", argv[0]);
        exit(1);
    }

    *n = atoi(argv[1]);
    *output_path = argv[2];
    *size = (*n) * (*n) * sizeof(int);

    if (*n <= 0)
    {
        fprintf(stderr, "Error: n debe ser un entero positivo\n");
        exit(1);
    }
}

int *generate_matrix(int n)
{
    int *A = (int *)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = i;
        }
    }

    return A;
}

void print_timing_stats(const std::vector<float> &tiempos)
{
    float suma = std::accumulate(tiempos.begin(), tiempos.end(), 0.0f);
    float promedio = suma / tiempos.size();
    float varianza = 0.0;
    for (float t : tiempos)
    {
        varianza += pow(t - promedio, 2);
    }
    float desv_est = sqrt(varianza / tiempos.size());

    printf("Promedio (10 iteraciones): %f ms\n", promedio);
    printf("Desviación estándar: %f ms\n", desv_est);
}

void write_file(const char *path, int *output, int length)
{
    FILE *f = fopen(path, "w");
    if (f == NULL)
    {
        fprintf(stderr, "Error: Could not create %s file\n", path);
        exit(1);
    }

    int size = length * length;
    for (int i = 0; i < size; i++)
    {
        fprintf(f, "%d ", output[i]);
        if ((i + 1) % length == 0)
        {
            fprintf(f, "\n");
        }
    }

    fclose(f);
}
