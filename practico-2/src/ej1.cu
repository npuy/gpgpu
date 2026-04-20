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

void read_file(const char *, int *);
void write_file(const char *, int *, int);
int get_text_length(const char *fname);

#define A 15
#define B 27
#define M 256
#define A_MMI_M -17

__device__ int modulo(int a, int b)
{
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void decrypt_kernel(int *d_message, int length)
{
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i < length)
	{															// esto para que no se pase del largo del mensaje, porque el bloque de threads puede ser mas grande que el mensaje
		d_message[i] = modulo(A_MMI_M * (d_message[i] - B), M); // Aca habria que usar la funcion modulo que nos dan ellos!
	}
}

int main(int argc, char *argv[])
{
	int *h_message;
	int *d_message;
	unsigned int size;

	const char *fname;

	if (argc < 2)
		printf("Debe ingresar el nombre del archivo\n");
	else
		fname = argv[1];

	int length = get_text_length(fname);

	size = length * sizeof(int); // Me dan esto que es el size del texto !!

	// reservar memoria para el mensaje
	h_message = (int *)malloc(size);

	// leo el archivo de la entrada
	read_file(fname, h_message);

	// 2. Reservar memoria en la GPU para el texto a desencriptar.
	// d_message puntero DESTINO en device, me lo dan declarado los profs
	cudaMalloc((void **)&d_message, size);

	// 3. Copiar el texto a la memoria de la GPU.
	cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice);

	// 4.Configurar la grilla de threads con un tama˜no de bloque fijo (a elecci´on)
	// y suficientes hilos para cubrir todo el texto.
	// Luego invocar el kernel con dicha configuraci´on.

	// Creo que puede ser de 1 dimension.
	// necesito tantos threads como cantidad de caracteres del texto (length)
	int blockSize = 256;
	int numBlocks = (length + blockSize - 1) / blockSize;

	dim3 grid_size(numBlocks);
	dim3 block_size(blockSize);

	decrypt_kernel<<<grid_size, block_size>>>(d_message, length);

	// sincronizar para esperar a que terminen los threads
	cudaDeviceSynchronize();

	cudaMemcpy(h_message, d_message, size, cudaMemcpyDeviceToHost);

	// guardo el mensaje desencriptado en un archivo de salida
	write_file("output1.txt", h_message, length);

	// libero la memoria en la CPU
	free(h_message);

	return 0;
}

int get_text_length(const char *fname)
{
	char filepath[512];
	snprintf(filepath, sizeof(filepath), "../test/%s", fname);
	FILE *f = NULL;
	f = fopen(filepath, "r"); // read and binary flags

	size_t pos = ftell(f);
	fseek(f, 0, SEEK_END);
	size_t length = ftell(f);
	fseek(f, pos, SEEK_SET);

	fclose(f);

	return length;
}

void read_file(const char *fname, int *input)
{
	char filepath[512];
	snprintf(filepath, sizeof(filepath), "../test/%s", fname);
	FILE *f = NULL;
	f = fopen(filepath, "r"); // read and binary flags
	if (f == NULL)
	{
		fprintf(stderr, "Error: Could not find %s file \n", filepath);
		exit(1);
	}

	int c;
	while ((c = getc(f)) != EOF)
	{
		*(input++) = c;
	}

	fclose(f);
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
		putc(output[i], f);
	}

	fclose(f);
}
