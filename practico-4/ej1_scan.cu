#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

static int g_num_elements = 1024 * 2;
static int g_values_cycle = 8;

constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(err__));                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Este kernel realiza un exclusive scan dentro de cada bloque usando Blelloch scan
// en memoria compartida. Ademas guarda la suma total de cada bloque para una etapa posterior.
__global__ void exclusive_scan_block_kernel(
    const int *input,
    int *output,
    int *block_sums,
    int n
) {
    extern __shared__ int temp[];

    const int tid = threadIdx.x;
    const int block_start = 2 * blockIdx.x * blockDim.x;
    const int i0 = block_start + tid;
    const int i1 = block_start + blockDim.x + tid;

    temp[tid] = (i0 < n) ? input[i0] : 0;
    temp[blockDim.x + tid] = (i1 < n) ? input[i1] : 0;

    for (int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index < 2 * blockDim.x) {
            temp[index] += temp[index - stride];
        }
    }

    int total_sum = 0;
    if (tid == 0) {
        total_sum = temp[2 * blockDim.x - 1];
        temp[2 * blockDim.x - 1] = 0;
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = total_sum;
        }
    }

    for (int stride = blockDim.x; stride > 0; stride >>= 1) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index < 2 * blockDim.x) {
            int left = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += left;
        }
    }

    __syncthreads();

    if (i0 < n) {
        output[i0] = temp[tid];
    }
    if (i1 < n) {
        output[i1] = temp[blockDim.x + tid];
    }
}

// Este kernel suma a cada elemento el offset acumulado de su bloque, obtenido
// al hacer scan sobre el vector de sumas parciales por bloque.
__global__ void add_block_offsets_kernel(int *output, const int *block_offsets, int n) {
    const int block_start = 2 * blockIdx.x * blockDim.x;
    const int i0 = block_start + threadIdx.x;
    const int i1 = block_start + blockDim.x + threadIdx.x;
    const int offset = block_offsets[blockIdx.x];

    if (i0 < n) {
        output[i0] += offset;
    }
    if (i1 < n) {
        output[i1] += offset;
    }
}

// Esta funcion aplica la version manual para un vector de largo arbitrario:
// primero hace scan por bloque, luego scan recursivo de las sumas de bloque,
// y finalmente propaga esos offsets al resultado final.
void exclusive_scan_manual_recursive(const int *d_input, int *d_output, int n) {
    const int elements_per_block = 2 * BLOCK_SIZE;
    const int num_blocks = (n + elements_per_block - 1) / elements_per_block;

    int *d_block_sums = nullptr;
    if (num_blocks > 1) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_block_sums), num_blocks * sizeof(int)));
    }

    exclusive_scan_block_kernel<<<num_blocks, BLOCK_SIZE, elements_per_block * sizeof(int)>>>(
        d_input, d_output, d_block_sums, n
    );
    CUDA_CHECK(cudaGetLastError());

    if (num_blocks > 1) {
        int *d_block_offsets = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_block_offsets), num_blocks * sizeof(int)));

        exclusive_scan_manual_recursive(d_block_sums, d_block_offsets, num_blocks);

        add_block_offsets_kernel<<<num_blocks, BLOCK_SIZE>>>(d_output, d_block_offsets, n);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_block_offsets));
        CUDA_CHECK(cudaFree(d_block_sums));
    }
}

// Este wrapper expone la solucion manual y evita lanzar trabajo si el vector esta vacio.
void exclusive_scan_manual(const int *d_input, int *d_output, int n) {
    if (n <= 0) {
        return;
    }
    exclusive_scan_manual_recursive(d_input, d_output, n);
}

// Esta funcion resuelve el exclusive scan usando la primitiva DeviceScan::ExclusiveSum de CUB.
void exclusive_scan_cub(const int *d_input, int *d_output, int n) {
    void *temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        temp_storage, temp_storage_bytes, d_input, d_output, n
    ));

    CUDA_CHECK(cudaMalloc(&temp_storage, temp_storage_bytes));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        temp_storage, temp_storage_bytes, d_input, d_output, n
    ));

    CUDA_CHECK(cudaFree(temp_storage));
}

// Esta funcion resuelve el exclusive scan usando la primitiva thrust::exclusive_scan.
void exclusive_scan_thrust(const int *d_input, int *d_output, int n) {
    thrust::device_ptr<const int> input_begin(d_input);
    thrust::device_ptr<const int> input_end(d_input + n);
    thrust::device_ptr<int> output_begin(d_output);
    thrust::exclusive_scan(input_begin, input_end, output_begin);
}

// Esta funcion genera un vector de prueba deterministico con un patron ciclico conocido.
std::vector<int> generate_input(int n) {
    std::vector<int> input(n);
    for (int i = 0; i < n; ++i) {
        input[i] = (i % g_values_cycle) + 1;
    }
    return input;
}

// Esta funcion compara dos resultados para chequear que dos implementaciones GPU
// produzcan exactamente la misma salida.
bool compare_results(
    const std::vector<int> &result_a,
    const std::vector<int> &result_b,
    const char *label_a,
    const char *label_b
) {
    if (result_a.size() != result_b.size()) {
        return false;
    }
    for (size_t i = 0; i < result_a.size(); ++i) {
        if (result_a[i] != result_b[i]) {
            std::fprintf(
                stderr,
                "Mismatch entre %s y %s en posicion %zu: %d vs %d\n",
                label_a,
                label_b,
                i,
                result_a[i],
                result_b[i]
            );
            return false;
        }
    }
    return true;
}

// Esta funcion imprime una muestra corta del vector para inspeccion visual rapida.
void print_sample(const char *label, const std::vector<int> &data, int max_items = 16) {
    const int count = std::min<int>(max_items, static_cast<int>(data.size()));
    std::printf("%s [", label);
    for (int i = 0; i < count; ++i) {
        std::printf("%d", data[i]);
        if (i + 1 < count) {
            std::printf(", ");
        }
    }
    if (count < static_cast<int>(data.size())) {
        std::printf(", ...");
    }
    std::printf("]\n");
}

// En main se genera el arreglo de prueba, se reservan buffers en GPU,
// se ejecutan las tres soluciones pedidas y se copian los resultados a host.
int main(int argc, char **argv) {
    if (argc > 1) {
        g_num_elements = std::atoi(argv[1]);
    }

    if (g_num_elements <= 0) {
        std::fprintf(stderr, "La cantidad de elementos debe ser positiva.\n");
        return EXIT_FAILURE;
    }

    const size_t bytes = static_cast<size_t>(g_num_elements) * sizeof(int);
    std::vector<int> h_input = generate_input(g_num_elements);
    std::vector<int> h_manual(g_num_elements);
    std::vector<int> h_cub(g_num_elements);
    std::vector<int> h_thrust(g_num_elements);

    int *d_input = nullptr;
    int *d_manual = nullptr;
    int *d_cub = nullptr;
    int *d_thrust = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_input), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_manual), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_cub), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_thrust), bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    exclusive_scan_manual(d_input, d_manual, g_num_elements);
    exclusive_scan_cub(d_input, d_cub, g_num_elements);
    exclusive_scan_thrust(d_input, d_thrust, g_num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_manual.data(), d_manual, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cub.data(), d_cub, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_thrust.data(), d_thrust, bytes, cudaMemcpyDeviceToHost));

    const bool manual_vs_cub_ok = compare_results(h_manual, h_cub, "manual", "cub");
    const bool manual_vs_thrust_ok = compare_results(h_manual, h_thrust, "manual", "thrust");

    std::printf("N = %d\n", g_num_elements);
    print_sample("input   =", h_input);
    print_sample("manual  =", h_manual);
    print_sample("cub     =", h_cub);
    print_sample("thrust  =", h_thrust);
    std::printf("manual == cub: %s\n", manual_vs_cub_ok ? "si" : "no");
    std::printf("manual == thrust: %s\n", manual_vs_thrust_ok ? "si" : "no");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_manual));
    CUDA_CHECK(cudaFree(d_cub));
    CUDA_CHECK(cudaFree(d_thrust));

    return (manual_vs_cub_ok && manual_vs_thrust_ok) ? EXIT_SUCCESS : EXIT_FAILURE;
}
