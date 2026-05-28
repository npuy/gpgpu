#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#define K 10

#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess)                                              \
        {                                                                      \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct bin_functor
{
    __host__ __device__ int operator()(int x) const
    {
        return x / K;
    }
};

void inicializar_vector(std::vector<int> &v)
{
    const int n = static_cast<int>(v.size());

    for (int i = 0; i < n; i++)
    {
        v[i] = i;
    }

    for (int i = 0; i < n; i++)
    {
        int j = (37 * i + 13) % n;
        std::swap(v[i], v[j]);
    }
}

void agrupar_serial(
    const std::vector<int> &input,
    std::vector<int> &output,
    std::vector<int> &bin_counts,
    std::vector<int> &bin_offsets)
{
    const int n = static_cast<int>(input.size());
    int max_bin = 0;

    for (int i = 0; i < n; i++)
    {
        max_bin = std::max(max_bin, input[i] / K);
    }

    const int num_bins = max_bin + 1;
    output.resize(n);
    bin_counts.assign(num_bins, 0);
    bin_offsets.resize(num_bins);

    for (int i = 0; i < n; i++)
    {
        bin_counts[input[i] / K]++;
    }

    bin_offsets[0] = 0;
    for (int b = 1; b < num_bins; b++)
    {
        bin_offsets[b] = bin_offsets[b - 1] + bin_counts[b - 1];
    }

    std::vector<int> pos_actual = bin_offsets;
    for (int i = 0; i < n; i++)
    {
        int bin = input[i] / K;
        int pos = pos_actual[bin]++;
        output[pos] = input[i];
    }
}

void agrupar_thrust(
    const std::vector<int> &h_input,
    std::vector<int> &h_output,
    std::vector<int> &h_bin_counts,
    std::vector<int> &h_bin_offsets,
    float *gpu_ms)
{
    const int n = static_cast<int>(h_input.size());
    if (n == 0)
    {
        h_output.clear();
        h_bin_counts.clear();
        h_bin_offsets.clear();
        *gpu_ms = 0.0f;
        return;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    thrust::device_vector<int> d_input = h_input;
    thrust::device_vector<int> d_bins(n);
    thrust::device_vector<int> d_output = d_input;

    CUDA_CHECK(cudaEventRecord(start));

    // thrust::transform aplica bin_functor a cada elemento de d_input en paralelo.
    // El resultado es d_bins[i] = d_input[i] / K, es decir, la clave de agrupamiento
    // que se usara para ordenar y contar.
    thrust::transform(d_input.begin(), d_input.end(), d_bins.begin(), bin_functor());

    // thrust::max_element busca en paralelo el iterador al mayor bin calculado.
    // Con ese valor se obtiene num_bins = max_bin + 1, igual que en la version serial.
    int max_bin = *thrust::max_element(d_bins.begin(), d_bins.end());
    const int num_bins = max_bin + 1;

    // thrust::stable_sort_by_key ordena las claves d_bins y reordena d_output con
    // los mismos movimientos. Al ser estable, dos elementos con el mismo bin quedan
    // en el mismo orden relativo en el que aparecian en input.
    thrust::stable_sort_by_key(d_bins.begin(), d_bins.end(), d_output.begin());

    thrust::device_vector<int> d_unique_bins(num_bins);
    thrust::device_vector<int> d_counts_tmp(num_bins);
    thrust::device_vector<int> d_bin_counts(num_bins);
    thrust::device_vector<int> d_bin_offsets(num_bins);

    // thrust::fill inicializa todos los conteos en cero. Esto permite representar
    // tambien bins vacios, porque reduce_by_key solo genera entradas para bins que
    // aparecen al menos una vez.
    thrust::fill(d_bin_counts.begin(), d_bin_counts.end(), 0);

    // thrust::make_constant_iterator(1) crea un iterador virtual que devuelve 1
    // para cada posicion sin reservar un vector real de unos en memoria.
    thrust::constant_iterator<int> ones_begin = thrust::make_constant_iterator(1);

    // thrust::reduce_by_key recorre los bins ya ordenados y combina las secuencias
    // contiguas con la misma clave. Sumando los unos del iterador constante produce
    // pares (bin, cantidad) para cada bin presente.
    auto reduced_end = thrust::reduce_by_key(
        d_bins.begin(),
        d_bins.end(),
        ones_begin,
        d_unique_bins.begin(),
        d_counts_tmp.begin());
    int present_bins = static_cast<int>(reduced_end.first - d_unique_bins.begin());

    // thrust::scatter copia cada count temporal a la posicion indicada por su bin.
    // Asi, si d_unique_bins[j] == b, entonces d_bin_counts[b] = d_counts_tmp[j].
    thrust::scatter(
        d_counts_tmp.begin(),
        d_counts_tmp.begin() + present_bins,
        d_unique_bins.begin(),
        d_bin_counts.begin());

    // thrust::exclusive_scan calcula los offsets iniciales de cada bin:
    // d_bin_offsets[0] = 0 y d_bin_offsets[b] = suma de los counts anteriores.
    thrust::exclusive_scan(d_bin_counts.begin(), d_bin_counts.end(), d_bin_offsets.begin());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(gpu_ms, start, stop));

    h_output.resize(n);
    h_bin_counts.resize(num_bins);
    h_bin_offsets.resize(num_bins);

    // thrust::copy transfiere los resultados desde los vectores en device hacia
    // los vectores std::vector del host para poder imprimir, guardar y comparar.
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());
    thrust::copy(d_bin_counts.begin(), d_bin_counts.end(), h_bin_counts.begin());
    thrust::copy(d_bin_offsets.begin(), d_bin_offsets.end(), h_bin_offsets.begin());

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void imprimir_y_guardar_csv(
    const std::vector<int> &input,
    const std::vector<int> &output,
    const std::vector<int> &bin_counts,
    const std::vector<int> &bin_offsets,
    const char *csv_filename)
{
    FILE *f = std::fopen(csv_filename, "w");
    if (f == nullptr)
    {
        std::printf("Error abriendo archivo CSV\n");
        return;
    }

    std::fprintf(f, "input,output,bin_counts,bin_offsets\n");
    std::printf("-------------------------------------------------------------\n");
    std::printf("%10s %10s %15s %15s\n", "input", "output", "bin_counts", "bin_offsets");
    std::printf("-------------------------------------------------------------\n");

    const int n = static_cast<int>(input.size());
    const int num_bins = static_cast<int>(bin_counts.size());
    const int max_rows = std::max(n, num_bins);

    for (int i = 0; i < max_rows; i++)
    {
        if (i < n)
            std::printf("%10d ", input[i]);
        else
            std::printf("%10s ", "");

        if (i < n)
            std::printf("%10d ", output[i]);
        else
            std::printf("%10s ", "");

        if (i < num_bins)
            std::printf("%15d ", bin_counts[i]);
        else
            std::printf("%15s ", "");

        if (i < num_bins)
            std::printf("%15d ", bin_offsets[i]);
        else
            std::printf("%15s ", "");

        std::printf("\n");

        if (i < n)
            std::fprintf(f, "%d,", input[i]);
        else
            std::fprintf(f, ",");

        if (i < n)
            std::fprintf(f, "%d,", output[i]);
        else
            std::fprintf(f, ",");

        if (i < num_bins)
            std::fprintf(f, "%d,", bin_counts[i]);
        else
            std::fprintf(f, ",");

        if (i < num_bins)
            std::fprintf(f, "%d", bin_offsets[i]);
        std::fprintf(f, "\n");
    }

    std::fclose(f);
    std::printf("-------------------------------------------------------------\n");
    std::printf("CSV guardado en: %s\n", csv_filename);
}

bool mismo_vector(const std::vector<int> &a, const std::vector<int> &b, const char *nombre)
{
    if (a.size() != b.size())
    {
        std::fprintf(stderr, "%s tiene tamanos distintos: %zu vs %zu\n", nombre, a.size(), b.size());
        return false;
    }

    for (size_t i = 0; i < a.size(); i++)
    {
        if (a[i] != b[i])
        {
            std::fprintf(stderr, "Diferencia en %s[%zu]: %d vs %d\n", nombre, i, a[i], b[i]);
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv)
{
    int n = 1000;
    if (argc > 1)
    {
        n = std::atoi(argv[1]);
    }
    bool pedir_csv = false;
    if (argc > 2)
    {
        pedir_csv = std::string(argv[2]) == "--csv";
    }

    if (n <= 0)
    {
        std::fprintf(stderr, "N debe ser positivo.\n");
        return EXIT_FAILURE;
    }

    std::vector<int> input(n);
    inicializar_vector(input);

    std::vector<int> output_serial;
    std::vector<int> counts_serial;
    std::vector<int> offsets_serial;

    auto serial_start = std::chrono::high_resolution_clock::now();
    agrupar_serial(input, output_serial, counts_serial, offsets_serial);
    auto serial_stop = std::chrono::high_resolution_clock::now();
    double serial_ms = std::chrono::duration<double, std::milli>(serial_stop - serial_start).count();

    std::vector<int> output_thrust;
    std::vector<int> counts_thrust;
    std::vector<int> offsets_thrust;
    float thrust_ms = 0.0f;
    agrupar_thrust(input, output_thrust, counts_thrust, offsets_thrust, &thrust_ms);

    bool ok = mismo_vector(output_serial, output_thrust, "output") && mismo_vector(counts_serial, counts_thrust, "bin_counts") && mismo_vector(offsets_serial, offsets_thrust, "bin_offsets");

    std::printf("N = %d\n", n);
    std::printf("Tiempo serial CPU: %.4f ms\n", serial_ms);
    std::printf("Tiempo Thrust GPU: %.4f ms\n", thrust_ms);
    std::printf("Resultado: %s\n", ok ? "OK" : "ERROR");

    if (pedir_csv && n == 1000)
    {
        imprimir_y_guardar_csv(input, output_thrust, counts_thrust, offsets_thrust, "resultado_ej3_thrust.csv");
    }

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
