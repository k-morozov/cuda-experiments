#include "gemm_demo.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

void RunSmallCorrectnessDemo(GemmFn gemm) {
    constexpr size_t N = 64;

    std::vector<float> a_host(N * N);
    std::vector<float> b_host(N * N);
    for (size_t row = 0; row < N; row++) {
        for (size_t col = 0; col < N; col++) {
            a_host[row * N + col] = static_cast<float>((row + col) % 5);
            b_host[row * N + col] = static_cast<float>((row * col) % 7);
        }
    }

    std::vector<__half> a_half(N * N);
    std::vector<__half> b_half(N * N);
    for (size_t i = 0; i < N * N; i++) {
        a_half[i] = __float2half(a_host[i]);
        b_half[i] = __float2half(b_host[i]);
    }

    DeviceMatrix a{nullptr, N, N, N};
    DeviceMatrix b{nullptr, N, N, N};
    DeviceMatrix d{nullptr, N, N, N};

    cudaMalloc(&a.data, N * N * sizeof(__half));
    cudaMalloc(&b.data, N * N * sizeof(__half));
    cudaMalloc(&d.data, N * N * sizeof(__half));

    cudaMemcpy(a.data, a_half.data(), N * N * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(b.data, b_half.data(), N * N * sizeof(__half),
               cudaMemcpyHostToDevice);

    gemm(a, b, d);

    std::vector<__half> d_half(N * N);
    cudaMemcpy(d_half.data(), d.data, N * N * sizeof(__half),
               cudaMemcpyDeviceToHost);

    std::vector<float> expected(N * N, 0.0f);
    for (size_t row = 0; row < N; row++) {
        for (size_t col = 0; col < N; col++) {
            float acc = 0.0f;
            for (size_t k = 0; k < N; k++) {
                acc += a_host[row * N + k] * b_host[k * N + col];
            }
            expected[row * N + col] = acc;
        }
    }

    for (size_t row = 0; row < N; row++) {
        for (size_t col = 0; col < N; col++) {
            float result = __half2float(d_half[row * N + col]);
            printf("[%zu][%zu] result = %f, expected = %f\n", row, col, result,
                   expected[row * N + col]);
        }
    }

    cudaFree(a.data);
    cudaFree(b.data);
    cudaFree(d.data);
}

void RunLargeProfilingDemo(GemmFn gemm) {
    constexpr size_t N = 10'000;

    std::vector<__half> a_half(N * N);
    std::vector<__half> b_half(N * N);
    for (size_t i = 0; i < N * N; i++) {
        a_half[i] = __float2half(static_cast<float>(i % 5));
        b_half[i] = __float2half(static_cast<float>(i % 7));
    }

    DeviceMatrix a{nullptr, N, N, N};
    DeviceMatrix b{nullptr, N, N, N};
    DeviceMatrix d{nullptr, N, N, N};

    cudaMalloc(&a.data, N * N * sizeof(__half));
    cudaMalloc(&b.data, N * N * sizeof(__half));
    cudaMalloc(&d.data, N * N * sizeof(__half));

    cudaMemcpy(a.data, a_half.data(), N * N * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(b.data, b_half.data(), N * N * sizeof(__half),
               cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemm(a, b, d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    double gflops = 2.0 * N * N * N / (elapsed_ms * 1e6);

    double bytes_read_a = static_cast<double>(N) * N * N * sizeof(__half);
    double bytes_read_b = static_cast<double>(N) * N * N * sizeof(__half);
    double bytes_write_d = static_cast<double>(N) * N * sizeof(__half);
    double gb_per_s =
        (bytes_read_a + bytes_read_b + bytes_write_d) / (elapsed_ms * 1e6);

    printf("large GEMM %zux%zu: %.3f ms, %.2f GFLOP/s, %.2f GB/s\n", N, N,
           elapsed_ms, gflops, gb_per_s);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(a.data);
    cudaFree(b.data);
    cudaFree(d.data);
}
