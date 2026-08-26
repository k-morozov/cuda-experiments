#include <cstdio>
#include <vector>

#include "gemm.h"

int main() {
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
    std::vector<__half> c_half(N * N, __float2half(0.0f));
    for (size_t i = 0; i < N * N; i++) {
        a_half[i] = __float2half(a_host[i]);
        b_half[i] = __float2half(b_host[i]);
    }

    DeviceMatrix a{nullptr, N, N, N};
    DeviceMatrix b{nullptr, N, N, N};
    DeviceMatrix c{nullptr, N, N, N};
    DeviceMatrix d{nullptr, N, N, N};

    cudaMalloc(&a.data, N * N * sizeof(__half));
    cudaMalloc(&b.data, N * N * sizeof(__half));
    cudaMalloc(&c.data, N * N * sizeof(__half));
    cudaMalloc(&d.data, N * N * sizeof(__half));

    cudaMemcpy(a.data, a_half.data(), N * N * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(b.data, b_half.data(), N * N * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(c.data, c_half.data(), N * N * sizeof(__half),
               cudaMemcpyHostToDevice);

    GEMM(a, b, c, d);

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
            printf("[%zu][%zu] result = %f, expected = %f\n", row, col,
                   result, expected[row * N + col]);
        }
    }

    cudaFree(a.data);
    cudaFree(b.data);
    cudaFree(c.data);
    cudaFree(d.data);

    return 0;
}
