#include "gemm_square_demo.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace {

void PrintMatrix(const char *name, const std::vector<float> &m, int n) {
    printf("%s (%dx%d):\n", name, n, n);
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            printf("%4.0f", m[row * n + col]);
        }
        printf("\n");
    }
    printf("\n");
}

} // namespace

void RunSquareDemo(void (*gemm)(const DeviceMatrix &, const DeviceMatrix &,
                                DeviceMatrix &),
                   int n) {
    const size_t elems = static_cast<size_t>(n) * n;

    std::vector<float> a_host(elems);
    std::vector<float> b_host(elems);
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            a_host[row * n + col] = static_cast<float>((row + col) % 4);
            b_host[row * n + col] = static_cast<float>((row + 2 * col) % 3);
        }
    }

    std::vector<__half> a_half(elems);
    std::vector<__half> b_half(elems);
    for (size_t i = 0; i < elems; i++) {
        a_half[i] = __float2half(a_host[i]);
        b_half[i] = __float2half(b_host[i]);
    }

    DeviceMatrix a{nullptr, static_cast<size_t>(n), static_cast<size_t>(n),
                   static_cast<size_t>(n)};
    DeviceMatrix b{nullptr, static_cast<size_t>(n), static_cast<size_t>(n),
                   static_cast<size_t>(n)};
    DeviceMatrix d{nullptr, static_cast<size_t>(n), static_cast<size_t>(n),
                   static_cast<size_t>(n)};

    cudaMalloc(&a.data, elems * sizeof(__half));
    cudaMalloc(&b.data, elems * sizeof(__half));
    cudaMalloc(&d.data, elems * sizeof(__half));

    cudaMemcpy(a.data, a_half.data(), elems * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(b.data, b_half.data(), elems * sizeof(__half),
               cudaMemcpyHostToDevice);

    gemm(a, b, d);

    std::vector<__half> d_half(elems);
    cudaMemcpy(d_half.data(), d.data, elems * sizeof(__half),
               cudaMemcpyDeviceToHost);

    std::vector<float> d_host(elems);
    for (size_t i = 0; i < elems; i++) {
        d_host[i] = __half2float(d_half[i]);
    }

    std::vector<float> expected(elems, 0.0f);
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float acc = 0.0f;
            for (int k = 0; k < n; k++) {
                acc += a_host[row * n + k] * b_host[k * n + col];
            }
            expected[row * n + col] = acc;
        }
    }

    PrintMatrix("A", a_host, n);
    PrintMatrix("B", b_host, n);
    PrintMatrix("A x B (GEMM_v4)", d_host, n);
    PrintMatrix("A x B (reference)", expected, n);

    cudaFree(a.data);
    cudaFree(b.data);
    cudaFree(d.data);
}
