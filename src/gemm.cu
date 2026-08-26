#include "gemm.h"

#include <cstdio>

namespace {

__global__ void gemm_impl(const DeviceMatrix &a, const DeviceMatrix &b,
                          const DeviceMatrix &c, DeviceMatrix &d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= a.rows || col >= b.cols) {
        return;
    }

    // int stride = gridDim.x * blockIdx.x;

    __half result{};

    for (int i = 0; i < a.cols; i++) {
        result +=
            *(a.data + row * a.stride + i) * *(b.data + i * b.stride + col);
    }

    *(d.data + row * d.stride + col) = result;
}

} // namespace

void GEMM(const DeviceMatrix &a, const DeviceMatrix &b, const DeviceMatrix &c,
          DeviceMatrix &d) {
    dim3 threads(16, 16);
    dim3 blocks((d.cols + threads.x - 1) / threads.x,
                (d.rows + threads.y - 1) / threads.y);
    gemm_impl<<<blocks, threads>>>(a, b, c, d);
    cudaDeviceSynchronize();
}
