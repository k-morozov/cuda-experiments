#include "gemm.h"

#include <cuda/cmath>

namespace {

__global__ void gemm_impl_v1(const DeviceMatrix &a, const DeviceMatrix &b,
                             DeviceMatrix &d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= a.rows || col >= b.cols) {
        return;
    }

    __half result{};

    for (int i = 0; i < a.cols; i++) {
        result +=
            *(a.data + row * a.stride + i) * *(b.data + i * b.stride + col);
    }

    *(d.data + row * d.stride + col) = result;
}

} // namespace

void GEMM_v1(const DeviceMatrix &a, const DeviceMatrix &b, DeviceMatrix &d) {
    dim3 threads(32, 16);
    dim3 blocks(cuda::ceil_div(d.cols, threads.x),
                cuda::ceil_div(d.rows, threads.y));
    gemm_impl_v1<<<blocks, threads>>>(a, b, d);
    cudaDeviceSynchronize();
}
