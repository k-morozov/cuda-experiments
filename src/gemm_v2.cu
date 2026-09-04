#include "gemm.h"

namespace {

__global__ void gemm_impl_v2(const DeviceMatrix &a, const DeviceMatrix &b,
                             DeviceMatrix &d) {
    __shared__ __half smem[256];

    const int col = blockIdx.x;
    const int row = blockIdx.y;

    if (row >= d.rows || col >= d.cols) {
        return;
    }

    const int stride = blockDim.x;

    __half result{};

    for (int offset = threadIdx.x; offset < a.cols; offset += stride) {
        result += *(a.data + row * a.stride + offset) *
                  *(b.data + offset * b.stride + col);
    }

    smem[threadIdx.x] = result;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            smem[threadIdx.x] += smem[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *(d.data + row * d.stride + col) = smem[0];
    }
}

} // namespace

void GEMM_v2(const DeviceMatrix &a, const DeviceMatrix &b, DeviceMatrix &d) {
    dim3 threads(256);
    dim3 blocks(d.cols, d.rows);
    gemm_impl_v2<<<blocks, threads>>>(a, b, d);
    cudaDeviceSynchronize();
}
