#include "gemm.h"

#include <cuda/cmath>

namespace {

template <int BM, int BN, int BK>
__global__ void gemm_impl_v3(const DeviceMatrix a, const DeviceMatrix b,
                             DeviceMatrix d) {
    __shared__ __half a_smem[BM][BK];
    __shared__ __half b_smem[BK][BN];

    const int col = blockIdx.x * BM + threadIdx.x;
    const int row = blockIdx.y * BN + threadIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    if (row < d.rows && col < d.cols) {
        *(d.data + row * d.stride + col) = 0;
    }

    __half partial_acc{};

    for (int k = 0; k < a.cols; k += BK) {
        if (row < a.rows && (tx + k) < a.cols) {
            a_smem[ty][tx] = *(a.data + row * a.stride + (tx + k));
        } else {
            a_smem[ty][tx] = 0.f;
        }

        if ((ty + k) < b.rows && col < b.cols) {
            b_smem[ty][tx] = *(b.data + (ty + k) * b.stride + col);
        } else {
            b_smem[ty][tx] = 0.f;
        }

        __syncthreads();

        for (int kk = 0; kk < BK; kk++) {
            partial_acc += a_smem[ty][kk] * b_smem[kk][tx];
        }

        __syncthreads();
    }

    if (row < d.rows && col < d.cols) {
        *(d.data + row * d.stride + col) += partial_acc;
    }
}

} // namespace

void GEMM_v3(const DeviceMatrix &a, const DeviceMatrix &b, DeviceMatrix &d) {
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    constexpr int TILE_K = 16;
    dim3 threads(TILE_M, TILE_N);
    dim3 blocks(cuda::ceil_div(d.cols, threads.x),
                cuda::ceil_div(d.rows, threads.y));
    gemm_impl_v3<TILE_M, TILE_N, TILE_K><<<blocks, threads>>>(a, b, d);
    cudaDeviceSynchronize();
}
