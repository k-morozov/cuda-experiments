#include "gemm.h"

#include <cuda/cmath>

#include <cstdio>

namespace {

__global__ void gemm_impl_v1(const DeviceMatrix &a, const DeviceMatrix &b,
                             DeviceMatrix &d) {
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

template <int BM, int BN, int BK>
__global__ void gemm_impl_v4(const DeviceMatrix a, const DeviceMatrix b,
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

    __half partial_acc[4][4]{};

    for (int k = 0; k < a.cols; k += BK) {

        for (int i = 0; i < blockDim.x; i += blockDim.x) {
            for (int j = 0; j < blockDim.y; j += 8) {
                if ((row + j) < a.rows && (tx + k + i) < a.cols) {
                    a_smem[ty + j][tx + i] =
                        *(a.data + (row + j) * a.stride + (tx + k + i));
                } else {
                    a_smem[ty + j][tx + i] = 0.f;
                }

                if ((ty + k + i) < b.rows && (col + j) < b.cols) {
                    b_smem[ty + i][tx + j] =
                        *(b.data + (ty + k + i) * b.stride + col + j);
                } else {
                    b_smem[ty + i][tx + j] = 0.f;
                }
            }
        }

        __syncthreads();

        for (int kk = 0; kk < BK; kk += 8) {
            for (int i = 0; i < blockDim.x; i += 8) {
                for (int j = 0; j < blockDim.y; j += 8) {
                    for (int ks = 0; ks < 8; ks++) {
                        partial_acc[j][i] +=
                            a_smem[ty + j][kk + ks] * b_smem[kk + ks][tx + i];
                    }
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < blockDim.x; i += 8) {
        for (int j = 0; j < blockDim.y; j += 8) {
            if ((row + j) < d.rows && (col + i) < d.cols) {
                *(d.data + (row + j) * d.stride + (col + i)) =
                    partial_acc[j][i];
            }
        }
    }
}
} // namespace

void GEMM_v1(const DeviceMatrix &a, const DeviceMatrix &b, DeviceMatrix &d) {
    dim3 threads(32, 16);
    dim3 blocks(cuda::ceil_div(d.cols, threads.x),
                cuda::ceil_div(d.rows, threads.y));
    gemm_impl_v1<<<blocks, threads>>>(a, b, d);
    cudaDeviceSynchronize();
}

void GEMM_v2(const DeviceMatrix &a, const DeviceMatrix &b, DeviceMatrix &d) {
    dim3 threads(256);
    dim3 blocks(d.cols, d.rows);
    gemm_impl_v2<<<blocks, threads>>>(a, b, d);
    cudaDeviceSynchronize();
}

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

void GEMM_v4(const DeviceMatrix &a, const DeviceMatrix &b, DeviceMatrix &d) {
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 32;
    dim3 threads(TILE_M / 4, TILE_N / 4);
    dim3 blocks(cuda::ceil_div(d.cols, threads.x),
                cuda::ceil_div(d.rows, threads.y));
    gemm_impl_v4<TILE_M, TILE_N, TILE_K><<<blocks, threads>>>(a, b, d);
    cudaDeviceSynchronize();
}
