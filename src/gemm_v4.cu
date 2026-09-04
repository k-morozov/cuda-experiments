#include "gemm.h"

#include <cstdio>

#include <cuda/cmath>

namespace {

template <int BLOCK_TILE_M, int BLOCK_TILE_N, int BLOCK_TILE_K, int THREADS_M,
          int THREADS_N>
__global__ void gemm_impl(const DeviceMatrix a, const DeviceMatrix b,
                          DeviceMatrix d) {

    constexpr int SUBBLOCKS_M = BLOCK_TILE_M / THREADS_M;
    constexpr int SUBBLOCKS_N = BLOCK_TILE_N / THREADS_N;

    // 32 * 8
    __shared__ __half a_smem[BLOCK_TILE_M][BLOCK_TILE_K];

    // 8 * 32
    __shared__ __half b_smem[BLOCK_TILE_K][BLOCK_TILE_N];

    const int bix = blockIdx.x;
    const int biy = blockIdx.y;

    // 0..8
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int outx = bix * BLOCK_TILE_N + tx;
    const int outy = biy * BLOCK_TILE_M + ty;

    // BLOCK_TILE_M / threads count

    __half subblock_acc[SUBBLOCKS_M][SUBBLOCKS_N]{};

    for (int k = 0; k < a.cols; k += BLOCK_TILE_K) {

        for (int j = 0; j < SUBBLOCKS_M; j++) {
            if ((outy + j * THREADS_M) < a.rows && (tx + k) < a.cols) {
                a_smem[ty + j * THREADS_M][tx] =
                    *(a.data + (outy + j * THREADS_M) * a.stride + (tx + k));
                ;
            } else {
                a_smem[ty + j * THREADS_M][tx] = 0.f;
            }
        }
        for (int i = 0; i < SUBBLOCKS_N; i++) {
            if ((ty + k) < b.rows && (outx + i * THREADS_N) < b.cols) {
                b_smem[ty][tx + i * THREADS_N] =
                    *(b.data + (ty + k) * b.stride + (outx + i * THREADS_N));
            } else {
                b_smem[ty][tx + i * THREADS_N] = 0.f;
            }
        }

        __syncthreads();

        for (int i = 0; i < SUBBLOCKS_N; i++) {
            for (int j = 0; j < SUBBLOCKS_M; j++) {
                for (int ks = 0; ks < BLOCK_TILE_K; ks++) {
                    subblock_acc[j][i] += a_smem[ty + j * THREADS_M][ks] *
                                          b_smem[ks][tx + i * THREADS_N];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < SUBBLOCKS_N; i++) {
        for (int j = 0; j < SUBBLOCKS_M; j++) {

            if ((outy + j * THREADS_M) < d.rows &&
                (outx + i * THREADS_N) < d.cols) {
                *(d.data + (outy + j * THREADS_M) * d.stride + outx +
                  i * THREADS_N) = subblock_acc[j][i];
            }
        }
    }
}

} // namespace

void GEMM_v4(const DeviceMatrix &a, const DeviceMatrix &b, DeviceMatrix &d) {
    constexpr int BLOCK_TILE_M = 32;
    constexpr int BLOCK_TILE_N = 32;
    constexpr int TILE_K = 8;

    const int THREADS_M = 8;
    const int THREADS_N = 8;

    // const int SUBBLCOKS_M = BLOCK_TILE_M / THREADS_M;
    // const int SUBBLCOKS_N = BLOCK_TILE_M / THREADS_N;

    dim3 threads(THREADS_N, THREADS_M);
    dim3 blocks(cuda::ceil_div(d.cols, BLOCK_TILE_N),
                cuda::ceil_div(d.rows, BLOCK_TILE_M));

    gemm_impl<BLOCK_TILE_M, BLOCK_TILE_N, TILE_K, THREADS_M, THREADS_N>
        <<<blocks, threads>>>(a, b, d);
    cudaDeviceSynchronize();
}
