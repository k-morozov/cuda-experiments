#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <vector>

#include "gemm.h"

// Host-side counterpart of DeviceMatrix: owns __half data on the host,
// can upload itself to the GPU, and supports multiplication so it can be
// used to compute an expected result for correctness tests.
class HostMatrix {
public:
    HostMatrix(size_t rows, size_t cols, size_t stride)
        : rows_(rows), cols_(cols), stride_(stride), data_(rows * stride) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t stride() const { return stride_; }

    __half &at(size_t row, size_t col) { return data_[row * stride_ + col]; }
    __half at(size_t row, size_t col) const { return data_[row * stride_ + col]; }

    DeviceMatrix TransformToGpu() const {
        DeviceMatrix device{nullptr, rows_, cols_, stride_};
        size_t bytes = rows_ * stride_ * sizeof(__half);

        cudaError_t status = cudaMalloc(&device.data, bytes);
        assert(status == cudaSuccess);

        status = cudaMemcpy(device.data, data_.data(), bytes,
                             cudaMemcpyHostToDevice);
        assert(status == cudaSuccess);

        return device;
    }

    friend HostMatrix operator*(const HostMatrix &a, const HostMatrix &b) {
        assert(a.cols_ == b.rows_);

        HostMatrix result(a.rows_, b.cols_, b.cols_);
        for (size_t row = 0; row < a.rows_; row++) {
            for (size_t col = 0; col < b.cols_; col++) {
                float acc = 0.0f;
                for (size_t k = 0; k < a.cols_; k++) {
                    acc += __half2float(a.at(row, k)) * __half2float(b.at(k, col));
                }
                result.at(row, col) = __float2half(acc);
            }
        }
        return result;
    }

private:
    size_t rows_;
    size_t cols_;
    size_t stride_;
    std::vector<__half> data_;
};
