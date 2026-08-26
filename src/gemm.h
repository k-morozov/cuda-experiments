#pragma once

#include <cuda_fp16.h>

struct DeviceMatrix {
    __half *data;
    size_t rows;
    size_t cols;
    size_t stride;
};

void GEMM(const DeviceMatrix &a, const DeviceMatrix &b, const DeviceMatrix &c,
          DeviceMatrix &d);
