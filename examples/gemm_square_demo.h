#pragma once

#include "gemm.h"

void RunSquareDemo(void (*gemm)(const DeviceMatrix &, const DeviceMatrix &,
                                DeviceMatrix &),
                   int n);
