#pragma once

#include "gemm.h"

void RunUniqueDemo(void (*gemm)(const DeviceMatrix &, const DeviceMatrix &,
                                DeviceMatrix &),
                   int n);
